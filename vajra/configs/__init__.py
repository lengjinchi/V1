import contextlib
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union
from vajra.utils import (
    ASSETS, 
    HYPERPARAMS_CFG_DICT,
    HYPERPARAMS_CFG,
    HYPERPARAMS_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    SETTINGS,
    SETTINGS_YAML,
    RUNS_DIR,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    colorstr,
    yaml_load,
    yaml_print,
    deprecation_warn
)

modes = {'train', 'val', 'predict', 'export', 'track'}
tasks = {'detect', 'segment', 'classify', 'multilabel_classify', 'pose', 'obb', 'small_obj_detect'}

models_for_tasks = {'detect' : 'vajra-v1-nano-det.pt',
                    'small_obj_detect': 'vajra-v1-nano-pose.pt',
                    'segment' : 'vajra-v1-nano-seg.pt',
                    'classify' : 'vajra-v1-nano-cls.pt',
                    'multilabel_classify': 'vajra-v1-nano-cls.pt',
                    'pose' : 'vajra-v1-nano-pose.pt',
                    'obb': 'vajra-v1-nano-obb.pt'}

metrics_for_tasks = {'detect' : 'metrics/mAP50-95(Box)',
                     'small_obj_detect': 'metrics/mAP50-95(Box)',
                     'segment' : 'metrics/mAP50-95(Mask)',
                     'classify' : 'metrics/accuracy_top1',
                     'multilabel_classify': 'metrics/accuracy_top1',
                     'pose': 'metrics/mAP50-95(Pose)',
                     'obb': 'metrics/mAP50-95(Box)'}

data_for_tasks = {'detect' : 'coco8.yaml',
                  'small_obj_detect': 'coco8.yaml',
                  'segment': 'coco8-seg.yaml',
                  'classify': 'imagenet10',
                  'multilabel_classify': 'imagenet10',
                  'pose': 'coco8-pose.yaml',
                  'obb':'dota8.yaml'}

CLI_HELP_MESSAGE = ' '

CFG_INT_KEYS = {'epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 
                'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'workspace', 
                'nominal_batch_size', 'save_period'}

CFG_FLOAT_KEYS = {'box', 'cls', 'seg', 'dfl', 'degrees', 'shear', 'time', 'warmup_epochs'}

CFG_FRACTION_KEYS = {'dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 
                     'warmup_momentum', 'warmup_bias_lr', 'label_smoothing', 
                     'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 
                     'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 
                     'mixup', 'copy_paste', 'conf', 'fraction'}

CFG_BOOL_KEYS = {'save', 'exist_ok', 'verbose', 'deterministic', 'single_cls',
                 'rect', 'cos_lr', 'overlap_mask', 'val', 'save_json', 
                 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt',
                 'save_conf', 'save_crop', 'save_frames', 'show_labels',
                 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks',
                 'show_boxes', 'keras', 'optimize', 'int8', 'dynamic', 'simplify',
                 'nms', 'profile', 'multi_scale'}

CFG_BOOL_KEYS.update({'use_cfsa', 'use_cfsav1', 'cfsav1_gate_smooth', 'cfsav1_global_context',
                      'cfsav1_semantic_context', 'samx_proto_share_coeff'})
CFG_FLOAT_KEYS.add('samx_proto_offset_bound')
CFG_FLOAT_KEYS.update({'mabd_v1_boundary_base', 'mabd_v1_boundary_gain', 'mabd_v1_boundary_cap'})
CFG_INT_KEYS.update({'cfsav1_groups', 'cfsav1_context_channels',
                     'cfsav1_warmup_epochs', 'cfsav1_gate_ramp_epochs', 'samx_proto_channels',
                     'mabd_v1_boundary_radius'})
CFG_FRACTION_KEYS.update({'cfsav1_rho', 'cfsav1_gate_ramp_start', 'samx_offset_lr_scale',
                          'samx_geometry_gain', 'samx_proto_gain'})

MODELS = frozenset({models_for_tasks[task] for task in tasks})


def config_to_dict(config):
    if isinstance(config, (str, Path)):
        #if Path(config).suffix == '.py':
        #    config, _ = Config._file2dict(config)
        #else:
        config = yaml_load(config)
    elif isinstance(config, SimpleNamespace):
        config = vars(config) # Converts to dict

    return config

def _clean_retired_cfsav1_options(config):
    """Read disabled options in historical checkpoints without re-enabling removed training."""
    config = dict(config)
    if config.get('cfsav1_paired') or config.get('cfsav1_sensitivity') or config.get('cfsav1_reference'):
        raise ValueError('CFSAv1 paired views and reference supervision have been removed. '
                         'Use single-view training; enabled legacy runs cannot be resumed with this code.')
    for key in ('cfsav1_paired', 'cfsav1_sensitivity', 'cfsav1_reference',
                'cfsav1_sensitivity_gain', 'cfsav1_perturb_strength', 'cfsav1_tau_scale',
                'cfsav1_calibration_batches'):
        config.pop(key, None)
    return config


def get_config(config: Union[str, Path, Dict, SimpleNamespace] = HYPERPARAMS_CFG_DICT, model_configuration: Dict = None):
    config = _clean_retired_cfsav1_options(config_to_dict(config))

    if model_configuration:
        model_configuration = _clean_retired_cfsav1_options(config_to_dict(model_configuration))
        if "save_dir" not in config:
            model_configuration.pop("save_dir", None)
        check_dict_alignment(config, model_configuration)
        config = {**config, **model_configuration}

    for k in "project", "name":
        if k in config and isinstance(config[k], (int, float)):
            config[k] = str(config[k])
    if config.get("name") == "model":
        config["name"] = config.get("model", "").split(".")[0]
        LOGGER.warning(f'WARNING! "name = model" automatically updated to "name = {config["name"]}".')

    check_config(config)

    return IterableSimpleNamespace(**config)

def check_config(config, strict=True):
    for k, v in config.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'Valid {k} types are either int (i.e {k} = 0) or float (i.e {k} = 0.5)')
                config[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if strict:
                        raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                        f'Valid {k} types are int (i.e {k} = 0) or float (i.e {k} = 0.5)')
                    config[k] = v = float(v)

                if not (0.0 <= v <= 1.0):
                    raise ValueError(f'{k}={v} is an invalid value'
                                     f'Valid {k} values are between 0.0 and 1.0')
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'{k} must be an int (i.e {k} = 1)')
                config[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if strict:
                    raise TypeError(f'{k}={v} is of invalid type {type(v).__name__}'
                                    f'{k} must be a bool (i.e {k} = True or {k} = False)')
                config[k] = bool(v)

    if config.get('use_cfsa') and config.get('use_cfsav1'):
        raise ValueError('use_cfsa and use_cfsav1 are mutually exclusive')
    if config.get('mask_loss_mode', 'bce') not in ('bce', 'mabd', 'mabd_v1', 'mabd_v1_weights', 'mabd_v1_boundary'):
        raise ValueError('mask_loss_mode must be bce, mabd, mabd_v1, mabd_v1_weights or mabd_v1_boundary')
    radius = config.get('mabd_v1_boundary_radius', 1)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError('mabd_v1_boundary_radius must be a positive integer')
    for key, default in (('mabd_v1_boundary_base', .05), ('mabd_v1_boundary_gain', .15),
                         ('mabd_v1_boundary_cap', .20)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
    if not 0 < config.get('mabd_v1_boundary_cap', .20) <= 1:
        raise ValueError('mabd_v1_boundary_cap must be in (0, 1]')
    if config.get('mabd_v1_boundary_base', .05) > config.get('mabd_v1_boundary_cap', .20):
        raise ValueError('mabd_v1_boundary_base must not exceed boundary cap')
    if config.get('samx_mode', 'off') not in ('off', 'original', 'stage_global', 'local_global', 'geometry_residual', 'geometry_dual', 'branch_evidence'):
        raise ValueError('Unknown samx_mode')
    if not 0 < config.get('samx_offset_lr_scale', 0.1) <= 1:
        raise ValueError('samx_offset_lr_scale must be in (0, 1]')
    if isinstance(config.get('samx_geometry_gain', 0.1), bool) or not 0 < config.get('samx_geometry_gain', 0.1) <= 1:
        raise ValueError('samx_geometry_gain must be in (0, 1]')
    if config.get('samx_proto_mode','off') not in ('off','free','structured'):
        raise ValueError('samx_proto_mode must be off, free or structured')
    if config.get('samx_proto_mode','off')!='off' and (config.get('use_cfsa') or config.get('use_cfsav1')):
        raise ValueError('Proto geometry requires CFSA and CFSAv1 disabled')
    if isinstance(config.get('samx_proto_channels',16),bool) or config.get('samx_proto_channels',16)<=0:
        raise ValueError('samx_proto_channels must be a positive integer')
    if isinstance(config.get('samx_proto_gain',.1),bool) or not 0<config.get('samx_proto_gain',.1)<=1:
        raise ValueError('samx_proto_gain must be in (0,1]')
    bound = config.get('samx_proto_offset_bound', 0.0)
    if isinstance(bound, bool) or not math.isfinite(bound) or bound < 0:
        raise ValueError('samx_proto_offset_bound must be finite and nonnegative')
    if (config.get('samx_proto_share_coeff', False) or bound) and config.get('samx_proto_mode', 'off') != 'free':
        raise ValueError('Proto coefficient sharing and bounded offsets require samx_proto_mode="free"')
    if config.get('cfsav1_semantic_context') and not config.get('use_cfsav1'):
        raise ValueError('cfsav1_semantic_context requires use_cfsav1=True')
    for key in ('cfsav1_groups', 'cfsav1_context_channels'):
        if key in config and (isinstance(config[key], bool) or config[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    if 'cfsav1_rho' in config and not 0 < config['cfsav1_rho'] < 1:
        raise ValueError('cfsav1_rho must be strictly between 0 and 1')
    if config.get('cfsav1_warmup_epochs', 0) < 0:
        raise ValueError('cfsav1_warmup_epochs must be non-negative')
    if config.get('cfsav1_warmup_epochs') and not config.get('use_cfsav1'):
        raise ValueError('cfsav1_warmup_epochs requires use_cfsav1=True')
    ramp_epochs = config.get('cfsav1_gate_ramp_epochs', 0)
    if isinstance(ramp_epochs, bool) or not isinstance(ramp_epochs, int) or ramp_epochs < 0:
        raise ValueError('cfsav1_gate_ramp_epochs must be a non-negative integer')
    ramp_start = config.get('cfsav1_gate_ramp_start', 0.1)
    if isinstance(ramp_start, bool) or not isinstance(ramp_start, (int, float)) or not 0 < ramp_start <= 1:
        raise ValueError('cfsav1_gate_ramp_start must be in (0, 1]')
    if ramp_epochs and not config.get('use_cfsav1'):
        raise ValueError('cfsav1_gate_ramp_epochs requires use_cfsav1=True')
                

def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from vajra.utils.files import increment_path
        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)

def _handle_deprecation(custom):
    """Hardcoded function to handle deprecated config keys."""

    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")

    return custom

def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list. If
    any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
        e (Error, optional): An optional error that is passed by the calling function.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MESSAGE) from e

def copy_hyps_config():
    new_file = Path.cwd() / HYPERPARAMS_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(HYPERPARAMS_CFG_PATH, new_file)
    LOGGER.info(
        f"{HYPERPARAMS_CFG_PATH} copied to {new_file}\n"
        f"Example command with this new config:\n   vajra config={new_file} img_size=640 batch=16"
    )
