# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from copy import copy

from vajra.models import vajra
from vajra.nn.vajra import SegmentationModel
from vajra.nn.cfsav1 import CFSAv1
from vajra.utils import HYPERPARAMS_CFG_DICT, LOGGER, RANK
from vajra.utils.torch_utils import de_parallel
from vajra.plotting import plot_images, plot_results

class SegmentationTrainer(vajra.detect.DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration=None, _callbacks=None) -> None:
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "segment"
        super().__init__(config, model_configuration, _callbacks)

    def get_model(self, model_name=None, weights=None, verbose=True):
        # A resume path names a checkpoint, not an architecture.
        if weights is not None:
            model_name = getattr(weights, "model_name", model_name)
        options = {key: getattr(self.args, key, HYPERPARAMS_CFG_DICT[key]) for key in (
            "use_cfsav1", "use_cfsa", "cfsav1_groups", "cfsav1_context_channels",
            "cfsav1_rho", "cfsav1_gate_smooth", "cfsav1_global_context",
            "cfsav1_semantic_context", "samx_mode", "samx_offset_lr_scale", "samx_geometry_gain",
            "samx_proto_mode", "samx_proto_channels", "samx_proto_gain",
            "samx_proto_share_coeff", "samx_proto_offset_bound")}
        model = SegmentationModel(model_name, channels=3, num_classes=self.data["nc"],
                                  verbose=verbose and RANK == -1, **options)
        if weights:
            model.load(weights)
        if options["samx_mode"] != "off":
            LOGGER.info(f"SAMX enabled at nano neck3: {options['samx_mode']}, "
                        f"offset lr multiplier={options['samx_offset_lr_scale']}")
        if options["use_cfsav1"]:
            LOGGER.info(f"CFSAv1 enabled (proto-only), "
                        f"gate smoothing={options['cfsav1_gate_smooth']}, "
                        f"global context={options['cfsav1_global_context']}, "
                        f"P4 semantic context={options['cfsav1_semantic_context']}")
        if options["samx_proto_mode"]!="off":
            LOGGER.info(f"P3 Proto geometry: {options['samx_proto_mode']}, "
                        f"hidden={options['samx_proto_channels']}, gain_init={options['samx_proto_gain']}")
            LOGGER.info(f"P3 mask variants: shared coefficients={options['samx_proto_share_coeff']}, "
                        f"offset bound={options['samx_proto_offset_bound']} P3 cells (0=free)")
        return model

    def _model_train(self):
        super()._model_train()
        epochs = getattr(self.args, "cfsav1_warmup_epochs", 0)
        if not epochs:
            return
        model = de_parallel(self.model)
        warmup = self.epoch < epochs
        # Keep optimizer/DDP parameter membership fixed; discard old-parameter
        # gradients at optimizer_step and freeze their running statistics here.
        if warmup:
            model.eval()
            model.model[-1].train()  # retain the training output contract
            for name, module in model.model[-1].named_children():
                if name != "proto_enhance":
                    module.eval()
        self._cfsav1_warmup_active = warmup

    def _update_cfsav1_gate_scale(self, ema_only=False, log=False):
        """Use the absolute epoch on resume; this schedule never freezes weights."""
        if not getattr(self.args, "use_cfsav1", False):
            return
        epochs = getattr(self.args, "cfsav1_gate_ramp_epochs", 0)
        start = getattr(self.args, "cfsav1_gate_ramp_start", 0.1)
        scale = start + (1.0 - start) * min(max(self.epoch, 0) / epochs, 1.0) if epochs else 1.0
        models = [] if ema_only else [de_parallel(self.model)]
        ema = getattr(self, "ema", None)
        if ema is not None:
            models.append(ema.ema)
        for model in models:
            for module in model.modules():
                if isinstance(module, CFSAv1):
                    # AMP validation casts EMA half -> float under inference_mode.
                    # That can replace buffers with inference tensors, which cannot
                    # be filled in normal training mode. Clone only in that case;
                    # assigning an existing buffer keeps it registered/persistent.
                    if module.gate_scale.is_inference():
                        module.gate_scale = module.gate_scale.clone()
                    module.gate_scale.fill_(scale)
        if log and epochs and RANK in (-1, 0):
            LOGGER.info(f"CFSAv1 gate ramp: epoch {self.epoch + 1}, scale={scale:.4f}, "
                        f"effective rho={self.args.cfsav1_rho * scale:.4f}")

    def optimizer_step(self):
        if getattr(self, "_cfsav1_warmup_active", False):
            model = de_parallel(self.model)
            prefix = f"model.{len(model.model) - 1}.proto_enhance."
            # None (not zero) also prevents optimizer momentum/weight decay.
            for name, parameter in model.named_parameters():
                if not name.startswith(prefix):
                    parameter.grad = None
        super().optimizer_step()
        # EMA averages floating buffers too. The schedule is deterministic state,
        # not a learned weight: restore its exact current value after every update.
        self._update_cfsav1_gate_scale(ema_only=True)

    def preprocess_batch(self, batch):
        if getattr(self, "_cfsav1_gate_epoch", None) != self.epoch:
            self._update_cfsav1_gate_scale(log=True)
            self._cfsav1_gate_epoch = self.epoch
        # The base loop calls model.train() directly, so apply the optional
        # warmup mode after that call, once at the first batch of each epoch.
        if getattr(self.args, "cfsav1_warmup_epochs", 0) and getattr(self, "_cfsav1_mode_epoch", None) != self.epoch:
            self._model_train()
            self._cfsav1_mode_epoch = self.epoch
        return super().preprocess_batch(batch)

    def build_dataset(self, img_path, mode="train", batch=None):
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return vajra.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / F"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
    def plot_metrics(self):
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)
