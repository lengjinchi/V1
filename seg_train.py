from vajra import Vajra
from samx_experiment_config import clear_validation_cache, training_options

# original: repaired original, retained but not queued.
# stage_global: experiment 2, independent stage global gates.
# local_global: experiment 3, original shared gate + local context.
# geometry_residual: new P4 geometric difference residual, gamma init 0.1.
SAMX_MODE = "geometry_residual"
USE_CFSA = False
USE_CFSAV1 = False


def main():
    options = training_options(SAMX_MODE)
    options.update(use_cfsa=USE_CFSA, use_cfsav1=USE_CFSAV1)
    model = Vajra("vajra-v1-nano-seg")
    model.add_callback("on_val_start", clear_validation_cache)
    results = model.train(**options)
    metrics = model.val(data=options["data"], img_size=1600, rect=True,
                        batch=8, workers=0, device=0, half=False,
                        conf=0.001, iou=0.7, max_det=300, augment=False)
    return results, metrics


if __name__ == "__main__":
    main()
