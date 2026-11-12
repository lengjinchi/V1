"""Fixed configurations: each follow-up has a predeclared comparison."""
from samx_experiment_config import ROOT, training_options as earlier_options

AUDIT = ROOT/'output/samx_followups_20261029'
MODES = ('dual_geometry','proto_free','proto_structured')
EXPERIMENTS = {
    'dual_geometry':'SAMX_dualGeometry_P4_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261029',
    'proto_free':'SAMX_localGlobalP4_protoFree_P3_c16_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261029',
    'proto_structured':'SAMX_localGlobalP4_protoStructured_P3_c16_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261029',
}
BASELINE_RESULTS = {
    'local_global':ROOT/'output/samx_experiments_20261027/local_global_result.json',
    'geometry_shared':ROOT/'output/samx_geometry_20261028/train_result.json',
}


def options(mode,smoke=False):
    if mode not in MODES:
        raise ValueError(mode)
    args = earlier_options('geometry_residual' if mode=='dual_geometry' else 'local_global',smoke=smoke)
    args.update(samx_mode='geometry_dual' if mode=='dual_geometry' else 'local_global',
                samx_proto_mode={'dual_geometry':'off','proto_free':'free','proto_structured':'structured'}[mode],
                samx_proto_channels=16,samx_proto_gain=.1,
                project=str(AUDIT/'smoke') if smoke else str(ROOT/'SAMX compare'),
                name=('smoke_'+mode) if smoke else EXPERIMENTS[mode],exist_ok=False)
    return args
