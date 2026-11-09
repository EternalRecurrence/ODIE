"""Capacity checks and configuration for existing official training entries.

Training stays in the vendored entry. This module only constructs the same
official model on CPU and resolves experiment overrides.
"""
from copy import deepcopy
from pathlib import Path
import sys

from .io import sha


RGB_PRETRAINING_CONFIG = 'configs/configs4paper/ablation_pretraining/done/combined_CE_pretraining.yaml'


def neural_config(repo, width):
    import yaml
    config = yaml.safe_load((repo/'configs/config_RGB.yaml').read_text())
    # Preserve the RGB entry's 80% joint / 20% expert-only fitting stages.
    config['MODEL']['decoder_hidden_dim'] = width
    config['DATA'].update(downsample_factor=1, keep_original_shape=True)
    return config


def neural_pretraining_config(repo, width, image, seed):
    """Run only the official RGB combined-CE segmentation stage, at matched width."""
    import yaml
    config = yaml.safe_load((repo/'configs/config_RGB.yaml').read_text())
    pretraining = yaml.safe_load((repo/RGB_PRETRAINING_CONFIG).read_text())
    # Older paper configs omit fields now required by the pinned RGB loss
    # constructor (e.g. rgbrecon_metric); inherit their official RGB defaults.
    for key, value in pretraining.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    stage = deepcopy(config['TRAINING']['stages'][0])
    if stage['loss_type'] != '1000segmentation' or stage['params'] != 'all':
        raise ValueError('Official RGB pretraining stage changed')
    # The official 0.1 fraction of 30000 iterations gives 3000 segmentation updates.
    epochs = round(config['TRAINING']['num_epochs'] * stage['end_iteration_frac'])
    if epochs < 1:
        raise ValueError('Empty RGB pretraining stage')
    stage['end_iteration_frac'] = 1.0
    config['seed'] = seed
    config['MODEL'].update(decoder_hidden_dim=width, load_pt_manager=False)
    config['DATA']['dataset_path'] = str(Path(image).parent)
    config['TRAINING'].update(num_epochs=epochs, stages=[stage])
    return config


def model_for_plan(plan, repo):
    """Return an official CPU model and the config passed to its native entry."""
    sys.path.insert(0, str(repo))
    method = plan['method']
    if method == 'siren':
        import modules
        # The official constructor moves sidelength to CUDA even with its
        # default downsample=False. It is unused in this MLP mode; omit it
        # for CPU capacity checks. The training entry still supplies it.
        model = modules.SingleBVPNet(out_features=3, type='sine', mode='mlp',
            hidden_features=plan['allocation']['width'])
        return model, None
    if method == 'finer':
        from models import Finer
        settings = deepcopy(plan['model'])
        return Finer(**settings), None
    if method == 'neural_experts':
        from models import build_model
        config = neural_config(repo, plan['allocation']['expert_width'])
        return build_model(config, config['LOSS'])[0], config
    raise ValueError(method)


def check_pretrained(model, checkpoint):
    """Require a complete state matching the configured upstream model, including experts."""
    import torch
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return dict(path=str(checkpoint), sha256=sha(checkpoint),
        load_scope='Complete official model state, including both encoders, router and RGB experts',
        loaded_parameter_count=sum(value.numel() for value in state.values()))
