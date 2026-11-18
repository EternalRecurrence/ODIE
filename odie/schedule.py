"""Epoch schedules shared by native training and the reference model."""
import math

GROUP_KEYS = {'grid': 'grid_lr', 'gate': 'gate_lr', 'decoder': 'decoder_lr'}


def validate_lr_schedule(config):
    schedule = config['learning_rate_schedule']
    final = config['final_epochs']
    if schedule['type'] not in ('constant', 'late_cosine'):
        raise ValueError('LR type must be constant or late_cosine')
    if schedule['type'] == 'late_cosine':
        start = schedule['constant_through_epoch']
        if type(start) is not int or not 0 <= start < final:
            raise ValueError('Cosine start must be an integer in [0, epochs)')
        if schedule.get('final_epoch', final) != final:
            raise ValueError('LR final_epoch must equal final_epochs')
        if not 0 < schedule['minimum_factor'] <= 1:
            raise ValueError('minimum_factor must be in (0, 1]')
    for key in GROUP_KEYS.values():
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')


def factor_for_epoch(epoch, config):
    if type(epoch) is not int or not 1 <= epoch <= config['final_epochs']:
        raise ValueError('Epoch must be within the declared training schedule')
    schedule = config['learning_rate_schedule']
    if schedule['type'] == 'constant':
        return 1.
    start = schedule['constant_through_epoch']
    minimum = schedule['minimum_factor']
    if epoch <= start:
        return 1.
    return minimum + (1 - minimum) * .5 * (1 + math.cos(
        math.pi * (epoch - start) / (config['final_epochs'] - start)))


def apply_learning_rates(optimizer, config, epoch):
    factor = factor_for_epoch(epoch, config)
    expected = set(GROUP_KEYS) if config.get('router_enabled', True) else {'grid', 'decoder'}
    if {g['name'] for g in optimizer.param_groups} != expected:
        raise ValueError(f'Expected optimizer groups: {sorted(expected)}')
    rates = {}
    for group in optimizer.param_groups:
        group['lr'] = float(config[GROUP_KEYS[group['name']]]) * factor
        rates[group['name']] = group['lr']
    return dict(epoch=epoch, factor=factor, learning_rates_by_group=rates)
