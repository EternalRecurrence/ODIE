"""After-stage live scoring and one-time in-place support/feature activation."""
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from .selection import residual_scores, select_mean_rank, select_benefit_cost


def _tensor_hash(digest, name, value):
    digest.update(name.encode('utf8'))
    digest.update(str((tuple(value.shape), value.dtype)).encode('ascii'))
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())


def _weights_hash(model, excluded_plane=None):
    digest = sha256()
    for name, parameter in model.named_parameters():
        if name == 'grid' and excluded_plane is not None:
            item = model.layout['levels'][excluded_plane]
            start = item['plane_offset']
            stop = start + item['plane_node_capacity'] * item['plane_dim']
            _tensor_hash(digest, 'grid-before-new-plane', parameter[:start])
            _tensor_hash(digest, 'grid-after-new-plane', parameter[stop:])
        else:
            _tensor_hash(digest, name, parameter)
    return digest.hexdigest()


def _optimizer_hash(optimizer):
    if hasattr(optimizer, "native_state_digest"):
        return optimizer.native_state_digest()
    digest = sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        options = {key: value for key, value in group.items() if key != 'params'}
        digest.update(json.dumps(options, sort_keys=True).encode('utf8'))
        for index, parameter in enumerate(group['params']):
            state = optimizer.state.get(parameter, {})
            digest.update(f'{group_index}/{index}'.encode('ascii'))
            for key in sorted(state):
                value = state[key]
                if torch.is_tensor(value):
                    _tensor_hash(digest, key, value)
                else:
                    digest.update(json.dumps({key: value}, sort_keys=True).encode('utf8'))
    return digest.hexdigest()


def after_stage(ctx, policy):
    """Called after previous prefix finishes and before the next prefix opens."""
    started = time.perf_counter()
    level = ctx.next_prefix - 1
    model, record = ctx.model, ctx.support_record
    if ctx.completed_prefix != model.active_prefix or ctx.next_prefix != ctx.completed_prefix + 1:
        raise AssertionError('Dynamic callback must run between consecutive prefixes')
    if ctx.optimizer.state.get(ctx.plane_parameters[level]):
        raise AssertionError('Unopened plane already has optimizer state')
    if any(bool(support.any()) for support in record['supports'][level:]):
        raise AssertionError('Future support must be empty before live selection')
    grid_id, storage_pointer, optimizer_id = id(model.grid), model.grid.data_ptr(), id(ctx.optimizer)
    hash_started = time.perf_counter()
    live_before = _weights_hash(model)
    unchanged_before = _weights_hash(model, excluded_plane=level)
    optimizer_before = _optimizer_hash(ctx.optimizer)
    hash_seconds = time.perf_counter() - hash_started

    resolution = ctx.config['resolutions'][level]
    cap, dim = record['plane_node_caps'][level], record['plane_dims'][level]
    gpu_pipeline = ctx.config.get('region_backend', 'cpu') == 'cuda_device'
    if gpu_pipeline:
        # Stream predictions into GPU scoring without full-image downloads.
        from .gpu_residual import select_live
        metrics, support, selection, counts, gpu_timing, _ = select_live(
            model, ctx.target, ctx.shape, resolution, cap, dim, ctx.config['eval_batch_size'])
        evaluation_seconds = gpu_timing['inference_and_scoring_seconds']
        transfer_seconds, scoring_seconds = 0., None
        selection_seconds = gpu_timing['greedy_seconds']
        hash_started = time.perf_counter()
        live_after_evaluation = _weights_hash(model)
        hash_seconds += time.perf_counter() - hash_started
        if live_before != live_after_evaluation:
            raise AssertionError('GPU residual selection changed live weights')
    else:
        evaluation_started = time.perf_counter()
        metrics, prediction = ctx.evaluate(model, ctx.target, ctx.shape,
                                           ctx.config['eval_batch_size'], return_image=True)
        evaluation_seconds = time.perf_counter() - evaluation_started
        hash_started = time.perf_counter()
        live_after_evaluation = _weights_hash(model)
        hash_seconds += time.perf_counter() - hash_started
        if live_before != live_after_evaluation:
            raise AssertionError('Scoring evaluation changed live weights')

        transfer_started = time.perf_counter()
        target = ctx.target.detach().cpu().numpy()
        transfer_seconds = time.perf_counter() - transfer_started
        scoring_started = time.perf_counter()
        resolution = ctx.config['resolutions'][level]
        mean, total, counts = residual_scores(prediction.numpy(), target, ctx.shape, resolution)
        scoring_seconds = time.perf_counter() - scoring_started
        selection_started = time.perf_counter()
        cap = record['plane_node_caps'][level]
        dim = record['plane_dims'][level]
        if policy == 'B1':
            support, selection = select_mean_rank(mean, resolution, cap)
        elif policy in ('B2', 'E'):
            support, selection = select_benefit_cost(total, resolution, cap, dim, normalize_by_cost=(policy == 'B2'))
        else:
            raise ValueError('Dynamic experiment must be B1, B2 or E')
        selection_seconds = time.perf_counter() - selection_started
    activation_started = time.perf_counter()
    activation = model.activate_support(level, torch.from_numpy(support.copy()))
    record['supports'][level] = torch.from_numpy(support.copy())
    if model.grid.device.type == 'cuda':
        torch.cuda.synchronize(model.grid.device)
    activation_seconds = time.perf_counter() - activation_started
    hash_started = time.perf_counter()
    unchanged_after = _weights_hash(model, excluded_plane=level)
    optimizer_after = _optimizer_hash(ctx.optimizer)
    live_after_activation = _weights_hash(model)
    hash_seconds += time.perf_counter() - hash_started
    if unchanged_before != unchanged_after or optimizer_before != optimizer_after:
        raise AssertionError('Activation changed previous/future weights, heads or optimizer state')
    if id(model.grid) != grid_id or model.grid.data_ptr() != storage_pointer or id(ctx.optimizer) != optimizer_id:
        raise AssertionError('Activation replaced packed parameter storage or optimizer')
    if ctx.optimizer.state.get(ctx.plane_parameters[level]):
        raise AssertionError('Activation must not create new-plane Adam state before training')

    pixels = int(np.prod(ctx.shape))
    event = {
        'policy': policy, 'epoch': int(ctx.epoch) - 1, 'opening_epoch': int(ctx.epoch),
        'completed_updates': int(ctx.updates),
        'completed_prefix': int(ctx.completed_prefix), 'next_prefix': int(ctx.next_prefix),
        'new_plane_zero_based': level, 'resolution': resolution, 'plane_dim': dim,
        'weight_hash_live_before_scoring': live_before,
        'weight_hash_live_after_evaluation': live_after_evaluation,
        'weight_hash_live_after_activation': live_after_activation,
        'unchanged_weights_hash_excluding_new_plane_before': unchanged_before,
        'unchanged_weights_hash_excluding_new_plane_after': unchanged_after,
        'optimizer_state_hash_before': optimizer_before, 'optimizer_state_hash_after': optimizer_after,
        'evaluation_left_live_weights_unchanged': True,
        'previous_planes_and_heads_unchanged': True, 'future_planes_unchanged': True,
        'optimizer_state_unchanged': True, 'packed_parameter_identity_unchanged': True,
        'new_plane_adam_state_not_yet_created': True,
        'no_best_rollback': True, 'no_weight_reset': True, 'no_optimizer_reset': True,
        'score_input': 'current live prefix prediction, unclipped CPU FP32 RGB; original GT FP32',
        'score_reduction': 'FP32 difference/square, FP64 RGB sum/3, FP64 native-cell accumulation',
        'cell_score': 'pixel mean RGB MSE' if policy == 'B1' else 'total pixel mean-RGB squared error',
        'scoring_full_image_prediction_passes': 1,
        'scoring_prediction_forward_batches': math.ceil(pixels / ctx.config['eval_batch_size']),
        'scoring_prediction_pixels': pixels, 'prediction_gpu_to_cpu_copies': math.ceil(pixels / ctx.config['eval_batch_size']),
        'gt_gpu_to_cpu_copies': 1, 'gt_copied_pixels': pixels,
        'gt_pixels_accessed_by_evaluation': pixels, 'gt_pixels_accessed_by_residual_scoring': pixels,
        'gt_pixels_accessed_total': pixels * 2,
        'evaluation_seconds_including_prediction_transfer': evaluation_seconds,
        'prediction_transfer_seconds': None,
        'prediction_transfer_timing_note': 'CPU image transfer occurs inside common evaluate and is included in evaluation time',
        'gt_cpu_transfer_seconds': transfer_seconds, 'scoring_seconds': scoring_seconds,
        'greedy_selection_seconds': selection_seconds, 'activation_seconds': activation_seconds,
        'weight_and_optimizer_hash_seconds': hash_seconds,
        'selected_pixel_coverage': float(counts[support].sum() / pixels),
        'selection': selection, 'activation': activation,
        'provenance': 'live current-prefix RGB residual only; no historical masks, GT highpass or extra teacher',
    }
    event['selection_seconds'] = time.perf_counter() - started
    if gpu_pipeline:
        event.update(score_input='current live prefix GPU FP32 RGB; original GT GPU FP32',
                     prediction_gpu_to_cpu_copies=0, gt_gpu_to_cpu_copies=0, gt_copied_pixels=0,
                     evaluation_seconds_including_prediction_transfer=None,
                     prediction_transfer_seconds=0.,
                     prediction_transfer_timing_note='GPU streaming residual; no prediction or GT CPU copies',
                     gpu_pipeline_timing=gpu_timing)
    timing = None if gpu_pipeline else getattr(ctx.evaluate, 'last_timing', None)
    if timing is not None:
        event['evaluation_timing'] = timing
        event['prediction_cpu_copy_wait_seconds'] = timing.get('prediction_cpu_copy_wait_seconds')
        event['prediction_transfer_timing_note'] = 'copy wall time includes waiting for queued CUDA work; pure transfer is not isolated'
    events = record.setdefault('metadata', {}).setdefault('dynamic_selection_events', [])
    events.append(event)
    print(f'Expert {level+1}: selected coverage {event["selected_pixel_coverage"]:.2%}'
          f' | selection {event["selection_seconds"]:.3f}s',flush=True)
    torch.save(record, Path(ctx.output) / 'support_current.pt')
    print(json.dumps({'dynamic_selection': policy, 'completed_epoch': ctx.epoch - 1, 'opening_epoch': ctx.epoch,
                      'new_prefix': ctx.next_prefix,
                      'live_psnr_before_opening': metrics['psnr_db'], 'selection': selection,
                      'selection_seconds': event['selection_seconds']}), flush=True)
    return event
