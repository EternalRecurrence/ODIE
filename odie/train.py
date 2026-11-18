"""Single progressive training loop for all configured images and backends."""
import json
import math
from pathlib import Path
import time
import sys
from types import SimpleNamespace

from . import runtime_env
from .optimizer import make_training_optimizer
from .model import build_model
from . import regions
from .evaluation import coordinates, evaluate
from .data import load_target, shuffled_batches

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from scripts._common.io import update_run

from .optimizer import active_prefix_for_epoch, validate_schedule
from .schedule import apply_learning_rates, validate_lr_schedule
from .support_codec import STORAGE_VERSION, encode_supports, restore_support_record, support_storage_statistics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def copy_parameters(model, state):
    with torch.no_grad():
        actual = dict(model.named_parameters())
        if set(actual) != set(state):
            raise ValueError('Saved parameter names do not match the model')
        for name, parameter in actual.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def snapshot(model, dtype=None):
    return {name: parameter.detach().cpu().to(dtype=dtype or parameter.dtype).clone()
            for name, parameter in model.named_parameters()}


def construct_model(builder, config, record, args):
    return builder(config, record['supports'], record['plane_dims'], record['plane_node_caps']).cuda()


def support_payload(record):
    return {key: value for key, value in record.items() if key != 'supports'}


def same_support_value(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_support_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_support_value(a, b) for a, b in zip(left, right))
    return left == right


def fit(args):
    process_started = time.perf_counter()
    config = args.configuration
    unlock_epochs, joint_epochs, stage_epochs = validate_schedule(config)
    validate_lr_schedule(config)
    torch.set_num_threads(1)
    training_stream = torch.cuda.Stream()
    training_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(training_stream)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    support_record = args.support_record
    builder = build_model

    target, shape = load_target(args.input, config['target_storage'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model = construct_model(builder, config, support_record, args)
    params = list(model.parameters())
    if not params or any(p.dtype != torch.float32 for p in params):
        raise ValueError('All trainable masters must remain FP32')
    model.set_active_prefix(active_prefix_for_epoch(1, config))
    assert args.loss_scale == 128. and args.adam_eps == 1e-15
    optimizer, plane_parameters = make_training_optimizer(model, target, shape, config)
    if config['backend'] == 'native':
        update_run(root,native_build=optimizer.build_manifest)
    actual_parameters = sum(p.numel() for p in params)
    if actual_parameters != args.parameter_cap:
        raise ValueError(f'Parameter allocation {actual_parameters} differs from matched target {args.parameter_cap}')
    if actual_parameters * 2 > args.fp16_parameter_budget:
        raise ValueError('FP16 parameter payload exceeds the parameter-byte budget')
    sampling = torch.Generator(device='cpu' if config['sampling'] == 'global_full' else 'cuda').manual_seed(config['seed'])
    epochs, batch = int(config['final_epochs']), int(config['batch_size'])
    updates = processed = evaluation_pixels = 0
    loop_seconds = 0.
    history = []
    learning_rate_history = []
    settings = {'plane_dims': support_record['plane_dims'],
                'plane_node_caps': support_record['plane_node_caps'],
                'axis_mode': args.axis_mode, 'kernel_a': args.kernel_a,
                'kernel_omega': args.kernel_omega, 'external_loss_scale': args.loss_scale,
                'adam_eps': args.adam_eps, 'sampling_seed': config['seed'],
                'per_batch_finite_gradient_check': False, 'representation': config['representation'],
                'unlock_epochs': unlock_epochs, 'joint_epochs': joint_epochs, 'passes_per_prefix': stage_epochs,
                'prefix_schedule': list(range(int(config.get('initial_active_prefix', 2)), int(config['n_levels']))) if unlock_epochs else [], 'joint_prefix': int(config['n_levels']),
                'initial_active_prefix': int(config.get('initial_active_prefix', 2)),
                'gate_coarse_levels_K': int(config['coarse_levels']),
                'router_enabled': config.get('router_enabled', True),
                'learning_rate_schedule': config['learning_rate_schedule'],
                'full_cover_coarse_layers': config.get('full_cover_coarse_layers', [0, 1]),
                'full_cover_detection': config.get('full_cover_detection', []),
                'support_policy': config.get('support_policy', 'predeclared masks held fixed throughout training; see support metadata'),
                'model_options': config.get('model_options', {}),
                'region_selection': config['dynamic_policy'],
                'grid_adam_state': ('TCNN persistent per-parameter state; zero-gradient skip; new parameters start at age zero'
                                    if config['backend'] == 'native' else 'PyTorch Adam; persistent per-plane state'),
                'training_backend': config['backend'],
                'optimizer_change': 'TCNN Adam and reference PyTorch Adam differ in epsilon placement and per-parameter zero-gradient handling',
                'target_storage': config['target_storage'], 'sampling': config['sampling'],
                'permutation_device': str(sampling.device),
                'tail_batch': ('fixed full random batch; no traversal tail' if config['sampling']=='stratified' else
                               'repeat beginning of this epoch permutation' if config['sampling']=='global_full' else 'short')}
    if config['sampling'] == 'stratified':
        settings.update(sampling_rule='one random pixel per square spatial stratum per step',
                        strata_per_axis=int(batch ** .5),
                        epoch_semantics='ceil(image pixels / batch size) steps; sampling with replacement',
                        permutation_device=None)
    update_run(root,settings=settings)
    torch.cuda.synchronize()
    fit_started = time.perf_counter()
    first_sample_ids = None
    ctx = SimpleNamespace(model=model, target=target, shape=shape, config=config,
                          support_record=support_record, optimizer=optimizer,
                          plane_parameters=plane_parameters, output=root, epoch=0, updates=0,
                          completed_prefix=None, next_prefix=None, evaluate=evaluate, coordinates=coordinates)
    interactive = sys.stderr.isatty()
    progress = tqdm(total=epochs, desc='Training', unit='epoch', disable=not interactive)
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        lr_record = apply_learning_rates(optimizer, config, epoch)
        learning_rate_history.append(lr_record)
        prefix = active_prefix_for_epoch(epoch, config)
        ctx.epoch, ctx.updates = epoch, updates
        if prefix != model.active_prefix:
            optimizer.release_graph()
            ctx.completed_prefix, ctx.next_prefix = model.active_prefix, prefix
            regions.after_stage(ctx, config['dynamic_policy'])
            model.set_active_prefix(prefix)
            tqdm.write(f'Opened expert {prefix}/{config["n_levels"]} at epoch {epoch}')
        torch.cuda.synchronize()
        loop_started = time.perf_counter()
        optimizer.prepare_epoch()
        epoch_loss = torch.zeros((), dtype=torch.float64, device='cuda')
        epoch_pixels = 0
        for ids in shuffled_batches(target.shape[0], batch, sampling, config['sampling'], config['chunk_pixels'], shape=shape):
            if first_sample_ids is None:
                first_sample_ids = ids[:10].cpu().tolist()
            ctx.updates = updates
            loss = optimizer.train_batch(ids)
            updates += 1
            processed += ids.numel()
            epoch_pixels += ids.numel()
            epoch_loss += loss.detach().double() * ids.numel()
        torch.cuda.synchronize()
        loop_seconds += time.perf_counter() - loop_started
        if epoch % config['eval_interval'] == 0 or epoch == epochs:
            metrics, _ = evaluate(model, target, shape, config['eval_batch_size'])
            evaluation_pixels += target.shape[0]
            entry = {'epoch': epoch, 'active_prefix': prefix,
                     'training_phase': 'unlock' if epoch <= unlock_epochs else 'joint',
                     'actual_updates': updates, 'rgb_processed_pixels': processed,
                     'train_mean_mse': float(epoch_loss.item() / epoch_pixels), **metrics}
            entry.update(learning_rate_factor=lr_record['factor'], learning_rates_by_group=lr_record['learning_rates_by_group'])
            history.append(entry)
            tqdm.write(f"Eval epoch {epoch} | PSNR {metrics['psnr_db']} dB")
        elapsed = time.perf_counter() - fit_started
        loss_value = float(epoch_loss.item() / epoch_pixels)
        if interactive:
            progress.set_postfix(steps=updates, experts=prefix, loss=f'{loss_value:.6g}')
            progress.update()
        else:
            print(f'Epoch {epoch}/{epochs} | steps {updates}/{epochs * math.ceil(target.shape[0]/batch)}'
                  f' | experts {prefix}/{config["n_levels"]} | loss {loss_value:.6g}'
                  f' | elapsed {elapsed:.1f}s | ETA {elapsed/epoch*(epochs-epoch):.1f}s',flush=True)
    torch.cuda.synchronize()
    training_loop_seconds = time.perf_counter() - training_started
    progress.close()
    native_graph_recaptures = optimizer.recaptures
    optimizer.close()
    last, _ = evaluate(model, target, shape, config['eval_batch_size'])
    evaluation_pixels += target.shape[0]
    inference_weights = snapshot(model, torch.float16)
    # Export only the final state, then verify the exact FP16 inference payload.
    copy_parameters(model, inference_weights)
    exported, image = evaluate(model, target, shape, config['eval_batch_size'], True)
    evaluation_pixels += target.shape[0]
    support_compression_started = time.perf_counter()
    support_encodings = encode_supports(support_record['supports'])
    support_compression_seconds = time.perf_counter() - support_compression_started
    artifact = {'version': 'packed-capacity-control-v1', 'configuration': config,
                'settings': settings, 'support_storage_version': STORAGE_VERSION,
                'support_encodings': support_encodings,
                'support_num_cells': [s.numel() for s in support_record['supports']],
                'support_payload': support_payload(support_record),
                'support_metadata': support_record.get('metadata', {}),
                'parameters': inference_weights, 'checkpoint': 'latest',
                'epoch': epochs, 'optimizer_steps': updates, 'metrics': exported}
    torch.save(artifact, root / 'model_latest_fp16.pt')
    reloaded = torch.load(root / 'model_latest_fp16.pt', map_location='cpu', weights_only=True)
    if reloaded['configuration'] != config:
        raise AssertionError('Saved configuration, including gate K, changed during round trip')
    restored_record = restore_support_record(reloaded)
    if not same_support_value(restored_record, support_record):
        raise AssertionError('Saved support bitmap or extra support payload changed during round trip')
    fresh = construct_model(builder, reloaded['configuration'], restored_record, args)
    if fresh.gate_coarse_levels_K != int(reloaded['configuration']['gate_coarse_levels_K']):
        raise AssertionError('Reloaded model did not reconstruct the serialized gate K')
    fresh.set_active_prefix(int(config['n_levels']))
    if hasattr(fresh, 'set_training_update'):
        fresh.set_training_update(updates)
    copy_parameters(fresh, reloaded['parameters'])
    reloaded_metrics, _ = evaluate(fresh, target, shape, config['eval_batch_size'])
    evaluation_pixels += target.shape[0]
    if reloaded_metrics != exported:
        raise AssertionError('Saved FP16 parameter round trip changed reconstruction metrics')
    values = image.reshape(*shape, 3).numpy()
    Image.fromarray(np.rint(np.clip(values, 0., 1.) * 255).astype(np.uint8)).save(root / 'reconstruction.png')
    if config['save_npy']:
        np.save(root / 'reconstruction.npy', values, allow_pickle=False)
    if config['ssim'] != 'off':
        from .metrics import image_metrics
        result = image_metrics(args.input, root / ('reconstruction.npy' if config['ssim'] == 'raw' and config['save_npy'] else 'reconstruction.png'),
                               config['ssim_stripe_rows'], prediction_array=values if config['ssim'] == 'raw' else None)
        write_json(root / 'image_metrics.json', result)
    torch.cuda.synchronize()
    stats = model.storage_stats()
    selection_events = support_record.get('metadata', {}).get('dynamic_selection_events', [])
    selection_keys = ['selection_seconds', 'scoring_prediction_pixels', 'scoring_full_image_prediction_passes',
                      'scoring_prediction_forward_batches', 'gt_pixels_accessed_total',
                      'gt_pixels_accessed_by_evaluation', 'gt_pixels_accessed_by_residual_scoring', 'gt_copied_pixels']
    selection_totals = {key: sum(event.get(key, 0) for event in selection_events) for key in selection_keys}
    additional_prediction_pixels = selection_totals['scoring_prediction_pixels']
    metrics = {'run_id': args.run_id, 'configuration': config, 'settings': settings,
               'shape_hw': list(shape), 'actual_updates': updates,
               'rgb_processed_pixels': processed, 'evaluation_pixels': evaluation_pixels,
               'hook_prediction_pixels': additional_prediction_pixels,
               'total_evaluation_and_hook_prediction_pixels': evaluation_pixels + additional_prediction_pixels,
               'selection_totals': selection_totals, 'selection_event_count': len(selection_events),
               'selection_events': selection_events,
               'checkpoint': 'latest', 'epoch': epochs,
               'final_rgb_metrics': exported, 'last_before_fp16_payload_metrics': last,
               'allocated_parameter_count': actual_parameters,
               'fp16_parameter_payload_bytes': actual_parameters * 2,
               'inference_package_bytes': (root / 'model_latest_fp16.pt').stat().st_size,
               'support_storage': support_storage_statistics(support_encodings),
               'support_compression_seconds': support_compression_seconds,
               'budget_basis': 'native FP16 parameter payload cap; count and complete file bytes separately reported',
               'fp16_parameter_budget_bytes': args.fp16_parameter_budget,
               'matched_parameter_target': args.parameter_cap,
               'optimization_loop_seconds': loop_seconds,
               'training_loop_seconds': training_loop_seconds,
               'training_time_scope': 'Entire training loop, including sampling, online region allocation and periodic evaluation; excludes setup and post-loop evaluation/export/readback',
               'native_graph_recaptures': native_graph_recaptures,
               'fit_wall_seconds': time.perf_counter() - fit_started,
               'process_wall_seconds': time.perf_counter() - process_started,
               'elapsed_seconds': time.perf_counter() - fit_started,
               'storage': stats, 'history': history,
               'learning_rate_history': learning_rate_history,
               'final_learning_rates_by_group': {group['name']: float(group['lr']) for group in optimizer.param_groups},
               'first_sample_ids': first_sample_ids,
               'support_provenance': support_record.get('metadata', {}),
               'metric_contract': 'Unclipped FP32 RGB differences; FP64 sum; endpoint coordinates',
               'prepared_mask_budget_note': support_record.get('metadata', {}),
               'runtime_index_bytes_are_not_trainable_parameters': True,
               'saved_fp16_payload_and_bitmap_roundtrip_verified': True,
               'saved_configuration_and_gate_K_roundtrip_verified': True,
               'early_stop': 'off', 'periodic_checkpoints': False}
    expected_pixels = updates * batch if config['sampling'] in ('global_full', 'stratified') else epochs * target.shape[0]
    if updates != epochs * ((target.shape[0] + batch - 1) // batch) or processed != expected_pixels:
        raise AssertionError('Successful RGB budget differs from the predeclared budget')
    write_json(root / 'metrics.json', metrics)
    (root/'support_current.pt').unlink(missing_ok=True)
    print(json.dumps({'run_id': args.run_id, 'checkpoint': 'latest', 'psnr': exported['psnr_db'],
                      'seconds': metrics['training_loop_seconds']}), flush=True)

