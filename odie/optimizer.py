"""Persistent per-plane Adam state and the declared full-pass prefix schedule."""
import torch


def validate_schedule(config):
    unlock, joint = int(config['unlock_epochs']), int(config['joint_epochs'])
    stages = [int(value) for value in config.get('stage_epochs', [])]
    initial = int(config.get('initial_active_prefix', 2))
    if not 2 <= initial <= int(config['n_levels']):
        raise ValueError('Initial prefix must be between 2 and declared level count independently of gate K')
    if unlock < 0 or joint <= 0 or unlock + joint != int(config['final_epochs']):
        raise ValueError('Unlock and joint epochs must sum to final_epochs')
    if unlock and (len(stages) != int(config['n_levels']) - initial or min(stages) <= 0 or sum(stages) != unlock):
        raise ValueError('Expected one positive stage duration per sparse level')
    if not unlock and stages:
        raise ValueError('All-layer training must not declare unlock stages')
    return unlock, joint, stages


def active_prefix_for_epoch(epoch, config):
    if not 1 <= epoch <= int(config['final_epochs']):
        raise ValueError('Expected a one-based full pass within the declared schedule')
    if epoch > int(config['unlock_epochs']):
        return int(config['n_levels'])
    boundary = 0
    for prefix, passes in enumerate(config['stage_epochs'], int(config.get('initial_active_prefix', 2))):
        boundary += int(passes)
        if epoch <= boundary:
            return prefix
    raise ValueError('Incomplete prefix schedule')


def make_optimizer(model, config, eps):
    planes = []
    for level in model.layout['levels']:
        offset = level['plane_offset']
        length = level['plane_node_capacity'] * level['plane_dim']
        planes.append(torch.nn.Parameter(model.grid[offset:offset + length].detach()))
    groups = [{'name': 'grid', 'params': planes, 'lr': config['grid_lr']}]
    if model.gate is not None:
        groups.append({'name': 'gate', 'params': list(model.gate.parameters()), 'lr': config['gate_lr']})
    groups.append({'name': 'decoder', 'params': list(model.decoder.parameters()), 'lr': config['decoder_lr']})
    optimizer = torch.optim.Adam(groups, betas=(.9, .999), eps=eps, weight_decay=0., foreach=False)
    return optimizer, planes


def assign_plane_gradients(model, planes):
    for index, (parameter, level) in enumerate(zip(planes, model.layout['levels'])):
        offset = level['plane_offset']
        parameter.grad = (model.grid.grad[offset:offset + parameter.numel()]
                          if index < model.active_prefix else None)


class ReferenceOptimizer:
    """Autograd reference using PyTorch Adam with the same model.

    This is a correctness/readability reference, not a speed-equivalent native Adam.
    """
    recaptures = 0

    def __init__(self, model, target, shape, config):
        self.model, self.target, self.shape = model, target, shape
        self.optimizer, self.planes = make_optimizer(model, config, eps=1e-15)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def prepare_epoch(self):
        pass

    def release_graph(self):
        pass

    def train_batch(self, ids):
        from .evaluation import coordinates
        self.optimizer.zero_grad(set_to_none=True)
        self.model.grid.grad = None
        prediction = self.model(coordinates(self.shape, ids)).float()
        loss = (prediction - self.target[ids]).square().mean()
        (loss * 128.).backward()
        assign_plane_gradients(self.model, self.planes)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.div_(128.)
        self.optimizer.step()
        return loss.detach()

    def close(self):
        pass


def make_training_optimizer(model, target, shape, config):
    if config['backend'] == 'native':
        from .native_backend import make_native_optimizer
        return make_native_optimizer(model, target, shape, config)
    optimizer = ReferenceOptimizer(model, target, shape, config)
    return optimizer, optimizer.planes
