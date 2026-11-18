"""No-on-demand model using the official TCNN HashGrid, with optional routing."""
from copy import deepcopy

import torch
from torch import nn


class HashGrid(nn.Module):
    def __init__(self, config, supports, plane_dims, plane_node_caps):
        super().__init__()
        if supports or config['initial_active_prefix'] != config['n_levels']:
            raise ValueError('HashGrid requires no supports and all levels active')
        import tinycudann as tcnn
        self.config = deepcopy(config)
        self.router_enabled = config['router_enabled']
        self.n_levels = self.active_prefix = config['n_levels']
        self.gate_coarse_levels_K = config['gate_coarse_levels_K']
        self.input_width = sum(plane_dims)
        self.plane_dims, self.plane_node_caps = list(plane_dims), list(plane_node_caps)
        self.supports = []
        self.register_buffer('feature_levels', torch.tensor(
            [level for level, dim in enumerate(plane_dims) for _ in range(dim)],
            dtype=torch.int64, device='cuda'), persistent=False)
        # Use TCNN's own initialization for both modules on its default stream.
        stream = torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            self.encoding = tcnn.Encoding(2, config['hash_encoding'], seed=config['seed'], dtype=torch.float16)
            self.gate = (tcnn.Network(self.input_width, self.n_levels,
                dict(otype='FullyFusedMLP', activation='ReLU', output_activation='None',
                     n_neurons=32, n_hidden_layers=1), seed=(config['seed']+2000003) % 2**31)
                if self.router_enabled else None)
            self.decoder = tcnn.Network(self.input_width, 3,
                dict(otype='FullyFusedMLP', activation='ReLU', output_activation='None',
                     n_neurons=64, n_hidden_layers=2), seed=(config['seed']+4000037) % 2**31)
        torch.cuda.current_stream().wait_stream(stream)
        offset, levels = 0, []
        for dim, cap in zip(plane_dims, plane_node_caps):
            levels.append(dict(plane_offset=offset, plane_node_capacity=cap, plane_dim=dim))
            offset += dim*cap
        self.layout = dict(levels=levels)
        if (self.grid.numel() != offset or self.decoder.params.numel() != config['decoder_parameters'] or
                (self.gate.params.numel() if self.gate is not None else 0) != config['gate_parameters']):
            raise RuntimeError('Actual TCNN allocation differs from the reference HashGrid budget')

    @property
    def grid(self):
        return self.encoding.params

    def set_active_prefix(self, prefix):
        if prefix != self.n_levels:
            raise ValueError('The no-on-demand model always trains every level')

    def forward(self, coordinates):
        raw = self.encoding(coordinates)
        if self.gate is not None:
            from .model import router_weights
            weights = router_weights(self.gate(raw).float(), self.config['router_output_activation'])
            k = self.gate_coarse_levels_K
            if k:
                weights = torch.cat((torch.ones_like(weights[:, :k]), weights[:, k:]), dim=1)
            raw = raw * weights.to(raw.dtype).index_select(1, self.feature_levels)
        return self.decoder(raw).float()[:, :3]

    def storage_stats(self):
        grid, decoder = self.grid.numel(), self.decoder.params.numel()
        gate = self.gate.params.numel() if self.gate is not None else 0
        heads = gate + decoder
        return dict(actual_params=grid+heads, grid_params=grid, head_params=heads,
            gate_params=gate, decoder_params=decoder, router_enabled=self.router_enabled,
            axis_params=0, plane_params=grid, plane_caps=self.plane_node_caps,
            fp16_parameter_payload_bytes=2*(grid+heads), fp32_master_bytes=4*(grid+heads),
            adam_moment_bytes=8*(grid+heads), full_gradient_bytes=2*(grid+heads),
            index_bytes=0, node_bitmap_bytes=0, node_prefix_bytes=0, bitmap_bytes=0,
            runtime_index_bytes=None, input_width=self.input_width,
            active_layers_from_update1=self.n_levels, active_prefix=self.n_levels,
            full_cover_coarse_layers=list(range(self.n_levels)), plane_dims=self.plane_dims,
            coordinate_version='unit-endpoints-v1', layout_version='official-tcnn-hash-grid-v1',
            addressing_version='official-tcnn-hash-grid', initialization='official TCNN',
            levels=[dict(level=i, resolution=r, **item) for i,(r,item) in
                    enumerate(zip(self.config['resolutions'],self.layout['levels']))])
