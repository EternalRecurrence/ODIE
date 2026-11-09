"""CPU live-residual allocation using native query cells and complete node closure."""
from functools import lru_cache
import heapq
import struct
import numpy as np


def align8(value):
    return (int(value) + 7) // 8 * 8


def _float_from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


@lru_cache(maxsize=128)
def native_thresholds(resolution):
    """Compute native FMA cell thresholds without Torch or configuration dependencies.
    Binary-search FP32 bits against exact fused-result rounding boundaries.
    """
    if type(resolution) is not int or not 2 <= resolution <= 2**23:
        raise ValueError('Resolution must be an integer between 2 and 2^23')
    scale = resolution - 1
    thresholds = []
    for region in range(1, resolution):
        region_bits = struct.unpack('<I', struct.pack('<f', region))[0]
        previous_num, previous_den = _float_from_bits(region_bits - 1).as_integer_ratio()
        boundary_num = (region - 1) * previous_den + previous_num
        boundary_den = 2 * previous_den
        include_tie = region_bits % 2 == 0
        lower, upper = 0, 0x3F800000
        while lower < upper:
            middle = (lower + upper) // 2
            numerator, denominator = _float_from_bits(middle).as_integer_ratio()
            left = scale * numerator * boundary_den
            right = boundary_num * denominator
            if left > right or (left == right and include_tie):
                upper = middle
            else:
                lower = middle + 1
        thresholds.append(_float_from_bits(lower))
    return tuple(thresholds)


def pixel_cells(shape, resolution):
    """Match FP32 endpoint coordinates and torch.bucketize(right=True)."""
    height, width = shape
    if min(height, width) < 2:
        raise ValueError('Endpoint image axes must each contain at least two pixels')
    thresholds = np.asarray(native_thresholds(resolution), dtype=np.float32)
    x = np.arange(width, dtype=np.float32) / np.float32(width - 1)
    y = np.arange(height, dtype=np.float32) / np.float32(height - 1)
    xb = np.searchsorted(thresholds, x, side='right')
    yb = np.searchsorted(thresholds, y, side='right')
    return (yb[:, None] * resolution + xb[None, :]).reshape(-1)


def residual_scores(prediction, target, shape, resolution):
    """FP32 RGB differences/squares followed by FP64 pixel/cell summation."""
    prediction = np.asarray(prediction, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    if prediction.shape != target.shape or target.shape[0] != int(np.prod(shape)):
        raise ValueError('Prediction and target must match the complete image')
    difference = prediction - target
    squared = difference * difference
    per_pixel = squared.sum(axis=1, dtype=np.float64) / 3.
    if not np.isfinite(per_pixel).all():
        raise FloatingPointError('Live RGB residual is nonfinite')
    cells = pixel_cells(shape, resolution)
    total = np.bincount(cells, weights=per_pixel, minlength=resolution**2)
    counts = np.bincount(cells, minlength=resolution**2)
    mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
    return mean, total, counts


def closure_nodes(cells, resolution):
    cells = np.asarray(cells, dtype=np.int64)
    return np.unique(np.concatenate([cells, cells + 1, cells + resolution,
                                     cells + resolution + 1]) % align8(resolution**2))


def _validate(scores, resolution, cap):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (resolution**2,) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Scores must be a finite nonnegative R^2 vector')
    if type(cap) is not int or cap % 8 or not 0 <= cap <= align8(resolution**2):
        raise ValueError('Node cap must be 8-aligned and within the native plane')
    return scores


def _result(selected, nodes, resolution, cap, policy, iterations):
    closure = closure_nodes(np.flatnonzero(selected), resolution)
    if not np.array_equal(closure, np.flatnonzero(nodes)) or len(closure) > cap:
        raise AssertionError('Selection must preserve its complete node closure')
    return selected, {
        'policy': policy, 'selected_cells': int(selected.sum()),
        'closure_nodes': int(nodes.sum()), 'node_capacity': cap,
        'unused_allocated_node_slots': int(cap - nodes.sum()),
        'aligned_used_node_slots': align8(nodes.sum()),
        'node_capacity_utilization': float(nodes.sum() / cap) if cap else 0.,
        'aligned_capacity_utilization': float(align8(nodes.sum()) / cap) if cap else 0.,
        'complete_interpolation_closure': True,
        'tie_order': 'descending score, ascending native cell ID',
        'iterations': iterations,
    }


def select_mean_rank(mean, resolution, cap):
    """B1 original region-mean ranking, admitting only closures within node cap."""
    scores = _validate(mean, resolution, cap)
    points = align8(resolution**2)
    selected = np.zeros(resolution**2, dtype=bool)
    nodes = np.zeros(points, dtype=bool)
    used = 0
    # Stable descending sort keeps initial ascending IDs for exact score ties.
    ranking = np.argsort(-scores, kind='stable')
    for cell in ranking:
        corners = (cell + np.array([0, 1, resolution, resolution + 1])) % points
        added = corners[~nodes[corners]]
        if used + len(added) <= cap:
            selected[cell] = True
            nodes[added] = True
            used += len(added)
    return _result(selected, nodes, resolution, cap,
                   'B1 live RGB region mean; stable cell ranking adapted to complete node cap',
                   len(ranking))


def select_benefit_cost(benefit, resolution, cap, plane_dim=2, normalize_by_cost=True):
    """B2 exact greedy total residual/new closure-node parameter cost.

    Five lazy priority heaps correspond to current missing corner count 0..4.
    Adding a node immediately updates all its incident candidates. Infeasible
    candidates remain eligible because shared corners can reduce future cost.
    Zero-cost candidates are taken first, in ascending ID, including zero benefit.
    """
    scores = _validate(benefit, resolution, cap)
    if plane_dim not in (1, 2, 4):
        raise ValueError('Plane dimension must be one, two or four')
    cells, points = resolution**2, align8(resolution**2)
    selected = np.zeros(cells, dtype=bool)
    nodes = np.zeros(points, dtype=bool)
    missing = np.full(cells, 4, dtype=np.uint8)
    heaps = [[] for _ in range(5)]
    heaps[4] = [(-float(value), cell) for cell, value in enumerate(scores)]
    heapq.heapify(heaps[4])
    used = iterations = 0
    deltas = (0, 1, resolution, resolution + 1)

    def clean(cost):
        heap = heaps[cost]
        while heap and (selected[heap[0][1]] or int(missing[heap[0][1]]) != cost):
            heapq.heappop(heap)
        return heap[0] if heap else None

    while True:
        free = clean(0)
        if free is not None:
            # Free heap stores (ID, ID), so benefit does not affect admission.
            cell, cost = free[1], 0
        else:
            best = None
            for cost in range(1, min(4, cap - used) + 1):
                candidate = clean(cost)
                if candidate is None:
                    continue
                candidate_score, cell = -candidate[0], candidate[1]
                if best is None:
                    best = (candidate_score, cost, cell)
                else:
                    # Common plane_dim cancels; cross products avoid division.
                    left, right = ((candidate_score * best[1], best[0] * cost)
                                   if normalize_by_cost else (candidate_score, best[0]))
                    if left > right or (left == right and cell < best[2]):
                        best = (candidate_score, cost, cell)
            if best is None:
                break
            _, cost, cell = best
        heapq.heappop(heaps[cost])
        selected[cell] = True
        corners = [(cell + delta) % points for delta in deltas]
        added = [node for node in corners if not nodes[node]]
        if len(added) != cost:
            raise AssertionError('Exact marginal closure cost is stale')
        nodes[added] = True
        used += cost
        iterations += 1
        changed = set()
        for node in added:
            for delta in deltas:
                incident = (node - delta) % points
                if incident < cells and not selected[incident]:
                    missing[incident] -= 1
                    changed.add(incident)
        for incident in changed:
            cost_now = int(missing[incident])
            key = incident if cost_now == 0 else -float(scores[incident])
            heapq.heappush(heaps[cost_now], (key, incident))
    support, record = _result(selected, nodes, resolution, cap,
                             ('B2 live RGB residual total / exact newly added closure-node parameters'
                              if normalize_by_cost else 'E live RGB residual total; exact node capacity constraint'),
                             iterations)
    record.update(plane_dim=plane_dim, normalize_by_cost=normalize_by_cost,
                  marginal_parameter_cost='new unique closure nodes times plane dimension',
                  cost_update='exact after each selected cell; all incident candidates updated',
                  zero_cost_order='free cells admitted first, ascending cell ID, including zero benefit')
    return support, record
