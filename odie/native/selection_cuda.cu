// Exact CUDA greedy support selection for progressive expert activation.
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

constexpr int THREADS = 256;
constexpr int FAN = 32;

__device__ uint64_t score_key(int id, const double* scores, int cost) {
    return id < 0 || cost == 0 ? 0 :
        (uint64_t(__double_as_longlong(scores[id])) & 0x7fffffffffffffffull);
}

__device__ bool better(int a, uint64_t av, int b, uint64_t bv, int cost) {
    if (a < 0) return false;
    if (b < 0) return true;
    return cost == 0 ? a < b : (av > bv || (av == bv && a < b));
}

__device__ int warp_best(int id, const double* scores, int cost) {
    // Nonnegative finite FP64 values have the same ordering as their bit patterns.
    // Clear the sign bit so -0 and +0 tie exactly as in Python. No quantization.
    uint64_t value = score_key(id, scores, cost);
    int lane = threadIdx.x % FAN;
    unsigned mask = __activemask();
    for (int step = FAN / 2; step; step /= 2) {
        int other = __shfl_down_sync(mask, id, step, FAN);
        uint64_t ov = __shfl_down_sync(mask, value, step, FAN);
        if (lane + step < FAN && better(other, ov, id, value, cost)) {
            id = other;
            value = ov;
        }
    }
    return id;
}

__global__ void init_state(int n, int points, int* missing, uint8_t* chosen,
                           uint8_t* nodes, int64_t* stats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { missing[i] = 4; chosen[i] = 0; }
    if (i < points) nodes[i] = 0;
    if (i < 4) stats[i] = 0; // used, iterations, done, invariant error
}

__global__ void build_level(const double* scores, const int* missing,
                            const uint8_t* chosen, int* tree,
                            int offset, int previous_offset,
                            int children, int parents, bool leaves) {
    int task = (blockIdx.x * blockDim.x + threadIdx.x) / FAN;
    int parent = task / 5, cost = task % 5;
    if (parent >= parents) return; // entire warp takes the same branch
    int child = parent * FAN + (threadIdx.x % FAN);
    int id = -1;
    if (child < children) {
        id = leaves ? ((!chosen[child] && missing[child] == cost) ? child : -1)
                    : tree[previous_offset + child * 5 + cost];
    }
    id = warp_best(id, scores, cost);
    if ((threadIdx.x % FAN) == 0) tree[offset + parent * 5 + cost] = id;
}

// One persistent block: exact sequential choices, parallel local tree repairs.
// Bounded chunks let other GPU work run between launches.
__global__ void greedy(const double* scores, int n, int points, int resolution,
                        int cap, bool normalized, int* missing, uint8_t* chosen,
                        uint8_t* nodes, int* tree, const int* offsets,
                        const int* sizes, int levels, int chunk, int64_t* stats, int* trace) {
    __shared__ int cell, cost, added[4], nadded, affected[17], naffected;
    __shared__ int tasks[40], next_tasks[40], proposals[40], next_proposals[40];
    __shared__ int changed_tasks[40], changed_ids[40], old_missing[17], ntasks, stop;
    int tid = threadIdx.x, lane = tid % FAN, warp = tid / FAN;
    int deltas[4] = {0, 1, resolution, resolution + 1};
    for (int iteration = 0; iteration < chunk; ++iteration) {
        if (tid == 0) {
            int root = offsets[levels - 1];
            cell = tree[root]; cost = 0;
            if (cell < 0) {
                int remaining = cap - int(stats[0]);
                for (int k = 1; k <= 4 && k <= remaining; ++k) {
                    int candidate = tree[root + k];
                    if (candidate < 0) continue;
                    double left = normalized && cell >= 0
                        ? __dmul_rn(scores[candidate], double(cost)) : scores[candidate];
                    double right = cell < 0 ? 0. : (normalized
                        ? __dmul_rn(scores[cell], double(k)) : scores[cell]);
                    if (cell < 0 || left > right || (left == right && candidate < cell)) {
                        cell = candidate; cost = k;
                    }
                }
            }
            stop = cell < 0;
            if (stop) stats[2] = 1;
            if (!stop) {
                chosen[cell] = 1;
                nadded = 0;
                for (int d = 0; d < 4; ++d) {
                    int node = (cell + deltas[d]) % points;
                    if (!nodes[node]) { nodes[node] = 1; added[nadded++] = node; }
                }
                if (nadded != cost) { stats[3] = 1; stop = 1; }
                if (trace) trace[stats[1]] = cell;
                stats[0] += nadded; ++stats[1];
                naffected = 1; affected[0] = cell;
                for (int a = 0; a < nadded; ++a) {
                    for (int d = 0; d < 4; ++d) {
                        int id = (added[a] - deltas[d] + points) % points;
                        if (id >= n || chosen[id]) continue;
                        bool found = false;
                        for (int j = 0; j < naffected; ++j) found |= affected[j] == id;
                        if (!found) affected[naffected++] = id;
                    }
                }
            }
        }
        __syncthreads();
        if (stop) return;
        // Each affected cell has one writer; shared corners reduce cost multiple times.
        if (tid < naffected) {
            int id = affected[tid];
            old_missing[tid] = missing[id];
            if (id != cell) {
                int decrement = 0;
                for (int a = 0; a < nadded; ++a)
                    for (int d = 0; d < 4; ++d)
                        decrement += ((id + deltas[d]) % points == added[a]);
                missing[id] -= decrement;
            }
        }
        __syncthreads();
        if (tid == 0) {
            ntasks = 0;
            for (int j = 0; j < naffected; ++j) {
                int id = affected[j], parent = id / FAN;
                for (int which = 0; which < (id == cell ? 1 : 2); ++which) {
                    int category = which ? missing[id] : old_missing[j];
                    int task = parent * 5 + category, proposal = which ? id : -1;
                    // Removing a non-winner cannot change this tree node.
                    if (!which && tree[offsets[0] + task] != id) continue;
                    int index = 0;
                    while (index < ntasks && tasks[index] != task) ++index;
                    if (index == ntasks) {
                        tasks[ntasks] = task; proposals[ntasks++] = proposal;
                    } else if (better(proposal, score_key(proposal,scores,category),
                                      proposals[index], score_key(proposals[index],scores,category),category)) {
                        proposals[index] = proposal;
                    }
                }
            }
        }
        __syncthreads();
        for (int level = 0; level < levels; ++level) {
            for (int task = warp; task < ntasks; task += THREADS / FAN) {
                int parent = tasks[task] / 5, category = tasks[task] % 5;
                int address = offsets[level] + parent * 5 + category;
                int previous = tree[address], proposal = proposals[task], id = previous;
                if (previous >= 0 && (chosen[previous] || missing[previous] != category)) {
                    // Only an invalidated winner requires a scan of its children.
                    int child = parent * FAN + lane;
                    id = -1;
                    if (level == 0) {
                        if (child < n && !chosen[child] && missing[child] == category) id = child;
                    } else if (child < sizes[level - 1]) {
                        id = tree[offsets[level - 1] + child * 5 + category];
                    }
                    id = warp_best(id, scores, category);
                } else if (better(proposal,score_key(proposal,scores,category),
                                  previous,score_key(previous,scores,category),category)) {
                    id = proposal;
                }
                if (lane == 0) {
                    changed_tasks[task] = previous != id;
                    changed_ids[task] = id;
                    tree[address] = id;
                }
            }
            __syncthreads();
            if (tid == 0 && level + 1 < levels) {
                int count = 0;
                for (int j = 0; j < ntasks; ++j) {
                    if (!changed_tasks[j]) continue;
                    int next = (tasks[j] / 5 / FAN) * 5 + tasks[j] % 5;
                    int proposal = changed_ids[j], category = next % 5;
                    int index = 0;
                    while (index < count && next_tasks[index] != next) ++index;
                    if (index == count) {
                        next_tasks[count] = next; next_proposals[count++] = proposal;
                    } else if (better(proposal,score_key(proposal,scores,category),
                                      next_proposals[index],score_key(next_proposals[index],scores,category),category)) {
                        next_proposals[index] = proposal;
                    }
                }
                ntasks = count;
                for (int j = 0; j < count; ++j) {
                    tasks[j] = next_tasks[j]; proposals[j] = next_proposals[j];
                }
            }
            __syncthreads();
            if (!ntasks) break;
        }
    }
}

EXPORT int selection_init(const double* scores, int n, int points,
    int* missing, uint8_t* chosen, uint8_t* nodes, int* tree, int64_t* stats,
    const int* host_offsets, const int* host_sizes, int levels, uint64_t stream_bits) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_bits);
    init_state<<<(points + 255) / 256, 256, 0, stream>>>(n, points, missing, chosen, nodes, stats);
    for (int l = 0; l < levels; ++l) {
        int children = l == 0 ? n : host_sizes[l - 1];
        int tasks = host_sizes[l] * 5;
        build_level<<<(tasks + THREADS / FAN - 1) / (THREADS / FAN), THREADS, 0, stream>>>(scores, missing, chosen, tree,
            host_offsets[l], l ? host_offsets[l - 1] : 0, children, host_sizes[l], l == 0);
    }
    return int(cudaGetLastError());
}

EXPORT int selection_chunk(const double* scores, int n, int points, int resolution,
    int cap, bool normalized, int* missing, uint8_t* chosen, uint8_t* nodes,
    int* tree, const int* offsets, const int* sizes, int levels, int chunk,
    int64_t* stats, int* trace, uint64_t stream_bits) {
    greedy<<<1, THREADS, 0, reinterpret_cast<cudaStream_t>(stream_bits)>>>(scores, n,
        points, resolution, cap, normalized, missing, chosen, nodes, tree, offsets,
        sizes, levels, chunk, stats, trace);
    return int(cudaGetLastError());
}
