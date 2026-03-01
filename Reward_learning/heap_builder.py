import numpy as np
from reward_utils import get_indices


def build_tournament_heap(dataset, traj_total, config):
    """
    Forest of Tournament Max-Heaps construction.

    Builds multiple independent heaps (forest) to increase sample diversity.
    Each heap: divide segments into groups of K, Oracle picks the best,
    winners advance to next level. Segments that don't fill a complete
    group get a "bye" (advance directly).

    Args:
        dataset: dict with "rewards", "observations", "actions"
        traj_total: total number of trajectories in dataset
        config: TrainConfig with q_budget (K), feedback_num, segment_size,
                threshold, num_heaps

    Returns:
        all_groups: list of (parent_start, parent_return, child_starts, child_returns)
        total_queries: int, total oracle queries used across all heaps
    """
    K = config.q_budget
    segment_size = config.segment_size
    threshold = config.threshold
    budget = config.feedback_num
    num_heaps = getattr(config, "num_heaps", 5)

    queries_per_heap = budget // num_heaps
    pool_size_per_heap = queries_per_heap * (K - 1)

    # Dataset size validation
    max_segments = len(dataset["rewards"]) - segment_size
    assert pool_size_per_heap <= max_segments, (
        f"Dataset too small: need pool_size={pool_size_per_heap} but only "
        f"{max_segments} possible segments. Reduce feedback_num or K."
    )

    all_groups = []
    total_queries = 0

    for heap_idx in range(num_heaps):
        # Sample unique segment starts for this heap
        pool = []
        seen = set()
        while len(pool) < pool_size_per_heap:
            idx = get_indices(traj_total, config)
            k = idx[0][0]
            if k in seen:
                continue
            seen.add(k)
            pool.append(k)

        # Precompute returns for all segments in pool
        returns_cache = {}
        for k in pool:
            seg_idx = list(range(k, k + segment_size))
            returns_cache[k] = float(np.sum(dataset["rewards"][seg_idx]))

        heap_queries = 0
        current_level = list(pool)

        while len(current_level) >= K and heap_queries < queries_per_heap:
            np.random.shuffle(current_level)

            num_matches = len(current_level) // K
            byes = current_level[num_matches * K:]  # remainder gets a bye
            competing = current_level[: num_matches * K]

            winners = []
            for g in range(num_matches):
                if heap_queries >= queries_per_heap:
                    # Unprocessed segments don't form groups — just stop
                    break

                group_starts = competing[g * K: (g + 1) * K]
                group_returns = [returns_cache[s] for s in group_starts]

                # Oracle: pick best (handle ties with threshold)
                gap = segment_size * threshold
                mx = max(group_returns)
                best_cand = [
                    i for i, r in enumerate(group_returns) if abs(r - mx) <= gap
                ]
                best_idx = int(np.random.choice(best_cand))

                parent_start = group_starts[best_idx]
                parent_return = group_returns[best_idx]
                child_starts = [
                    group_starts[i] for i in range(K) if i != best_idx
                ]
                child_returns = [
                    group_returns[i] for i in range(K) if i != best_idx
                ]

                all_groups.append(
                    (parent_start, parent_return, child_starts, child_returns)
                )
                winners.append(parent_start)
                heap_queries += 1

            # Next level: winners + byes
            current_level = winners + byes

        total_queries += heap_queries

    return all_groups, total_queries
