"""Representative-day clustering adapted from the original thesis experiments."""

import math
import numpy as np
import pandas as pd


def standardize_matrix(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X - mean) / std, mean, std


def pairwise_euclidean_distance_matrix(X):
    X = np.asarray(X, dtype=float)
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    dist_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def initialize_kmedoids_plus_plus(D, k, rng):
    """Distance-based k-medoids++ style initialization."""
    n = D.shape[0]
    medoids = [int(rng.integers(0, n))]

    while len(medoids) < k:
        min_dist = np.min(D[:, medoids], axis=1)
        min_dist[medoids] = 0.0
        weights = min_dist**2
        total = weights.sum()

        if total <= 1e-12:
            remaining = [i for i in range(n) if i not in medoids]
            medoids.append(int(rng.choice(remaining)))
        else:
            probs = weights / total
            medoids.append(int(rng.choice(np.arange(n), p=probs)))

    return np.array(medoids, dtype=int)


def assign_points_to_medoids(D, medoids):
    distances_to_medoids = D[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    # Each distinct medoid represents its own cluster, including tied profiles.
    labels[medoids] = np.arange(len(medoids))
    assigned_distances = distances_to_medoids[np.arange(D.shape[0]), labels]
    total_cost = float(assigned_distances.sum())
    return labels, assigned_distances, total_cost


def update_medoids(D, labels, medoids, k):
    n = D.shape[0]
    new_medoids = medoids.copy()

    # Current distances are useful for empty-cluster repair.
    _, assigned_distances, _ = assign_points_to_medoids(D, medoids)

    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]

        if len(members) == 0:
            # Repair empty cluster by selecting the currently worst represented non-medoid point.
            non_medoids = np.array(
                [i for i in range(n) if i not in set(new_medoids)], dtype=int
            )
            if len(non_medoids) == 0:
                continue
            worst_idx = non_medoids[np.argmax(assigned_distances[non_medoids])]
            new_medoids[cluster_id] = int(worst_idx)
            continue

        subD = D[np.ix_(members, members)]
        within_cost = subD.sum(axis=1)
        best_member = members[np.argmin(within_cost)]
        new_medoids[cluster_id] = int(best_member)

    return new_medoids


def run_kmedoids(D, k, n_init=50, max_iter=100, random_seed=42):
    """
    Simple alternate-optimization k-medoids implementation.

    It is not a full PAM swap implementation, but it is robust enough for
    yearly representative-day selection with a small number of days.
    """

    n = D.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > n:
        raise ValueError(f"k={k} cannot be larger than number of days n={n}.")

    rng = np.random.default_rng(random_seed)

    best = None
    log_rows = []

    for init_id in range(1, n_init + 1):
        medoids = initialize_kmedoids_plus_plus(D, k, rng)
        previous_cost = math.inf

        for iteration in range(1, max_iter + 1):
            labels, assigned_distances, cost = assign_points_to_medoids(D, medoids)
            updated_medoids = update_medoids(D, labels, medoids, k)
            updated_labels, updated_distances, updated_cost = assign_points_to_medoids(
                D, updated_medoids
            )

            medoids_changed = not np.array_equal(
                np.sort(medoids), np.sort(updated_medoids)
            )
            improvement = previous_cost - updated_cost

            medoids = updated_medoids
            labels = updated_labels
            assigned_distances = updated_distances
            cost = updated_cost

            if abs(improvement) < 1e-10 or not medoids_changed:
                break

            previous_cost = cost

        labels, assigned_distances, cost = assign_points_to_medoids(D, medoids)

        log_rows.append(
            {
                "init_id": init_id,
                "iterations": iteration,
                "final_cost": cost,
                "mean_assigned_distance": float(assigned_distances.mean()),
                "max_assigned_distance": float(assigned_distances.max()),
            }
        )

        if best is None or cost < best["cost"]:
            best = {
                "medoids": medoids.copy(),
                "labels": labels.copy(),
                "assigned_distances": assigned_distances.copy(),
                "cost": cost,
                "init_id": init_id,
                "iterations": iteration,
            }

    return best, pd.DataFrame(log_rows)


def remap_clusters_by_medoid_day(labels, medoid_indices, days):
    """Returns stable cluster ids sorted by medoid day."""
    medoid_day_pairs = []
    for old_cluster_id, medoid_idx in enumerate(medoid_indices):
        medoid_day_pairs.append((old_cluster_id, days[medoid_idx]))

    medoid_day_pairs = sorted(medoid_day_pairs, key=lambda x: x[1])
    old_to_new = {old: new_id + 1 for new_id, (old, _) in enumerate(medoid_day_pairs)}

    new_labels = np.array([old_to_new[int(label)] for label in labels], dtype=int)

    medoid_rows = []
    for old_cluster_id, medoid_day in medoid_day_pairs:
        medoid_idx = int(medoid_indices[old_cluster_id])
        medoid_rows.append(
            {
                "cluster_id": old_to_new[old_cluster_id],
                "old_cluster_id": int(old_cluster_id),
                "medoid_index": medoid_idx,
                "representative_day": str(days[medoid_idx]),
            }
        )

    return new_labels, pd.DataFrame(medoid_rows)
