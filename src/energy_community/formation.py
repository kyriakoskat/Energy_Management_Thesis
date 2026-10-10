"""Fixed-size grouping: exact lexicographic MILP and multistart swap search."""

from itertools import combinations
import random
import time
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


def group_score(group, scores):
    return float(sum(scores.loc[i, j] for i, j in combinations(group, 2)))


def objective(groups, scores):
    values = [group_score(g, scores) for g in groups]
    return min(values), sum(values)


def canonical(groups):
    return sorted([sorted(g) for g in groups], key=lambda g: g[0])


def is_better(candidate, incumbent, tol=1e-10):
    return candidate[0] > incumbent[0] + tol or (
        abs(candidate[0] - incumbent[0]) <= tol and candidate[1] > incumbent[1] + tol
    )


def exact_grouping(players, scores, group_size, time_limit=120):
    """Maximize minimum group score, then total score subject to that minimum."""
    started = time.monotonic()
    number = len(players) // group_size

    def build(minimum=None, hint=None, seconds=None):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError("SCIP is unavailable")
        solver.SetNumThreads(1)
        solver.SetTimeLimit(max(1, round(seconds * 1000)))
        x = {
            (i, g): solver.BoolVar(f"x_{i}_{g}") for i in players for g in range(number)
        }
        for i in players:
            solver.Add(solver.Sum(x[i, g] for g in range(number)) == 1)
        for g in range(number):
            solver.Add(solver.Sum(x[i, g] for i in players) == group_size)
        solver.Add(x[min(players), 0] == 1)
        sums = []
        for g in range(number):
            terms = []
            for i, j in combinations(players, 2):
                y = solver.BoolVar(f"y_{i}_{j}_{g}")
                solver.Add(y <= x[i, g])
                solver.Add(y <= x[j, g])
                solver.Add(y >= x[i, g] + x[j, g] - 1)
                terms.append(float(scores.loc[i, j]) * y)
            sums.append(solver.Sum(terms))
        if minimum is None:
            z = solver.NumVar(
                -solver.infinity(), solver.infinity(), "minimum_group_score"
            )
            for expr in sums:
                solver.Add(expr >= z)
            solver.Maximize(z)
        else:
            for expr in sums:
                solver.Add(expr >= minimum - 1e-7)
            solver.Maximize(solver.Sum(sums))
        if hint:
            membership = {i: g for g, members in enumerate(hint) for i in members}
            solver.SetHint(list(x.values()), [float(membership[i] == g) for i, g in x])
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None, status
        return canonical(
            [
                [i for i in players if x[i, g].solution_value() > 0.5]
                for g in range(number)
            ]
        ), status

    first, status1 = build(seconds=time_limit * 0.75)
    if first is None:
        raise RuntimeError(
            f"Grouping MILP found no feasible partition (status {status1}); use local_search or a longer time limit"
        )
    second, status2 = build(
        minimum=objective(first, scores)[0],
        hint=first,
        seconds=max(0.001, time_limit - (time.monotonic() - started)),
    )
    groups = first if second is None else second
    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    }
    return groups, {
        "method": "milp",
        "stage1_status": status_names.get(status1, str(status1)),
        "stage2_status": status_names.get(status2, str(status2)),
        "certified_lexicographic_optimum": status1
        == status2
        == pywraplp.Solver.OPTIMAL,
        "stage2_used": second is not None,
        "solve_seconds": time.monotonic() - started,
    }


def local_grouping(
    players, scores, group_size, starts=100, passes=100, seed=42, patience=1000
):
    rng = random.Random(seed)
    matrix = scores.loc[players, players].to_numpy(float)
    index = {p: k for k, p in enumerate(players)}
    best, best_obj = None, (-np.inf, -np.inf)
    logs = []
    stalled = 0
    for start in range(starts):
        shuffled = list(players)
        rng.shuffle(shuffled)
        groups = canonical(
            [shuffled[i : i + group_size] for i in range(0, len(players), group_size)]
        )
        current = objective(groups, scores)
        values = [group_score(g, scores) for g in groups]
        swaps = 0
        for _ in range(passes):
            best_move, move_obj = None, current
            for g, h in combinations(range(len(groups)), 2):
                for a in groups[g]:
                    for b in groups[h]:
                        new_values = values.copy()
                        ia, ib = index[a], index[b]
                        new_values[g] += sum(
                            matrix[ib, index[p]] - matrix[ia, index[p]]
                            for p in groups[g]
                            if p != a
                        )
                        new_values[h] += sum(
                            matrix[ia, index[p]] - matrix[ib, index[p]]
                            for p in groups[h]
                            if p != b
                        )
                        candidate = (float(min(new_values)), float(sum(new_values)))
                        if is_better(candidate, move_obj):
                            best_move = (g, h, a, b, new_values)
                            move_obj = candidate
            if best_move is None:
                break
            g, h, a, b, values = best_move
            groups[g] = sorted([p for p in groups[g] if p != a] + [b])
            groups[h] = sorted([p for p in groups[h] if p != b] + [a])
            current = move_obj
            swaps += 1
        if is_better(current, best_obj):
            best, best_obj = canonical(groups), current
            stalled = 0
        else:
            stalled += 1
        logs.append(
            {
                "start": start + 1,
                "minimum_group_score": current[0],
                "total_score": current[1],
                "accepted_swaps": swaps,
            }
        )
        if stalled >= patience:
            break
    return (
        best,
        {
            "method": "local_search",
            "certified_lexicographic_optimum": False,
            "starts": len(logs),
        },
        pd.DataFrame(logs),
    )


def balanced_random_partition(players, prosumers, size, seed):
    """Equal-size random groups with prosumer counts differing by at most one."""
    rng = random.Random(seed)
    producers = [p for p in players if p in prosumers]
    consumers = [p for p in players if p not in prosumers]
    rng.shuffle(producers)
    rng.shuffle(consumers)
    groups = [[] for _ in range(len(players) // size)]
    for k, p in enumerate(producers):
        groups[k % len(groups)].append(p)
    for group in groups:
        while len(group) < size:
            group.append(consumers.pop())
    return canonical(groups)


def formation_table(groups, scores):
    return pd.DataFrame(
        [
            {
                "coalition_id": k + 1,
                "members": ",".join(map(str, g)),
                "coalition_size": len(g),
                "internal_pair_score_sum": group_score(g, scores),
            }
            for k, g in enumerate(groups)
        ]
    )
