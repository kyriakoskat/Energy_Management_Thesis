import pandas as pd
import pytest
from energy_community.formation import (
    exact_grouping,
    local_grouping,
    objective,
    balanced_random_partition,
)


def test_lexicographic_grouping_against_exhaustive_partitions():
    players = [1, 2, 3, 4]
    scores = pd.DataFrame(
        [[0, 10, 6, 2], [10, 0, 2, 6], [6, 2, 0, 1], [2, 6, 1, 0]],
        index=players,
        columns=players,
    )
    candidates = [[[1, 2], [3, 4]], [[1, 3], [2, 4]], [[1, 4], [2, 3]]]
    expected = max(objective(p, scores) for p in candidates)
    groups, metadata = exact_grouping(players, scores, 2, 10)
    assert objective(groups, scores) == pytest.approx(expected)
    assert metadata["certified_lexicographic_optimum"]
    assert groups == [[1, 3], [2, 4]]
    searched, _, _ = local_grouping(players, scores, 2, 10, 10, 42)
    assert objective(searched, scores) == pytest.approx(expected)


def test_odd_sized_balanced_random_partitions():
    players = list(range(1, 11))
    prosumers = set(range(1, 6))
    groups = balanced_random_partition(players, prosumers, 5, 42)
    assert sorted(p for g in groups for p in g) == players
    assert all(len(g) == 5 for g in groups)
    assert sorted(len(set(g) & prosumers) for g in groups) == [2, 3]
