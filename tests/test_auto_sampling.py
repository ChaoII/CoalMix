import pytest

from src.auto_sampling import SamplingConfig, SamplingError


def test_default_config_matches_original_hardcoded_values():
    cfg = SamplingConfig()
    assert cfg.grid_rows == 3
    assert cfg.grid_cols == 6
    assert cfg.region_row_span == 3
    assert cfg.region_col_span == 2
    assert cfg.shuffle_regions is False
    assert cfg.seed is None
    assert cfg.max_coordinate_attempts == 100


def test_num_regions_is_3_for_default_config():
    assert SamplingConfig().num_regions == 3


def test_invalid_config_raises_on_undividable_rows():
    with pytest.raises(ValueError):
        SamplingConfig(grid_rows=3, grid_cols=6, region_row_span=2, region_col_span=2)


def test_invalid_config_raises_on_undividable_cols():
    with pytest.raises(ValueError):
        SamplingConfig(grid_rows=4, grid_cols=6, region_row_span=2, region_col_span=4)


def test_sampling_error_is_exception():
    assert issubclass(SamplingError, Exception)


from collections import Counter

from src.auto_sampling import CoalSamplingOptimizer


def test_optimizer_creates_three_regions_of_six_cells():
    opt = CoalSamplingOptimizer()
    assert opt.num_regions == 3
    assert [len(v) for v in opt.region_to_cells.values()] == [6, 6, 6]


def test_plan_regions_returns_18_unique_points():
    opt = CoalSamplingOptimizer()
    points = opt.plan_regions()
    assert len(points) == 18
    assert len(set(points)) == 18


def test_plan_regions_covers_all_cells():
    opt = CoalSamplingOptimizer()
    points = opt.plan_regions()
    expected = {(r, c) for r in range(3) for c in range(6)}
    assert set(points) == expected


def test_plan_regions_balances_regions():
    opt = CoalSamplingOptimizer()
    points = opt.plan_regions()
    counts = Counter(opt.region_mask[r, c] for (r, c) in points)
    assert counts == {0: 6, 1: 6, 2: 6}


def test_plan_regions_first_9_points_mutually_non_adjacent():
    opt = CoalSamplingOptimizer()
    points = opt.plan_regions()
    first9 = points[:9]
    for i in range(9):
        for j in range(i + 1, 9):
            assert not opt.is_adjacent(first9[i], first9[j])


def test_plan_regions_reproducible_with_seed():
    cfg = SamplingConfig(shuffle_regions=True, seed=42)
    assert CoalSamplingOptimizer(cfg).plan_regions() == CoalSamplingOptimizer(cfg).plan_regions()


def test_get_automatic_sampling_regions_returns_18():
    from src.auto_sampling import get_automatic_sampling_regions
    assert len(get_automatic_sampling_regions()) == 18
