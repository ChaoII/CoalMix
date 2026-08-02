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


def test_sample_point_stays_within_allowed_region():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    for _ in range(50):
        x, y = opt.sample_point_in_region(0, 0)
        assert 100 < x < 10900
        assert 100 < y < 5400


def test_sample_point_avoids_lajin():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    for _ in range(100):
        x, y = opt.sample_point_in_region(0, 1)
        assert not (2000 <= x <= 2200 and 100 <= y <= 5400)


def test_sample_point_raises_sampling_error_when_impossible():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[], yx=[6000, 4000, 10900, 5400])
    with pytest.raises(SamplingError):
        opt.sample_point_in_region(0, 0)


def test_get_automatic_sampling_points_from_regions_shape():
    from src.auto_sampling import get_automatic_sampling_points_from_regions
    regions = CoalSamplingOptimizer().plan_regions()
    real_points, image = get_automatic_sampling_points_from_regions(
        11000, 5500, [[2000, 100, 2200, 5400]], [100, 100, 10900, 5400], regions)
    assert len(real_points) == 18
    assert all(len(p) == 2 for p in real_points)
    assert isinstance(image, str) and image.startswith("data:image/png;base64,")


from src.sampling_visualization import render_sampling_preview


def test_render_sampling_preview_returns_base64_png():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    image = render_sampling_preview(opt, regions, real_points)
    assert image.startswith("data:image/png;base64,")


def test_optimize_sampling_removed():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    assert not hasattr(opt, 'optimize_sampling')


def test_get_automatic_sampling_points_full_flow():
    from src.auto_sampling import get_automatic_sampling_points
    points, image = get_automatic_sampling_points(
        11000, 5500, [[2000, 100, 2200, 5400]], [100, 100, 10900, 5400])
    assert len(points) == 18
    assert image.startswith("data:image/png;base64,")


def test_render_sampling_preview_unchanged_after_draw_base_refactor():
    from src.sampling_visualization import render_sampling_preview
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    image = render_sampling_preview(opt, regions, real_points)
    assert image.startswith("data:image/png;base64,")


import base64


def test_render_sampling_animation_returns_base64_gif():
    from src.sampling_visualization import render_sampling_animation
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    gif = render_sampling_animation(opt, real_points)
    assert gif.startswith("data:image/gif;base64,")
    raw = base64.b64decode(gif.split(",", 1)[1])
    assert raw[:6] in (b"GIF89a", b"GIF87a")


def test_render_sampling_animation_wrong_point_count_raises():
    from src.sampling_visualization import render_sampling_animation
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    with pytest.raises(ValueError):
        render_sampling_animation(opt, [[0, 0]] * 5)
