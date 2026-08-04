import base64

import pytest

from src.auto_sampling import SamplingConfig, SamplingError


def test_default_config_matches_original_hardcoded_values():
    cfg = SamplingConfig()
    assert cfg.grid_rows == 3
    assert cfg.grid_cols == 6
    assert cfg.region_row_span == 3
    assert cfg.region_col_span == 2
    assert cfg.shuffle_regions is True
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


def test_invalid_config_raises_when_single_region():
    with pytest.raises(ValueError):
        SamplingConfig(grid_rows=3, grid_cols=6, region_row_span=3, region_col_span=6)


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


def test_render_sampling_animation_returns_base64_gif():
    from src.sampling_visualization import render_sampling_animation
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    gif = render_sampling_animation(opt, real_points, regions=regions)
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


def test_plan_regions_same_phase_points_never_adjacent():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    points = opt.plan_regions()
    for i, (r, c) in enumerate(points):
        prev_same_phase = [p for p in points[:i]
                           if (p[0] + p[1]) % 2 == (r + c) % 2]
        for p in prev_same_phase:
            assert not opt.is_adjacent((r, c), p)


def test_plan_regions_varied_across_calls_with_seed():
    p1 = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=1)).plan_regions()
    p2 = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=2)).plan_regions()
    assert p1 != p2, "不同 seed 的完整放置序列应不同"


def test_plan_regions_deterministic_when_shuffle_disabled():
    a = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=False)).plan_regions()
    b = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=False)).plan_regions()
    assert a == b


def test_plan_regions_adjacent_points_in_different_regions():
    opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=42))
    points = opt.plan_regions()
    for i in range(len(points) - 1):
        r1 = opt.region_mask[points[i][0], points[i][1]]
        r2 = opt.region_mask[points[i + 1][0], points[i + 1][1]]
        assert r1 != r2, f"相邻点 {points[i]} 与 {points[i + 1]} 来自同一大区 {r1}"


def test_phase_label_self_calibrates_with_shift():
    # 前 9 点应全为黑格相位；用首点奇偶性作为相位基准，所有前9点应一致
    for seed in range(8):
        opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=seed))
        points = opt.plan_regions()
        black_parity = (points[0][0] + points[0][1]) % 2
        first9 = points[:9]
        assert all((r + c) % 2 == black_parity for (r, c) in first9), \
            f"seed {seed}: 前9点相位不一致"


def test_plan_regions_region_order_right_to_left():
    opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=42))
    points = opt.plan_regions()
    regions = [int(opt.region_mask[r, c]) for (r, c) in points]
    expected = [2, 1, 0] * 6
    assert regions == expected, f"大区序列应为右→左循环，实际 {regions}"


def test_rolling_round1():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, cells = get_automatic_sampling_regions_rolling(used=[], need=5)
    assert len(nums) == 5
    assert len(cells) == 5
    assert set(nums) == {1, 2, 3, 4, 5}


def test_rolling_round2():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, _ = get_automatic_sampling_regions_rolling(used=[1, 2, 3, 4, 5], need=6)
    assert set(nums) == {6, 7, 8, 9, 10, 11}


def test_rolling_round3():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, _ = get_automatic_sampling_regions_rolling(used=list(range(1, 12)), need=5)
    assert set(nums) == {12, 13, 14, 15, 16}


def test_rolling_round4_wrap():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, _ = get_automatic_sampling_regions_rolling(used=list(range(1, 17)), need=6)
    assert set(nums) == {1, 2, 3, 4, 17, 18}


def test_rolling_empty_need():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, cells = get_automatic_sampling_regions_rolling(used=[1, 2], need=0)
    assert nums == [] and cells == []


def test_rolling_used_none():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, _ = get_automatic_sampling_regions_rolling(used=None, need=3)
    assert set(nums) == {1, 2, 3}


def test_rolling_cells_match_numbering():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    from src.auto_sampling import _NUMBERING
    nums, cells = get_automatic_sampling_regions_rolling(used=[], need=5)
    for n, cell in zip(nums, cells):
        assert _NUMBERING[n] == tuple(cell), f"编号{n}格子不匹配"


def test_rolling_numbering_fixed_across_calls():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    get_automatic_sampling_regions_rolling(used=[], need=1)
    from src.auto_sampling import _NUMBERING
    snapshot = dict(_NUMBERING)
    get_automatic_sampling_regions_rolling(used=[1], need=2)
    from src.auto_sampling import _NUMBERING
    assert _NUMBERING == snapshot, "编号映射应跨调用固定"


def test_rolling_numbering_column_first_right_to_left():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    from src.auto_sampling import _NUMBERING
    get_automatic_sampling_regions_rolling(used=[], need=1)
    expected = {}
    n = 1
    # 列优先从右往左：列 5..0，每列内行 0..2
    for c in range(5, -1, -1):
        for r in range(3):
            expected[n] = (r, c)
            n += 1
    assert _NUMBERING == expected, f"编号应按列优先从右往左映射，实际 {_NUMBERING}"
    # 大区分组：1-6 最右大区(列4,5)，7-12 中间(列2,3)，13-18 最左(列0,1)
    assert all(_NUMBERING[i][1] in (4, 5) for i in range(1, 7))
    assert all(_NUMBERING[i][1] in (2, 3) for i in range(7, 13))
    assert all(_NUMBERING[i][1] in (0, 1) for i in range(13, 19))


def test_rolling_ordering_follows_current_plan():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    from src.auto_sampling import CoalSamplingOptimizer, SamplingConfig, _NUMBERING
    cfg = SamplingConfig(shuffle_regions=True, seed=7)
    # 用固定 config 保证 _NUMBERING 与该 config 一致
    get_automatic_sampling_regions_rolling(used=[], need=5, config=cfg)
    opt = CoalSamplingOptimizer(cfg)
    current = opt.plan_regions()
    cur_pos = {tuple(cell): idx for idx, cell in enumerate(current)}
    nums, cells = get_automatic_sampling_regions_rolling(used=[], need=5, config=cfg)
    positions = [cur_pos[tuple(c)] for c in cells]
    assert positions == sorted(positions), "返回编号应按当次 plan 格子序排序"
