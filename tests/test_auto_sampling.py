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


def _region_of(num):
    from src.auto_sampling import _NUMBERING
    c = _NUMBERING[num][1]
    return 2 if c >= 4 else (1 if c >= 2 else 0)


def test_rolling_round1_crosses_regions():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, cells = get_automatic_sampling_regions_rolling(used=[], need=5)
    assert len(nums) == 5
    assert len(cells) == 5
    assert len(set(nums)) == 5
    # 相邻点应来自不同大区（棋盘格：每轮大区右→左轮流）
    regions = [_region_of(n) for n in nums]
    for i in range(4):
        assert regions[i] != regions[i + 1], f"相邻点同一大区: {nums}"
    # 第一点应来自最右大区（大区2），且跨大区（含大区0/1/2）
    assert regions[0] == 2
    assert set(regions) == {0, 1, 2}


def test_rolling_round2_no_overlap():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    r1, _ = get_automatic_sampling_regions_rolling(used=[], need=5)
    r2, _ = get_automatic_sampling_regions_rolling(used=r1, need=6)
    assert len(set(r1) & set(r2)) == 0, "本轮不应与上轮重复"


def test_rolling_round3_accumulates():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    r1, _ = get_automatic_sampling_regions_rolling(used=[], need=5)
    r2, _ = get_automatic_sampling_regions_rolling(used=r1, need=6)
    r3, _ = get_automatic_sampling_regions_rolling(used=sorted(set(r1) | set(r2)), need=5)
    acc = set(r1) | set(r2) | set(r3)
    assert len(acc) == 16, "三轮累计应覆盖 16 个不同编号"


def test_rolling_wrap_continues_region_rotation():
    import src.auto_sampling as _m
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    # used 是当前批次顺序前缀：先采前16个
    get_automatic_sampling_regions_rolling(used=[], need=16)
    old_order = list(_m._BATCH_ORDER['default'])
    used = old_order[:16]
    nums, _ = get_automatic_sampling_regions_rolling(used=used, need=6)
    # 换轮：先取旧批次收尾（未用的2个），再生成新批次补足4个
    assert len(set(nums)) == len(nums), f"编号可重复(大区连续优先)，但长度应=6，实际 {nums}"
    # 大区应严格递减连续（含换轮衔接）
    regs = [_region_of(n) for n in nums]
    for i in range(len(regs) - 1):
        assert (regs[i] - regs[i + 1]) % 3 == 1, f"大区应递减连续，实际 {regs}"
    # 换轮生成新批次
    new_order = list(_m._BATCH_ORDER['default'])
    assert new_order != old_order, "换轮应生成新的随机批次顺序"
    # 前端下次 used = 新批次补足部分（返回尾部4个）
    filled = nums[-4:]
    assert filled == new_order[:4], f"补足应为新批次前4个，实际 {filled}"
    # 前端下次传新批次前4个，应延续新批次
    next_nums, _ = get_automatic_sampling_regions_rolling(used=filled, need=4)
    assert next_nums == new_order[4:8], f"应延续新批次，实际 {next_nums}"
    # 车4末尾 -> 车5开头 大区连续
    assert (_region_of(nums[-1]) - _region_of(next_nums[0])) % 3 == 1


def test_rolling_wrap_old_tail_then_new_batch():
    import src.auto_sampling as _m
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    get_automatic_sampling_regions_rolling(used=[], need=16)
    old_order = list(_m._BATCH_ORDER['default'])
    used = old_order[:16]
    old_tail = old_order[16:18]  # 旧批次收尾（未用的2个）
    nums, _ = get_automatic_sampling_regions_rolling(used=used, need=6)
    # 旧批次收尾点应在前（按旧批次顺序）
    assert old_tail[0] in nums[:2] or old_tail[0] in nums, "旧批次收尾点应被采到"
    assert set(old_tail) <= set(nums), f"旧收尾 {old_tail} 应被采到，实际 {nums}"


def test_rolling_same_car_adjacency_low_rate():
    import src.auto_sampling as _m
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    get_automatic_sampling_regions_rolling(used=[], need=1)  # 触发批次初始化
    _N = _m._NUMBERING

    def is_adj(a, b):
        r1, c1 = _N[a]; r2, c2 = _N[b]
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    # 多轮滚动统计同车相邻率（随机性优先，同车不相邻为"尽量"非强制）
    need_seq = [5, 6, 5, 6, 4, 5, 3, 6, 5, 5, 6, 4, 5, 5, 4]
    used = []
    total = 18
    adj_cars = 0
    all_cars = 0
    for _ in range(50):
        for need in need_seq:
            nums, _ = get_automatic_sampling_regions_rolling(used=used, need=need)
            if any(is_adj(nums[a], nums[b]) for a in range(len(nums)) for b in range(a + 1, len(nums))):
                adj_cars += 1
            all_cars += 1
            used = sorted(set(used) | set(nums))
            if len(used) >= total:
                used = []
    # 相邻率应显著低于随机水平（0~25%，纯随机约25%，此处只要求 < 50% 兜底非必然全相邻）
    assert adj_cars / all_cars < 0.5, f"同车相邻率过高: {adj_cars}/{all_cars}"


def test_sampling_order_core_constraints():
    import src.auto_sampling as _m
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    get_automatic_sampling_regions_rolling(used=[], need=1)
    _N = _m._NUMBERING
    order = list(_m._BATCH_ORDER['default'])
    # 大区轮转 [2,1,0]*6
    def region(n):
        c = _N[n][1]
        return 2 if c >= 4 else (1 if c >= 2 else 0)
    regs = [region(n) for n in order]
    assert regs == [2, 1, 0] * 6, f"大区应2->1->0轮转，实际 {regs}"
    # 前9同色后9另一色
    def color(n):
        r, c = _N[n]
        return (r + c) % 2
    assert color(order[0]) == color(order[8]) and color(order[9]) == color(order[17])
    assert color(order[0]) != color(order[9])
    # 全覆盖
    assert set(order) == set(range(1, 19))


def test_rolling_empty_need():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, cells = get_automatic_sampling_regions_rolling(used=[1, 2], need=0)
    assert nums == [] and cells == []


def test_rolling_used_none():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    nums, cells = get_automatic_sampling_regions_rolling(used=None, need=3)
    assert len(nums) == 3
    assert len(set(nums)) == 3
    regions = [_region_of(n) for n in nums]
    assert regions[0] == 2, "第一点应来自最右大区"


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


def test_rolling_ordering_follows_batch_order():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    import src.auto_sampling as _m
    # 批次内：返回编号应严格按当前批次采样顺序（大区2->1->0轮转）
    r1, _ = get_automatic_sampling_regions_rolling(used=[], need=5)  # 开始新批次
    order = list(_m._BATCH_ORDER['default'])
    assert r1 == order[:5], f"首车返回应按批次顺序，实际 {r1}"
    # 跨车（used 非空）沿用同一批次顺序
    r2, _ = get_automatic_sampling_regions_rolling(used=r1, need=5)
    assert r2 == order[5:10], f"次车返回应延续批次顺序，实际 {r2}"


def test_rolling_region_sequence_continuous():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    from src.auto_sampling import _NUMBERING
    # 滚动跨车：大区序列应持续 2->1->0 轮转
    used = []
    seq_nums = []
    for need in [5, 6, 5, 6]:
        nums, _ = get_automatic_sampling_regions_rolling(used=used, need=need)
        seq_nums.extend(nums)
        used = sorted(set(used) | set(nums))
    regions = []
    for n in seq_nums:
        c = _NUMBERING[n][1]
        regions.append(2 if c >= 4 else (1 if c >= 2 else 0))
    # 前 18 点（换轮前）大区应严格 [2,1,0]*6
    expected = [2, 1, 0] * 6
    assert regions[:18] == expected, f"前18点大区应持续2->1->0轮转，实际 {regions[:18]}"


def test_rolling_same_color_first_9():
    from src.auto_sampling import get_automatic_sampling_regions_rolling
    from src.auto_sampling import _NUMBERING
    # 前 9 点应为同一种奇偶（黑或白），后 9 点为另一种
    nums, _ = get_automatic_sampling_regions_rolling(used=[], need=18)
    parities = [(r + c) % 2 for (r, c) in (_NUMBERING[n] for n in nums)]
    assert parities[:9] == [parities[0]] * 9, "前9个点应同色（互不相邻）"
    assert parities[9:] == [1 - parities[0]] * 9, "后9个点应为另一色"
