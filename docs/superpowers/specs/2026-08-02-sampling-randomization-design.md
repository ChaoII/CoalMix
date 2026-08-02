# 采样小区随机化设计

日期：2026-08-02
分支：`refactor/auto-sampling`

## 背景

用户反馈：默认配置下 `plan_regions()` 返回的 18 个采样小区布局完全确定（黑格 `(r+c)%2` 固定、大区轮序不 shuffle），每次来车采样点布局都一样，动画失去意义。需求：**每次调用采样小区布局随机不同**。

实测确认：
- 当前 `shuffle_regions=True` 只打乱放置顺序（`same set: True, same order: False`），18 格集合每次完全相同。
- 2×3 大区的最大不相邻集恰好 2 种（棋盘两色）；3 个大区组合上限 2³ = 8 种黑格集。
- 动画函数 `render_sampling_animation` 内部重调 `opt.plan_regions()`，在随机化后必然与调用方 regions 错位（desync）。

## 目标

1. 默认 `SamplingConfig.shuffle_regions = True`，`seed = None`（每次随机）。
2. `plan_regions()` 每次调用生成**真实不同的选格集合**：黑格相位随机（`(r+c+shift)%2`，shift 每次随机 0/1），配合大区轮序 shuffle，8 种黑格集 × 随机放置顺序。
3. 保持既有约束：18 格全覆盖无重复、每区 6 点、前 `num_regions` 轮（前 9 点）黑格互不相邻、后 9 点白格填满。
4. `seed` 固定时结果可复现（确定性选项保留）。
5. 修复动画 desync：`render_sampling_animation` 增加可选 `regions` 参数。
6. 生成多组不同 seed 的 GIF 展示布局差异。

## 设计

### 1. `SamplingConfig` 默认值变更

```python
shuffle_regions: bool = True   # 原 False
seed: int | None = None        # 不变（None=每次随机）
```

### 2. `plan_regions` 随机化

黑格判定从固定 `(r+c)%2` 改为 `(r+c+phase_shift)%2`，`phase_shift = self._rng.randint(0, 1)` 每次调用随机取 0/1：

```python
def plan_regions(self) -> list[tuple[int, int]]:
    """随机化规划采样小区。

    黑格相位随机（(r+c+shift)%2，shift 每次随机 0/1），配合大区轮序 shuffle，
    每次调用生成不同的黑格集（每区 2 种 × 3 区 = 8 种组合）与放置顺序。
    前 num_regions 轮分配黑格（互不相邻），后 num_regions 轮用白格填满，
    实现全覆盖、无重复、每区均匀。seed 固定时可复现。
    """
    shuffle = self.config.shuffle_regions
    phase_shift = 0 if not shuffle else self._rng.randint(0, 1)
    black = {r: [] for r in range(self.num_regions)}
    white = {r: [] for r in range(self.num_regions)}
    for region_id in range(self.num_regions):
        for (r, c) in self.region_to_cells[region_id]:
            if (r + c + phase_shift) % 2 == 0:
                black[region_id].append((r, c))
            else:
                white[region_id].append((r, c))

    constrained_rounds = len(black[0])
    rounds = self.num_points // self.num_regions
    region_order = list(range(self.num_regions))
    selected: list[tuple[int, int]] = []
    for round_idx in range(rounds):
        if shuffle:
            self._rng.shuffle(region_order)
        for region_id in region_order:
            pool = black if round_idx < constrained_rounds else white
            selected.append(pool[region_id].pop())
            logger.info(f"第{round_idx + 1}次采样，分配到区域{region_id}，位置{selected[-1]}")
    return selected
```

说明：当 `shuffle_regions=False` 时行为退化为原确定性棋盘式（shift 不影响布局集合唯一性，但仍会随机 shift——为保持"确定性"语义，需保证 `shuffle_regions=False` 时 phase_shift 固定为 0）。见下"确定性语义"节。

### 3. 确定性语义

- `shuffle_regions=True`：每次调用 `self._rng.randint(0,1)` + `self._rng.shuffle`，布局随机；`seed` 固定则同 seed 同结果。
- `shuffle_regions=False`：必须完全确定性（黑格集固定）。实现：`phase_shift = 0 if not self.config.shuffle_regions else self._rng.randint(0, 1)`，且不做轮序 shuffle。这样 `seed=None` + `shuffle_regions=False` 仍是固定棋盘式（向后兼容）。

### 4. 动画 desync 修复

`render_sampling_animation(opt, real_points, regions=None, fps=1, interval=1000)`：
- `regions` 参数传入时，动画用调用方的小区序列与 `real_points` 一一对应画点（消除 desync）。
- `regions=None` 时内部 `opt.plan_regions()`（向后兼容，但随机化下不建议依赖）。
- 更新 docstring：注明随机化下必须传 `regions`，否则内部重调 `plan_regions()` 结果可能与调用方不同。

### 5. 测试更新（`tests/test_auto_sampling.py`）

- 改 `test_default_config_matches_original_hardcoded_values`：`shuffle_regions is True`。
- 保留：`test_plan_regions_returns_18_unique_points`、`test_plan_regions_covers_all_cells`、`test_plan_regions_balances_regions`（随机化后仍成立）。
- 改 `test_plan_regions_first_9_points_mutually_non_adjacent`：随机化后相位不确定，改用 `phase_shift` 逻辑或直接按"前 9 点中同相位互不相邻"验证。保留"前 9 点两两不相邻"的验证——由于前 9 点全是黑格（同 shift），仍应互不相邻。
- 保留 `test_plan_regions_reproducible_with_seed`（seed 相同可复现）。
- 改 `test_plan_regions_same_phase_points_never_adjacent`：相位判定同步为 `(r+c+shift)%2`，但测试无法预知 shift——改为固定 `shuffle_regions=False` 配置断言，或断言"同相位点互不相邻"用更通用的棋盘性质。
- 动画测试 `test_render_sampling_animation_returns_base64_gif`：改传 `regions=regions`。
- 新增 `test_plan_regions_varied_across_calls`：`shuffle_regions=True` 且不同 seed 时，两次调用的选格集合可不同（集合不完全一致）。
- 新增 `test_plan_regions_deterministic_when_shuffle_disabled`：`shuffle_regions=False` 且 seed 任意时，两次调用结果完全一致。

### 6. 生成多组 GIF

用 seed 1/2/3/4 各生成一组 `sampling_anim_seed<N>.gif`，展示布局差异（供用户检查随机性）。

## 不做的事

- 不改 main.py / FastAPI 接口签名。
- 不引入新依赖。
- 不改动画的规则检查逻辑（同相位相邻性等）。

## 验证方式

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过。
- 手动：不同 seed 下 `plan_regions()` 集合不同、放置顺序不同；同 seed 下完全一致。
- `python -m py_compile src/auto_sampling.py src/sampling_visualization.py` 通过。
- 生成多组 GIF 可打开、布局明显不同。
