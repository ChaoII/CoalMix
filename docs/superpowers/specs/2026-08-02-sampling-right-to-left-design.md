# 采样大区从右往左遍历设计

日期：2026-08-02
分支：`refactor/auto-sampling`

## 背景

`plan_regions()` 当前每轮对大区顺序做随机 shuffle，并通过跨轮边界约束（`region_order[0] != prev_last`）保证相邻采样点来自不同大区。用户希望改为**每轮固定从右往左遍历大区**：每轮采样从最右大区开始，依次向左，规律化结构。

## 目标

1. 大区遍历顺序固定为从右往左（region `num_regions-1` → `0`），每轮都如此。
2. 去掉大区轮序 shuffle 和跨轮边界 while 约束（固定顺序天然保证相邻点不同区）。
3. 保留随机性：`phase_shift`（黑白划分 2 种）+ 每区黑/白格列表内部 shuffle。
4. 保持全部既有约束：前 9 点黑格互不相邻、18 格全覆盖无重复、每区 6 点、任意相邻点不同区。
5. `shuffle_regions=False` 完全确定性。

## 设计

### `plan_regions` 改动

`region_order` 从 `list(range(self.num_regions))` + 每轮 shuffle 改为固定递减序列 `list(range(self.num_regions - 1, -1, -1))`（即 `[2,1,0]`），去掉 shuffle 与跨轮 while 约束：

```python
    def plan_regions(self) -> list[tuple[int, int]]:
        """规划采样小区（从右往左）。

        全局黑格相位随机（(r+c+shift)%2，shift 每次随机 0/1）保证前 num_regions 轮
        黑格互不相邻；每轮大区固定从右往左（region num_regions-1 → 0），保证相邻
        采样点来自不同大区；黑格/白格集合内部随机排列。前 num_regions 轮分配黑格，
        后 num_regions 轮用白格填满，实现全覆盖、无重复、每区均匀。
        seed 固定时可复现。shuffle_regions=False 时完全确定性。
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
            if shuffle:
                self._rng.shuffle(black[region_id])
                self._rng.shuffle(white[region_id])

        constrained_rounds = len(black[0])
        rounds = self.num_points // self.num_regions
        region_order = list(range(self.num_regions - 1, -1, -1))  # 从右往左
        selected: list[tuple[int, int]] = []
        for round_idx in range(rounds):
            for region_id in region_order:
                pool = black if round_idx < constrained_rounds else white
                selected.append(pool[region_id].pop())
                logger.info(f"第{round_idx + 1}次采样，分配到区域{region_id}，位置{selected[-1]}")
        return selected
```

### 约束保持（实测验证）

- 大区序列严格 `[2,1,0,2,1,0,...]` → 任意相邻点不同区 ✓
- 前 9 点黑格（同相位棋盘性质）互不相邻 ✓
- 18 格全覆盖、无重复、每区 6 点 ✓
- `shuffle_regions=False`：shift=0、区内不 shuffle → 完全确定 ✓

### 测试更新（`tests/test_auto_sampling.py`）

- 既有 `test_plan_regions_adjacent_points_in_different_regions` 保持通过。
- 既有 `test_plan_regions_varied_across_calls_with_seed`、`test_plan_regions_deterministic_when_shuffle_disabled`、`test_plan_regions_first_9_points_mutually_non_adjacent`、`test_plan_regions_reproducible_with_seed` 保持通过。
- 新增 `test_plan_regions_region_order_right_to_left`：
  - 每轮首点（位置 0,3,6,...）都来自最右大区 `num_regions-1`。
  - 每轮内大区严格递减（region 序列为 2,1,0）。

```python
def test_plan_regions_region_order_right_to_left():
    opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=42))
    points = opt.plan_regions()
    regions = [int(opt.region_mask[r, c]) for (r, c) in points]
    expected = [2, 1, 0] * 6
    assert regions == expected, f"大区序列应为右→左循环，实际 {regions}"
```

## 不做的事

- 不改 main.py / FastAPI 接口。
- 不改动画逻辑。
- 不改 `SamplingConfig` 字段（`shuffle_regions` 语义保持，但 shuffle 不再作用于大区轮序，仅作用于黑白划分与区内格子——docstring 需说明）。

## 验证方式

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过（29 passed = 28 + 1 新）。
- 手动：多 seed 下大区序列均为 `[2,1,0]*6`；`shuffle_regions=False` 两次调用一致。
- `python -m py_compile src/auto_sampling.py` 通过。
