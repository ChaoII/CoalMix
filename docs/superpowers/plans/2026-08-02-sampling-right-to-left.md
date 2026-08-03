# 采样大区从右往左遍历 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `plan_regions()` 的大区遍历顺序改为每轮固定从右往左（region `num_regions-1` → `0`），去掉大区轮序随机 shuffle 和跨轮边界 while 约束。

**Architecture:** `plan_regions` 中 `region_order` 从"`list(range(num_regions))` + 每轮 `while` shuffle 且保证 `region_order[0] != prev_last`"改为固定递减序列 `list(range(num_regions-1, -1, -1))`（默认 `[2,1,0]`），并移除 shuffle 与 while 块。随机性仅来自 `phase_shift`（黑白划分 2 种）与每区黑/白格列表内部 shuffle。所有既有约束（相邻点不同区、前 9 点不相邻、全覆盖、每区 6 点、确定性）保持不变。

**Tech Stack:** Python 3.x, numpy, pytest。

## Global Constraints

- `plan_regions() -> list[tuple[int, int]]`：`region_order = list(range(self.num_regions - 1, -1, -1))`（默认 `[2,1,0]`），每轮都用此固定顺序，去掉大区轮序 shuffle 与跨轮边界 while。
- 保留 `phase_shift = 0 if not shuffle else self._rng.randint(0, 1)` 与每区黑/白格列表内部 `self._rng.shuffle`（当 `shuffle=True`）。
- 默认配置下大区序列严格为 `[2,1,0,2,1,0,...]`（18 点，6 轮）。
- 约束保持：18 格全覆盖无重复、每区 6 点、前 `num_regions` 轮（前 9 点）黑格互不相邻、任意相邻采样点来自不同大区。
- `shuffle_regions=False` 完全确定性（shift=0、区内不 shuffle）。
- 测试文件 `tests/test_auto_sampling.py`，运行命令 `python -m pytest tests/test_auto_sampling.py -v`。
- 不新增依赖；不改 main.py / FastAPI / 动画。

---

### Task 1: `plan_regions` 改为固定右→左遍历

**Files:**
- Modify: `src/auto_sampling.py`（`plan_regions` 方法）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: `SamplingConfig.shuffle_regions`、`seed`、`_rng`、`region_to_cells`、`region_mask`、`num_regions`、`num_points`（现有）。
- Produces: 固定右→左的 `plan_regions()`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 末尾追加：

```python
def test_plan_regions_region_order_right_to_left():
    opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=42))
    points = opt.plan_regions()
    regions = [int(opt.region_mask[r, c]) for (r, c) in points]
    expected = [2, 1, 0] * 6
    assert regions == expected, f"大区序列应为右→左循环，实际 {regions}"
```

（`CoalSamplingOptimizer` 与 `SamplingConfig` 已在测试文件顶部 import。）

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_auto_sampling.py::test_plan_regions_region_order_right_to_left -v`
Expected: FAIL — 当前实现大区顺序随机 shuffle，序列不是严格的 `[2,1,0]*6`。

- [ ] **Step 3: 实现**

替换 `src/auto_sampling.py` 中 `plan_regions` 方法（整块替换）：

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

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（29 passed = 28 + 1 新）。既有测试如 `test_plan_regions_adjacent_points_in_different_regions`、`test_plan_regions_first_9_points_mutually_non_adjacent`、`test_plan_regions_varied_across_calls_with_seed`、`test_plan_regions_deterministic_when_shuffle_disabled` 全部保持通过。

- [ ] **Step 5: 手动验证约束保持**

Run: `python -c "import sys; sys.path.insert(0, '.'); from src.auto_sampling import CoalSamplingOptimizer, SamplingConfig; o=CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=42)); p=o.plan_regions(); print([int(o.region_mask[r,c]) for r,c in p]); print(all(int(o.region_mask[p[i][0],p[i][1]])!=int(o.region_mask[p[i+1][0],p[i+1][1]]) for i in range(17)))"`
Expected: 打印 `[2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]` 和 `True`。

- [ ] **Step 6: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "feat: traverse sampling regions right-to-left"
```

---

## 完成标准

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过（29 passed）。
- 默认配置下 `plan_regions()` 大区序列严格为 `[2,1,0]*6`。
- 任意相邻采样点来自不同大区；前 9 点互不相邻；18 格全覆盖无重复、每区 6 点。
- `shuffle_regions=False` 两次调用结果一致（确定性）。
- `python -m py_compile src/auto_sampling.py` 通过。
