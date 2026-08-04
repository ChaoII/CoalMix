# 跨车滚动采样 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `get_automatic_sampling_regions_rolling(used, need, config)` 实现同批煤跨车滚动采样，18 个采样点循环分配，返回 `(编号列表, 对应格子列表)`。

**Architecture:** 模块级缓存 `_NUMBERING`（编号 1-18 → 格子，服务启动时首次 `plan_regions()` 建立，用 `threading.Lock` 保护首次初始化）；分配逻辑从未用编号取 `need` 个，未用不足则清空换轮从 1 重取；返回编号按当次 `plan_regions()` 格子序排序。真实坐标复用现有 `get_automatic_sampling_points_from_regions`。

**Tech Stack:** Python 3.x, numpy, pytest, threading。

## Global Constraints

- 新增函数 `get_automatic_sampling_regions_rolling(used: list[int] | None = None, need: int = 0, config: SamplingConfig | None = None) -> tuple[list[int], list[list[int]]]`，定义在 `src/auto_sampling.py`。
- 编号 1-18 固定映射格子：首次调用时用 `plan_regions()` 建立模块级缓存 `_NUMBERING`（服务生命周期不变），用 `_NUMBERING_LOCK = threading.Lock()` 保护初始化。
- `used` 为当前轮已用编号（无重复、最多 18 个）；`need` 为本车要采点数。
- 分配规则：未用编号数 ≥ need → 按编号序取前 need 个；未用数 < need → 先取全部未用，再清空换轮从编号 1 补足剩余。
- 返回编号按当次 `plan_regions()` 的格子序排序；返回 `(排序后编号列表, 对应格子列表)`。
- `need <= 0` → 返回空 `([], [])`；`used=None` 按空处理。
- 真实坐标复用现有 `get_automatic_sampling_points_from_regions(...)`，不修改。
- 不新增 FastAPI 接口。
- 测试文件 `tests/test_auto_sampling.py`，运行命令 `python -m pytest tests/test_auto_sampling.py -v`。

---

### Task 1: `get_automatic_sampling_regions_rolling` + 编号映射缓存

**Files:**
- Modify: `src/auto_sampling.py`（新增 `import threading`、模块级 `_NUMBERING`/`_NUMBERING_LOCK`、新增函数）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: `CoalSamplingOptimizer`、`plan_regions()`、`SamplingConfig`（现有）。
- Produces: `get_automatic_sampling_regions_rolling(used, need, config) -> tuple[list[int], list[list[int]]]`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 末尾追加：

```python
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
```

注意：`test_rolling_cells_match_numbering` 与 `test_rolling_ordering_follows_current_plan` 依赖 `_NUMBERING` 内部符号——用 `from src.auto_sampling import _NUMBERING` 引用。`_NUMBERING` 值为 `(row, col)` 元组，`_NUMBERING[n] == tuple(cell)` 比较。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_automatic_sampling_regions_rolling'`。

- [ ] **Step 3: 实现**

修改 `src/auto_sampling.py`：
1) 顶部新增 `import threading`（在 `import random` 之后）：

```python
import random
import threading
from typing import Tuple
from dataclasses import dataclass
from log.log import logger
import numpy as np

from src.sampling_visualization import render_sampling_preview
```

2) 在文件末尾（`__main__` 块之前）新增缓存与函数：

```python
# 编号 -> 格子映射缓存（确定性：列优先从右往左，生命周期不变）
_NUMBERING: dict[int, tuple[int, int]] | None = None
_NUMBERING_LOCK = threading.Lock()


def _ensure_numbering(config: SamplingConfig | None) -> dict[int, tuple[int, int]]:
    """建立并返回编号 1..num_points -> 格子 的固定映射。

    编号规则（列优先、从右往左）：按列从右到左遍历，每列内行从小到大。
    因此编号 1..num_regions*rows 对应最右大区，逐列左移。
    确定性映射，与 plan_regions() 的随机顺序无关。
    """
    global _NUMBERING
    if _NUMBERING is None:
        with _NUMBERING_LOCK:
            if _NUMBERING is None:
                opt = CoalSamplingOptimizer(config=config)
                rows, cols = opt.rows, opt.cols
                numbering: dict[int, tuple[int, int]] = {}
                n = 1
                for c in range(cols - 1, -1, -1):
                    for r in range(rows):
                        numbering[n] = (r, c)
                        n += 1
                _NUMBERING = numbering
    return _NUMBERING


def get_automatic_sampling_regions_rolling(
        used: list[int] | None = None,
        need: int = 0,
        config: SamplingConfig | None = None) -> tuple[list[int], list[list[int]]]:
    """跨车滚动采样：返回 (本车应采的编号列表, 对应格子列表)。

    used: 当前轮已用编号（无重复，1-18 范围）。need: 本车要采点数。
    未用编号数不足 need 时，先取全部未用，再清空 used 从编号 1 换轮重取。
    返回编号按当次 plan_regions() 格子序排序。编号->格子映射固定（_NUMBERING）。
    """
    if need <= 0:
        return [], []

    numbering = _ensure_numbering(config)
    total = len(numbering)

    used_set = set(used or [])
    unused = [i for i in range(1, total + 1) if i not in used_set]

    if len(unused) >= need:
        allocated = unused[:need]
    else:
        # 未用不足：取全部未用，再清空换轮从 1 补足
        wrap_count = need - len(unused)
        allocated = unused + list(range(1, wrap_count + 1))

    # 按当次 plan_regions() 格子序排序返回
    opt = CoalSamplingOptimizer(config=config)
    current = opt.plan_regions()
    cur_pos = {cell: idx for idx, cell in enumerate(current)}
    ordered = sorted(allocated, key=lambda n: cur_pos[numbering[n]])

    cells = [list(numbering[n]) for n in ordered]
    return ordered, cells
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（38 passed = 29 既有 + 9 新）。

- [ ] **Step 5: 手动验证用户 4 轮例子**

Run: `python -c "import sys; sys.path.insert(0, '.'); from src.auto_sampling import get_automatic_sampling_regions_rolling as r; print(r([],5)[0]); print(r([1,2,3,4,5],6)[0]); print(r(list(range(1,12)),5)[0]); print(r(list(range(1,17)),6)[0])"`
Expected 输出含：第 1 行 `[1, 2, 3, 4, 5]`，第 2 行 `[6, 7, 8, 9, 10, 11]`，第 3 行 `[12, 13, 14, 15, 16]`，第 4 行集合 `{1,2,3,4,17,18}`（编号序可能不同因排序）。

- [ ] **Step 6: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "feat: add rolling cross-truck sampling region allocation"
```

---

## 完成标准

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过（39 passed）。
- 用户 4 轮例子逐轮正确（round1 `{1..5}`、round2 `{6..11}`、round3 `{12..16}`、round4 `{1,2,3,4,17,18}`）。
- 编号→格子映射为**列优先从右往左**（编号 1-6 最右大区、7-12 中间、13-18 最左），确定性，跨调用固定。
- 返回编号按当次 plan 格子序排序（棋盘格不相邻处理）。
- `python -m py_compile src/auto_sampling.py` 通过。
