# Auto-Sampling 模块重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 `src/auto_sampling.py`，用 `SamplingConfig` 配置类 + 确定性棋盘式选点替代硬编码和随机重试算法，拆分可视化，修复类型标注，并补充 pytest 单测。

**Architecture:** 优化器 `CoalSamplingOptimizer` 改为接收 `SamplingConfig`（网格行/列、大区跨度、随机种子等），选点用棋盘式着色（黑格两两不相邻、每区均匀、全覆盖）替代 `get_min_regions` 的随机重试；坐标生成收敛为 `sample_point_in_region`，失败抛 `SamplingError`；可视化拆到 `src/sampling_visualization.py`。三个入口函数签名保持兼容（追加可选 `config` 参数），main.py 无需改动。

**Tech Stack:** Python 3.x, numpy, matplotlib, pytest, loguru。

## Global Constraints

- 入口函数名与位置不变：`get_automatic_sampling_points`、`get_automatic_sampling_regions`、`get_automatic_sampling_points_from_regions`，均可选追加 `config: SamplingConfig | None = None`。
- main.py 中三个 FastAPI 接口调用方式不变，`get_automatic_sampling_points` / `get_automatic_sampling_points_from_regions` 返回 `(real_points, image_base64)`，`get_automatic_sampling_regions` 返回 `list[tuple[int, int]]`。
- 默认配置 `SamplingConfig()` 必须复现原硬编码值：`grid_rows=3, grid_cols=6, region_row_span=3, region_col_span=2`。
- `SamplingConfig.__post_init__` 校验：`grid_rows % region_row_span == 0`、`grid_cols % region_col_span == 0`、`grid_rows * grid_cols % num_regions == 0`，不满足抛 `ValueError`。
- 自定义异常 `SamplingError` 用于坐标生成失败。
- 测试用 `pytest`，测试文件在 `tests/`，根目录 `conftest.py` 保证 `from src.auto_sampling import ...` 可解析。
- 提交信息遵循仓库现有风格（如 `update ...` / `feat: ...` / `docs: ...`）。

---

### Task 1: `SamplingConfig` 配置类与测试脚手架

**Files:**
- Create: `conftest.py`（仓库根目录）
- Create: `tests/test_auto_sampling.py`
- Modify: `src/auto_sampling.py`（顶部新增 import 与配置类，不动现有逻辑）

**Interfaces:**
- Produces: `SamplingConfig`（frozen dataclass，字段 `grid_rows=3, grid_cols=6, region_row_span=3, region_col_span=2, shuffle_regions=False, seed=None, max_coordinate_attempts=100`，只读属性 `num_regions`，`__post_init__` 校验）；`SamplingError(Exception)`。

- [ ] **Step 1: 写失败的测试**

创建 `conftest.py`（仓库根目录，保证 src 可导入）：

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

创建 `tests/test_auto_sampling.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.auto_sampling'`（当前文件不存在这些类，或 import 失败）

- [ ] **Step 3: 实现配置类**

在 `src/auto_sampling.py` 顶部（`from typing import ...` 之后）新增 import 与配置类：

```python
from dataclasses import dataclass


class SamplingError(Exception):
    """采样过程中无法生成满足约束的点时抛出。"""


@dataclass(frozen=True)
class SamplingConfig:
    grid_rows: int = 3
    grid_cols: int = 6
    region_row_span: int = 3
    region_col_span: int = 2
    shuffle_regions: bool = False
    seed: int | None = None
    max_coordinate_attempts: int = 100

    def __post_init__(self):
        if self.grid_rows % self.region_row_span != 0:
            raise ValueError(
                f"grid_rows({self.grid_rows}) 必须能被 region_row_span({self.region_row_span}) 整除"
            )
        if self.grid_cols % self.region_col_span != 0:
            raise ValueError(
                f"grid_cols({self.grid_cols}) 必须能被 region_col_span({self.region_col_span}) 整除"
            )
        if self.grid_rows * self.grid_cols % self.num_regions != 0:
            raise ValueError("单元格总数必须能被大区数整除")

    @property
    def num_regions(self) -> int:
        return (self.grid_rows // self.region_row_span) * (
            self.grid_cols // self.region_col_span
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add conftest.py tests/test_auto_sampling.py src/auto_sampling.py
git commit -m "feat: add SamplingConfig and SamplingError with validation tests"
```

---

### Task 2: 优化器改为配置驱动 + 确定性 `plan_regions`

**Files:**
- Modify: `src/auto_sampling.py`（重构 `__init__`、新增 `plan_regions`、删除 `get_min_regions`/`get_valid_choice_from_region`/`is_adjacent_any`、更新三个入口函数与 `optimize_sampling`）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: `SamplingConfig`, `SamplingError`（Task 1）。
- Produces: `CoalSamplingOptimizer(config=None, length=0, width=0, ljs=(), yx=())`，属性 `rows/cols/row_span/col_span/num_points/num_regions/region_mask/region_to_cells/_rng`；`plan_regions() -> list[tuple[int,int]]`（确定性：黑格先、白格后，前 `num_regions` 轮互不相邻，全覆盖无重复）；静态方法 `generate_region_mask`、`is_adjacent`。
- 入口函数 `get_automatic_sampling_regions(config=None)` 直接用 `plan_regions()`，去掉 100 次重试循环。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: FAIL — `get_automatic_sampling_regions` 仍走旧逻辑返回 tuple 列表，或 `plan_regions` 不存在（AttributeError）。

- [ ] **Step 3: 实现配置驱动优化器**

修改 `CoalSamplingOptimizer.__init__` 与相关方法（整块替换下列方法，其余方法如 `get_point`/`get_region_coordinates`/`get_point1`/可视化方法本任务先保留）：

```python
    def __init__(self, config: SamplingConfig | None = None, length: int = 0,
                 width: int = 0, ljs=tuple(), yx=tuple()):
        """初始化采样优化器"""
        self.config = config or SamplingConfig()
        self.length = length
        self.width = width
        self.ljs = list(ljs)
        self.yx = list(yx)
        self.rows = self.config.grid_rows
        self.cols = self.config.grid_cols
        self.row_span = self.config.region_row_span
        self.col_span = self.config.region_col_span
        self.num_points = self.rows * self.cols
        self.num_regions = self.config.num_regions

        # 随机源：seed 固定时可复现
        self._rng = random.Random(self.config.seed)

        # 生成大区划分
        self.region_mask = self.generate_region_mask(self.rows, self.cols,
                                                     self.row_span, self.col_span)

        # 构建区域到格子的映射
        self.region_to_cells = {}
        for region_id in range(self.num_regions):
            self.region_to_cells[region_id] = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.region_mask[i, j] == region_id:
                        self.region_to_cells[region_id].append((i, j))
```

新增 `plan_regions` 与辅助（替换 `get_min_regions`/`get_valid_choice_from_region`/`is_adjacent_any`）：

```python
    def plan_regions(self) -> list[tuple[int, int]]:
        """确定性规划采样小区。

        棋盘式着色：黑格 ((r+c)%2==0) 两两不相邻，且每个大区恰好 3 个。
        前 num_regions 轮分配黑格（保证互不相邻），后 num_regions 轮用白格填满，
        实现全覆盖、无重复、每区均匀。
        """
        black = {r: [] for r in range(self.num_regions)}
        white = {r: [] for r in range(self.num_regions)}
        for region_id in range(self.num_regions):
            for (r, c) in self.region_to_cells[region_id]:
                if (r + c) % 2 == 0:
                    black[region_id].append((r, c))
                else:
                    white[region_id].append((r, c))

        constrained_rounds = len(black[0])
        rounds = self.num_points // self.num_regions
        region_order = list(range(self.num_regions))
        selected: list[tuple[int, int]] = []
        for round_idx in range(rounds):
            if self.config.shuffle_regions:
                self._rng.shuffle(region_order)
            for region_id in region_order:
                pool = black if round_idx < constrained_rounds else white
                selected.append(pool[region_id].pop())
                logger.info(f"第{round_idx + 1}次采样，分配到区域{region_id}，位置{selected[-1]}")
        return selected
```

替换 `optimize_sampling`（其余 `optimize_sampling_points_from_regions` 保留到 Task 4 再改名）：

```python
    def optimize_sampling(self):
        result = self.plan_regions()
        return self.optimize_sampling_points_from_regions(result)
```

更新三个入口函数（去掉各自的重试循环，构造器改用 config）：

```python
def get_automatic_sampling_points(car_length: int,
                                  car_width: int,
                                  car_lj: tuple = tuple(),
                                  car_kx: tuple = tuple(),
                                  config: SamplingConfig | None = None):
    logger.info(f"汽车长度：{car_length}")
    logger.info(f"汽车宽度：{car_width}")
    logger.info(f"拉筋区域：{car_lj}")
    logger.info(f"允许区域：{car_kx}")
    opt = CoalSamplingOptimizer(config=config, length=car_length, width=car_width,
                                ljs=car_lj, yx=car_kx)
    return opt.optimize_sampling()


def get_automatic_sampling_regions(config: SamplingConfig | None = None):
    opt = CoalSamplingOptimizer(config=config)
    return opt.plan_regions()


def get_automatic_sampling_points_from_regions(
        car_length: int,
        car_width: int,
        car_lj: tuple = tuple(),
        car_kx: tuple = tuple(),
        regions: list[list[int, int]] = tuple(),
        config: SamplingConfig | None = None):
    logger.info(f"汽车长度：{car_length}")
    logger.info(f"汽车宽度：{car_width}")
    logger.info(f"拉筋区域：{car_lj}")
    logger.info(f"允许区域：{car_kx}")
    opt = CoalSamplingOptimizer(config=config, length=car_length, width=car_width,
                                ljs=car_lj, yx=car_kx)
    return opt.optimize_sampling_points_from_regions(regions)
```

说明：Task 2 中 `get_automatic_sampling_points` 与 `get_automatic_sampling_points_from_regions` 内部仍复用旧的 `optimize_sampling_points_from_regions`（其中 `get_point1`/可视化在 Task 3/4 改造），但入口已改为 config 驱动且无重试循环。`get_automatic_sampling_regions` 已完全走 `plan_regions`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: PASS（13 passed）。`get_automatic_sampling_regions` 返回 18 个 `tuple`。

- [ ] **Step 5: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "feat: deterministic checkerboard region planning via SamplingConfig"
```

---

### Task 3: 确定性坐标生成 `sample_point_in_region`

**Files:**
- Modify: `src/auto_sampling.py`（新增 `sample_point_in_region`、`realize_regions`，删除 `get_point`/`get_region_coordinates`/`get_point1`/`get_region_coordinates1`，更新入口函数）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: `plan_regions()`（Task 2）。
- Produces: `sample_point_in_region(row, col) -> list[float]`（在格子内随机撒点，避开拉筋 `ljs`、落在允许区域 `yx`，最多 `max_coordinate_attempts` 次，失败抛 `SamplingError`）；`realize_regions(regions) -> tuple[list[list[float]], str]`（真实坐标 + base64 图）。`optimize_sampling_points_from_regions` 改名为 `realize_regions`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: FAIL — `sample_point_in_region` 不存在（AttributeError）。

- [ ] **Step 3: 实现坐标生成**

删除方法：`get_point`、`get_region_coordinates`、`get_point1`、`get_region_coordinates1`。

新增方法（替换 `get_point1`，逻辑与原 `get_point1` 一致但去掉 `delta=0` 死参数、改用配置上限与自定义异常）：

```python
    def sample_point_in_region(self, row: int, col: int) -> list[float]:
        """在指定格子(row, col)内随机生成一个满足约束的真实坐标点。"""
        region_length = self.length / self.cols
        region_width = self.width / self.rows
        x0 = col * region_length
        y0 = row * region_width
        x1 = x0 + region_length
        y1 = y0 + region_width
        for _ in range(self.config.max_coordinate_attempts):
            x = self._rng.uniform(x0, x1)
            y = self._rng.uniform(y0, y1)
            if self._in_lajin(x, y):
                continue
            if self._in_allowed_region(x, y):
                return [x, y]
        raise SamplingError(
            f"区域({row},{col})在{self.config.max_coordinate_attempts}次尝试内无法生成满足约束的采样点，"
            "请检查拉筋和允许区域设置"
        )

    def _in_lajin(self, x: float, y: float) -> bool:
        return any(lj[0] <= x <= lj[2] and lj[1] <= y <= lj[3] for lj in self.ljs)

    def _in_allowed_region(self, x: float, y: float) -> bool:
        return self.yx[0] < x < self.yx[2] and self.yx[1] < y < self.yx[3]
```

将 `optimize_sampling_points_from_regions` 重命名为 `realize_regions`（内部仍调用本文件的 `visualize_sampling_points`，Task 4 拆分）：

```python
    def realize_regions(self, regions: list[list[int, int]]) -> tuple[list[list[float]], str]:
        real_points = []
        for (r, c) in regions:
            real_points.append(self.sample_point_in_region(r, c))
        fig = self.visualize_sampling_points(regions, real_points, self.rows * self.cols)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        image_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return real_points, image_base64
```

更新 `optimize_sampling` 与两个入口函数中对该方法的调用（`optimize_sampling_points_from_regions` → `realize_regions`）：

```python
    def optimize_sampling(self):
        result = self.plan_regions()
        return self.realize_regions(result)
```

```python
def get_automatic_sampling_points_from_regions(
        car_length: int,
        car_width: int,
        car_lj: tuple = tuple(),
        car_kx: tuple = tuple(),
        regions: list[list[int, int]] = tuple(),
        config: SamplingConfig | None = None):
    logger.info(f"汽车长度：{car_length}")
    logger.info(f"汽车宽度：{car_width}")
    logger.info(f"拉筋区域：{car_lj}")
    logger.info(f"允许区域：{car_kx}")
    opt = CoalSamplingOptimizer(config=config, length=car_length, width=car_width,
                                ljs=car_lj, yx=car_kx)
    return opt.realize_regions(regions)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: PASS（17 passed）

- [ ] **Step 5: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "feat: bounded deterministic coordinate sampling with SamplingError"
```

---

### Task 4: 可视化拆分到 `src/sampling_visualization.py`

**Files:**
- Create: `src/sampling_visualization.py`
- Modify: `src/auto_sampling.py`（删除可视化方法与相关 import，`realize_regions` 改用新模块）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Produces: `render_sampling_preview(opt, sampling_points, real_points) -> str`（接收优化器实例与小区/真实坐标，返回 base64 PNG）。`opt` 需要属性 `length, width, rows, cols, region_mask, num_regions, ljs, yx`。
- `auto_sampling.py` 的 `realize_regions` 调用 `from src.sampling_visualization import render_sampling_preview`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 追加：

```python
from src.sampling_visualization import render_sampling_preview


def test_render_sampling_preview_returns_base64_png():
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    image = render_sampling_preview(opt, regions, real_points)
    assert image.startswith("data:image/png;base64,")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auto_sampling.py::test_render_sampling_preview_returns_base64_png -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sampling_visualization'`

- [ ] **Step 3: 创建可视化模块**

创建 `src/sampling_visualization.py`（移植原 `visualize_sampling_points`，去掉 `animated`/`step_by_step` 死代码，改为模块级函数）：

```python
import base64
from io import BytesIO

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


def render_sampling_preview(opt, sampling_points, real_points):
    """绘制采样规划图，返回 base64 PNG 字符串。"""
    rows, cols = opt.rows, opt.cols
    num_regions = opt.num_regions
    cell_width = 1.0
    cell_height = opt.width / opt.length if opt.length else 1.0
    plot_width = cols * cell_width
    plot_height = rows * cell_height

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, plot_width)
    ax.set_ylim(0, plot_height)

    for i in range(rows + 1):
        ax.axhline(i * cell_height, color='black', linewidth=2)
    for j in range(cols + 1):
        ax.axvline(j * cell_width, color='black', linewidth=2)

    colors = plt.get_cmap('Pastel1', num_regions)
    for region_id in range(num_regions):
        region_color = colors(region_id)
        for i in range(rows):
            for j in range(cols):
                if opt.region_mask[i, j] == region_id:
                    rect = patches.Rectangle(
                        (j * cell_width, i * cell_height),
                        cell_width, cell_height,
                        edgecolor='none', facecolor=region_color, alpha=0.7
                    )
                    ax.add_patch(rect)

    ax.set_aspect('equal')

    width_scale = opt.width / (rows * cell_height) if opt.width else 1.0
    length_scale = opt.length / (cols * cell_width) if opt.length else 1.0

    for lj in opt.ljs:
        rect = patches.Rectangle(
            (lj[0] / length_scale, lj[1] / width_scale),
            (lj[2] - lj[0]) / length_scale,
            (lj[3] - lj[1]) / width_scale,
            edgecolor='red', facecolor='red', linewidth=2, alpha=0.3
        )
        ax.add_patch(rect)

    ax.add_patch(
        patches.Rectangle(
            (opt.yx[0] / length_scale, opt.yx[1] / width_scale),
            (opt.yx[2] - opt.yx[0]) / length_scale,
            (opt.yx[3] - opt.yx[1]) / width_scale,
            edgecolor='green', facecolor='None', linewidth=2
        )
    )

    for i in range(rows):
        for j in range(cols):
            center_x = (j + 0.5) * cell_width
            center_y = (i + 0.5) * cell_height
            ax.text(
                center_x, center_y + 0.2 * cell_height,
                f"({i},{j})",
                ha='center', va='center', fontsize=10, color='black'
            )

    for i in range(len(sampling_points)):
        current_row = sampling_points[i][0]
        current_col = sampling_points[i][1]
        real_point = real_points[i]
        cell_center_x = (current_col + 0.5) * cell_width
        cell_center_y = (current_row + 0.5) * cell_height
        ax.plot(
            cell_center_x, cell_center_y,
            'ro', markersize=36 * min(cell_width, cell_height),
            markeredgecolor='darkred', markeredgewidth=2, zorder=5
        )
        ax.text(
            cell_center_x, cell_center_y, f'{i + 1}',
            ha='center', va='center', color='white',
            fontweight='bold', fontsize=12, zorder=10
        )
        real_x = real_point[0] / length_scale
        real_y = real_point[1] / width_scale
        ax.plot(
            real_x, real_y, 'go', markersize=15 * min(cell_width, cell_height),
            markeredgecolor='darkred', markeredgewidth=2, zorder=6
        )
        ax.text(
            cell_center_x, cell_center_y + 0.3 * cell_height,
            f"({int(real_point[0])},{int(real_point[1])})",
            ha='center', va='center', fontsize=9, color='black'
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'长度方向 (cols) - 总长: {opt.length}mm', fontsize=12)
    ax.set_ylabel(f'宽度方向 (rows) - 总宽: {opt.width}mm', fontsize=12)

    title = f'汽车煤采样点规划 (共{rows * cols}个采样点)\n'
    title += f'网格: {rows}×{cols} ({rows}行×{cols}列)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    legend_elements = []
    for region_id in range(num_regions):
        color = colors(region_id)
        legend_elements.append(
            patches.Patch(facecolor=color, edgecolor='black', label=f'大区{region_id + 1}')
        )
    legend_elements.append(
        patches.Patch(facecolor="red", alpha=0.3, edgecolor='black', label='拉筋区域')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=12, markeredgecolor='darkred', markeredgewidth=2,
                   label='采样区域')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=12, markeredgecolor='darkred', markeredgewidth=2,
                   label='采样点')
    )
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=10, title="图例")

    info_text = f"每个单元格: {cell_width:.2f}×{cell_height:.2f}\n"
    if opt.length and opt.width:
        info_text += f"长宽比: {opt.length / opt.width:.2f}:1.00"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    ax.invert_yaxis()
    ax.invert_xaxis()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    image_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return image_base64
```

修改 `src/auto_sampling.py`：
- 删除方法：`visualize_sampling_points`、`visualize_sampling_points_animated`、`visualize_sampling_points_step_by_step`。
- 删除 import：`base64`、`BytesIO`、`matplotlib`、`matplotlib.animation`、`matplotlib.patches`、`matplotlib.pyplot`、`numpy` 如不再使用则一并删除（`generate_region_mask` 仍用 `np`，保留 `import numpy as np`）。
- 新增模块顶部 import：`from src.sampling_visualization import render_sampling_preview`。
- `realize_regions` 改为：

```python
    def realize_regions(self, regions: list[list[int, int]]) -> tuple[list[list[float]], str]:
        real_points = []
        for (r, c) in regions:
            real_points.append(self.sample_point_in_region(r, c))
        image = render_sampling_preview(self, regions, real_points)
        return real_points, image
```

- [ ] **Step 4: 运行全部测试确认通过**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: PASS（18 passed）。随后运行 `python -c "from src.auto_sampling import get_automatic_sampling_points"` 无异常。

- [ ] **Step 5: 提交**

```bash
git add src/sampling_visualization.py src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "refactor: extract sampling visualization to separate module"
```

---

### Task 5: 收尾清理 —— 死代码、demo、main.py 兼容性

**Files:**
- Modify: `src/auto_sampling.py`（删除 `optimize_sampling`、`pickle` demo、未使用 import，更新 `__main__` demo）
- Test: `tests/test_auto_sampling.py`、`main.py`（仅运行编译验证）

**Interfaces:**
- 最终 `auto_sampling.py` 对外暴露：`CoalSamplingOptimizer`、`SamplingConfig`、`SamplingError`、`get_automatic_sampling_points`、`get_automatic_sampling_regions`、`get_automatic_sampling_points_from_regions`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auto_sampling.py::test_optimize_sampling_removed -v`
Expected: FAIL — `assert not hasattr(...)` 失败（`optimize_sampling` 仍存在）。

- [ ] **Step 3: 收尾清理**

删除 `optimize_sampling` 方法（其职责已被 `plan_regions` + `realize_regions` 覆盖）。

`get_automatic_sampling_points` 改为直接组合（不再经过 `optimize_sampling`）：

```python
def get_automatic_sampling_points(car_length: int,
                                  car_width: int,
                                  car_lj: tuple = tuple(),
                                  car_kx: tuple = tuple(),
                                  config: SamplingConfig | None = None):
    logger.info(f"汽车长度：{car_length}")
    logger.info(f"汽车宽度：{car_width}")
    logger.info(f"拉筋区域：{car_lj}")
    logger.info(f"允许区域：{car_kx}")
    opt = CoalSamplingOptimizer(config=config, length=car_length, width=car_width,
                                ljs=car_lj, yx=car_kx)
    return opt.realize_regions(opt.plan_regions())
```

更新 `__main__` demo（去掉 pickle 与 matplotlib 绘图依赖）：

```python
if __name__ == '__main__':
    car_length = 11000
    car_width = 5500
    car_lj = [[2000, 100, 2200, 5400]]
    car_kx = [100, 100, 10900, 5400]
    regions = get_automatic_sampling_regions()
    real_points, image = get_automatic_sampling_points_from_regions(
        car_length, car_width, car_lj, car_kx, regions)
    print("小区:", regions)
    print("真实坐标:", real_points)
    print("图片base64前缀:", image[:40])
```

清理文件顶部 import：删除不再使用的 `base64`、`os`、`pickle`、`BytesIO`、`matplotlib*`、`random` 保留（`random.Random` 仍用）、`numpy as np` 保留（`generate_region_mask` 用）、`Tuple` 如未用则移除。确认 `from src.sampling_visualization import render_sampling_preview` 存在。

- [ ] **Step 4: 运行测试与编译验证**

Run: `pytest tests/test_auto_sampling.py -v`
Expected: PASS（20 passed）

Run: `python -m py_compile src/auto_sampling.py src/sampling_visualization.py main.py`
Expected: 无输出、退出码 0

Run: `python src/auto_sampling.py`
Expected: 打印小区列表、真实坐标、图片前缀，无异常

- [ ] **Step 5: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "refactor: remove dead code and finalize entry point composition"
```

---

## 完成标准

- `pytest tests/` 全部通过。
- `python -m py_compile src/auto_sampling.py src/sampling_visualization.py main.py` 通过。
- `python src/auto_sampling.py` demo 正常运行。
- `src/auto_sampling.py` 不再包含：`get_min_regions`、`get_valid_choice_from_region`、`is_adjacent_any`、`get_point`、`get_region_coordinates`、`get_point1`、`get_region_coordinates1`、`optimize_sampling`、两个动画可视化方法、`os.getenv('SHUFFLE_REGION')`、pickle demo。
- main.py 三个接口调用方式不变，无需修改。
