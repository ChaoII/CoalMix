# 采样小区随机化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `plan_regions()` 每次调用生成随机不同的采样小区布局（黑格相位随机 + 轮序随机），默认 `shuffle_regions=True`，并修复动画 desync（`render_sampling_animation` 增加 `regions` 可选参数）。

**Architecture:** `plan_regions()` 把固定黑格判定 `(r+c)%2` 改为 `(r+c+phase_shift)%2`，`phase_shift` 在 `shuffle_regions=True` 时每次 `_rng.randint(0,1)`；`shuffle_regions=False` 时 shift 固定 0、不 shuffle，保持确定性。动画函数增加 `regions` 参数让调用方传入小区序列，消除随机化后内部重调 `plan_regions()` 的错位。

**Tech Stack:** Python 3.x, numpy, matplotlib, pytest。

## Global Constraints

- `SamplingConfig.shuffle_regions` 默认改为 `True`（原 `False`），`seed` 默认 `None`（每次随机）。
- `plan_regions() -> list[tuple[int,int]]`：黑格判定 `(r+c+shift)%2`，`shuffle_regions=True` 时 `shift = self._rng.randint(0,1)`、黑格/白格列表内部 `self._rng.shuffle`、每轮 `self._rng.shuffle(region_order)`；`shuffle_regions=False` 时 `shift=0`、不 shuffle（确定性，向后兼容）。黑格集只有 2 种（全局 shift），多样性靠内部排列与轮序。
- 约束保持：18 格全覆盖无重复、每区 6 点、前 `num_regions` 轮（前 9 点）黑格互不相邻、后 9 点白格填满。
- `render_sampling_animation(opt, real_points, regions=None, fps=1, interval=1000) -> str`：`regions` 传入时用其作为小区序列；`None` 时内部 `opt.plan_regions()`。
- 测试文件 `tests/test_auto_sampling.py`，运行命令 `python -m pytest tests/test_auto_sampling.py -v`。
- 不新增依赖；不改 main.py / FastAPI 接口。

---

### Task 1: `plan_regions` 随机化 + 默认配置变更

**Files:**
- Modify: `src/auto_sampling.py`（`SamplingConfig.shuffle_regions` 默认值、`plan_regions` 方法）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: `SamplingConfig.shuffle_regions`、`seed`、`_rng`（现有）。
- Produces: 随机化的 `plan_regions()`；`SamplingConfig().shuffle_regions == True`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 中：
1) 修改 `test_default_config_matches_original_hardcoded_values`（第 14 行 `assert cfg.shuffle_regions is False` 改为 `True`）：

```python
def test_default_config_matches_original_hardcoded_values():
    cfg = SamplingConfig()
    assert cfg.grid_rows == 3
    assert cfg.grid_cols == 6
    assert cfg.region_row_span == 3
    assert cfg.region_col_span == 2
    assert cfg.shuffle_regions is True
    assert cfg.seed is None
    assert cfg.max_coordinate_attempts == 100
```

2) 在文件末尾追加两个新测试：

```python
def test_plan_regions_varied_across_calls_with_seed():
    p1 = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=1)).plan_regions()
    p2 = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=True, seed=2)).plan_regions()
    assert p1 != p2, "不同 seed 的完整放置序列应不同"


def test_plan_regions_deterministic_when_shuffle_disabled():
    a = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=False)).plan_regions()
    b = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=False)).plan_regions()
    assert a == b
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: FAIL — `test_default_config...`（shuffle 仍 False）+ `test_plan_regions_varied_across_calls_with_seed`（当前 seed 1/2 集合相同，`p1==p2` 断言失败）+ `test_plan_regions_deterministic_when_shuffle_disabled` 可能通过。

- [ ] **Step 3: 实现**

修改 `src/auto_sampling.py` 中 `SamplingConfig`：

```python
    shuffle_regions: bool = True
    seed: int | None = None
    max_coordinate_attempts: int = 100
```

替换 `plan_regions` 方法（整块替换；注意黑格/白格列表内部 shuffle 是实现要点，保证 2 种黑格集下布局仍随机）：

```python
    def plan_regions(self) -> list[tuple[int, int]]:
        """随机化规划采样小区。

        全局黑格相位随机（(r+c+shift)%2，shift 每次随机 0/1）保证前 num_regions 轮
        黑格互不相邻；黑格/白格集合内部与区域轮序均随机打乱，使每次调用布局差异大。
        前 num_regions 轮分配黑格，后 num_regions 轮用白格填满，实现全覆盖、无重复、
        每区均匀。seed 固定时可复现。shuffle_regions=False 时完全确定性。
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

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过。注意 `test_plan_regions_first_9_points_mutually_non_adjacent` 仍需通过——前 9 点全是黑格（同 shift 下互不相邻），此测试逻辑不变应通过。`test_plan_regions_same_phase_points_never_adjacent` 用 `(r+c)%2` 判定相位，但随机化后黑格相位可能是 `(r+c+1)%2`——此测试可能失败。

若 `test_plan_regions_same_phase_points_never_adjacent` 失败，按 Step 4b 处理。

- [ ] **Step 4b: 修复相位相关的既有测试（如失败）**

若 `test_plan_regions_same_phase_points_never_adjacent` 失败（因为黑格相位随机导致 `(r+c)%2` 不再等于实际相位），修改该测试为固定确定性配置，验证"确定性布局下同相位点互不相邻"：

```python
def test_plan_regions_same_phase_points_never_adjacent():
    opt = CoalSamplingOptimizer(SamplingConfig(shuffle_regions=False))
    points = opt.plan_regions()
    for i, (r, c) in enumerate(points):
        prev_same_phase = [p for p in points[:i]
                           if (p[0] + p[1]) % 2 == (r + c) % 2]
        for p in prev_same_phase:
            assert not opt.is_adjacent((r, c), p)
```

Run: `python -m pytest tests/test_auto_sampling.py::test_plan_regions_same_phase_points_never_adjacent -v`
Expected: PASS

- [ ] **Step 5: 运行全部测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（25 passed = 原 23 + 新增 2）

- [ ] **Step 6: 提交**

```bash
git add src/auto_sampling.py tests/test_auto_sampling.py
git commit -m "feat: randomize sampling region layout by default"
```

---

### Task 2: 动画 `regions` 参数 + 生成多组 GIF

**Files:**
- Modify: `src/sampling_visualization.py`（`render_sampling_animation` 加 `regions` 参数、更新 docstring、`sampling_points` 来源改为参数）
- Test: `tests/test_auto_sampling.py`

**Interfaces:**
- Consumes: Task 1 的 `plan_regions()`。
- Produces: `render_sampling_animation(opt, real_points, regions=None, fps=1, interval=1000) -> str`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_auto_sampling.py` 中，把 `test_render_sampling_animation_returns_base64_gif` 改为传 `regions`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_auto_sampling.py::test_render_sampling_animation_returns_base64_gif -v`
Expected: FAIL — `TypeError: render_sampling_animation() got an unexpected keyword argument 'regions'`

- [ ] **Step 3: 实现**

在 `src/sampling_visualization.py` 修改 `render_sampling_animation`：

签名改为 `def render_sampling_animation(opt, real_points, regions=None, fps: int = 1, interval: int = 1000) -> str:`。

更新 docstring，追加 desync 说明：

```python
    """生成逐点放置采样点的 GIF 动画，返回 base64 GIF 字符串。

    每帧放置 1 个采样点（帧数 = len(real_points)）。每帧在标题区标注：
    当前点序号、小区(row,col)、真实坐标(mm)、黑格/白格阶段，以及规则检查结果
    （相邻性、是否落在允许区域、是否避开拉筋）。

    regions: 采样小区序列，与 real_points 一一对应。必须传入——默认配置下
    plan_regions() 每次调用返回随机不同布局，若省略本参数，函数内部重调
    opt.plan_regions() 得到的小区序列可能与调用方生成 real_points 时的不同，
    导致小区标记与真实坐标错位。缺省为 None（仅在确定性子场景下可用）。

    interval 仅影响交互式显示，返回的 GIF 播放速度由 fps 决定。
    """
```

把 `sampling_points` 的定义从"函数末尾 `sampling_points = opt.plan_regions()`"改为在函数开头：

```python
    if regions is None:
        sampling_points = opt.plan_regions()
    else:
        sampling_points = list(regions)
```

放在 `total = len(real_points)` 附近（`update`/`_place_point` 闭包使用前）。其余逻辑不变。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（25 passed）

- [ ] **Step 5: 生成多组 GIF 验证随机性**

Run: 创建一个临时脚本生成 4 组不同 seed 的 GIF（放在 `C:\Users\aichao\AppData\Local\Temp\opencode` 之外，直接放项目根目录，命名 `sampling_anim_seed1.gif` 等）：

```python
import base64
from src.auto_sampling import CoalSamplingOptimizer, SamplingConfig
from src.sampling_visualization import render_sampling_animation

for seed in [1, 2, 3, 4]:
    cfg = SamplingConfig(shuffle_regions=True, seed=seed)
    opt = CoalSamplingOptimizer(cfg, length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    gif = render_sampling_animation(opt, real_points, regions=regions, fps=1)
    raw = base64.b64decode(gif.split(",", 1)[1])
    with open(f"sampling_anim_seed{seed}.gif", "wb") as f:
        f.write(raw)
    print(f"seed{seed}: {len(raw)} bytes, header {raw[:6]}, first region {regions[0]}")
```

Expected: 4 个文件生成，header 均为 `b'GIF89a'`，且 `first region` 各不同（验证布局随机）。

（此脚本是验证产物，不提交 git。4 个 GIF 文件也暂不提交，留给用户查看。）

- [ ] **Step 6: 提交**

```bash
git add src/sampling_visualization.py tests/test_auto_sampling.py
git commit -m "feat: accept regions param in animation to avoid desync"
```

---

## 完成标准

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过（25 passed）。
- `python -m py_compile src/auto_sampling.py src/sampling_visualization.py` 通过。
- 手动验证：
  - `SamplingConfig().shuffle_regions is True`。
  - 不同 seed 下 `plan_regions()` 完整序列不同（内部排列与轮序随机）；同 seed 完全一致。
  - `SamplingConfig(shuffle_regions=False)` 两次调用结果一致（确定性）。
  - 4 个 `sampling_anim_seed<N>.gif` 生成、header 为 GIF89a、布局顺序不同。
