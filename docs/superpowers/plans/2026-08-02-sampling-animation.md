# 采样点逐点放置 GIF 动画 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/sampling_visualization.py` 新增 `render_sampling_animation()`，生成 18 个采样点逐点放置的 GIF 动画（base64），辅助检查规则与优化点。

**Architecture:** 抽取 `_draw_base(ax, opt)` 共享底层绘制（网格/大区底色/拉筋/允许区域/格子标签/图例），`render_sampling_preview` 与新的 `render_sampling_animation` 共用；动画用 matplotlib `FuncAnimation` + `PillowWriter` 每点 1 帧，每帧实时用 `opt.is_adjacent`/`opt._in_allowed_region`/`opt._in_lajin` 检查规则并在标题标注。

**Tech Stack:** Python 3.x, matplotlib（FuncAnimation + PillowWriter，环境已确认可用）, numpy, pytest。

## Global Constraints

- 新函数 `render_sampling_animation(opt, real_points, fps: int = 1, interval: int = 1000) -> str` 定义在 `src/sampling_visualization.py`，返回 `data:image/gif;base64,...` 字符串。
- `len(real_points) != opt.rows * opt.cols` 时抛 `ValueError`。
- `render_sampling_preview` 的返回行为必须保持与重构前完全一致（有回归测试）。
- 共用帮助函数命名为 `_draw_base(ax, opt)`，返回 `(width_scale, length_scale)` 供坐标换算。
- 帧数 = `len(real_points)`（每点 1 帧）。当前点小区 `(r,c)`：`(r+c)%2==0` → "黑格"阶段，否则 "白格"阶段。
- 规则检查：相邻性用 `opt.is_adjacent(小区点)`；允许区域/拉筋用 `opt._in_allowed_region(x,y)` / `opt._in_lajin(x,y)`。
- 不新增 main.py/FastAPI 接口；不改 `auto_sampling.py` 的 `__main__`；不引入 imageio 等新依赖。
- 测试文件：`tests/test_auto_sampling.py`。运行命令 `python -m pytest tests/test_auto_sampling.py -v`。

---

### Task 1: 抽取 `_draw_base` 共享底层绘制

**Files:**
- Modify: `src/sampling_visualization.py`（抽取 `_draw_base`，重构 `render_sampling_preview` 调用它）
- Test: `tests/test_auto_sampling.py`（新增回归测试）

**Interfaces:**
- Produces: 模块级 `_draw_base(ax, opt) -> tuple[float, float]`，返回 `(width_scale, length_scale)`。`render_sampling_preview` 重构为：先 `width_scale, length_scale = _draw_base(ax, opt)`，再画点。
- 不改变 `render_sampling_preview(opt, sampling_points, real_points) -> str` 的签名与返回值。

- [ ] **Step 1: 写失败回归测试**

在 `tests/test_auto_sampling.py` 末尾追加：

```python
def test_render_sampling_preview_unchanged_after_draw_base_refactor():
    from src.sampling_visualization import render_sampling_preview
    opt = CoalSamplingOptimizer(length=11000, width=5500,
                                ljs=[[2000, 100, 2200, 5400]],
                                yx=[100, 100, 10900, 5400])
    regions = opt.plan_regions()
    real_points = [opt.sample_point_in_region(r, c) for (r, c) in regions]
    image = render_sampling_preview(opt, regions, real_points)
    assert image.startswith("data:image/png;base64,")
```

- [ ] **Step 2: 运行测试确认通过（基线）**

Run: `python -m pytest tests/test_auto_sampling.py::test_render_sampling_preview_unchanged_after_draw_base_refactor -v`
Expected: PASS（重构前基线，当前实现已满足）

- [ ] **Step 3: 重构 `src/sampling_visualization.py`**

将文件替换为以下内容（`render_sampling_preview` 改为调用 `_draw_base`，行为不变）：

```python
import base64
from io import BytesIO

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


def _draw_base(ax, opt):
    """绘制静态底层（网格、大区底色、拉筋、允许区域、格子标签、图例、坐标轴）。

    返回 (width_scale, length_scale) 供真实坐标到绘图坐标的换算。
    """
    rows, cols = opt.rows, opt.cols
    num_regions = opt.num_regions
    cell_width = 1.0
    cell_height = opt.width / opt.length if opt.length else 1.0
    plot_width = cols * cell_width
    plot_height = rows * cell_height

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

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'长度方向 (cols) - 总长: {opt.length}mm', fontsize=12)
    ax.set_ylabel(f'宽度方向 (rows) - 总宽: {opt.width}mm', fontsize=12)

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

    return width_scale, length_scale


def render_sampling_preview(opt, sampling_points, real_points):
    """绘制采样规划图，返回 base64 PNG 字符串。"""
    rows, cols = opt.rows, opt.cols
    cell_width = 1.0
    cell_height = opt.width / opt.length if opt.length else 1.0

    fig, ax = plt.subplots(figsize=(14, 8))
    width_scale, length_scale = _draw_base(ax, opt)

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

    title = f'汽车煤采样点规划 (共{rows * cols}个采样点)\n'
    title += f'网格: {rows}×{cols} ({rows}行×{cols}列)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

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

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（原 19 + 新增 1 = 20 passed）

- [ ] **Step 5: 提交**

```bash
git add src/sampling_visualization.py tests/test_auto_sampling.py
git commit -m "refactor: extract shared _draw_base for preview and animation"
```

---

### Task 2: 实现 `render_sampling_animation`

**Files:**
- Modify: `src/sampling_visualization.py`（新增 `render_sampling_animation`）
- Test: `tests/test_auto_sampling.py`（新增 2 个测试）

**Interfaces:**
- Consumes: `_draw_base(ax, opt)`（Task 1），`opt.is_adjacent`、`opt._in_allowed_region`、`opt._in_lajin`（`CoalSamplingOptimizer` 现有方法）。
- Produces: `render_sampling_animation(opt, real_points, fps=1, interval=1000) -> str`（base64 GIF，`data:image/gif;base64,...` 前缀）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_auto_sampling.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_auto_sampling.py::test_render_sampling_animation_returns_base64_gif tests/test_auto_sampling.py::test_render_sampling_animation_wrong_point_count_raises -v`
Expected: FAIL — `ImportError: cannot import name 'render_sampling_animation'` 或 AttributeError。

- [ ] **Step 3: 实现动画函数**

在 `src/sampling_visualization.py` 末尾（`render_sampling_preview` 之后）追加：

```python
def render_sampling_animation(opt, real_points, fps: int = 1, interval: int = 1000) -> str:
    """生成逐点放置采样点的 GIF 动画，返回 base64 GIF 字符串。

    每帧放置 1 个采样点（帧数 = len(real_points)）。每帧在标题区标注：
    当前点序号、小区(row,col)、真实坐标(mm)、黑格/白格阶段，以及规则检查结果
    （相邻性、是否落在允许区域、是否避开拉筋）。
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    expected = opt.rows * opt.cols
    if len(real_points) != expected:
        raise ValueError(f"real_points 长度 {len(real_points)} 必须等于 {expected}")

    rows, cols = opt.rows, opt.cols
    cell_width = 1.0
    cell_height = opt.width / opt.length if opt.length else 1.0
    total = len(real_points)

    fig, ax = plt.subplots(figsize=(14, 8))
    width_scale, length_scale = _draw_base(ax, opt)

    # 动态层：已放置的小区红圈、白字序号、真实坐标绿点
    cell_markers = []
    cell_labels = []
    real_markers = []
    coord_labels = []
    status_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    def _clear_dynamic():
        for artist in cell_markers + cell_labels + real_markers + coord_labels:
            artist.remove()
        cell_markers.clear()
        cell_labels.clear()
        real_markers.clear()
        coord_labels.clear()

    def _place_point(k):
        (r, c) = sampling_points[k]
        (x, y) = real_points[k]
        cell_cx = (c + 0.5) * cell_width
        cell_cy = (r + 0.5) * cell_height
        cell_markers.append(
            ax.plot(cell_cx, cell_cy, 'ro',
                    markersize=36 * min(cell_width, cell_height),
                    markeredgecolor='darkred', markeredgewidth=2, zorder=5)[0]
        )
        cell_labels.append(
            ax.text(cell_cx, cell_cy, f'{k + 1}', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=12, zorder=10)
        )
        rx, ry = x / length_scale, y / width_scale
        real_markers.append(
            ax.plot(rx, ry, 'go', markersize=15 * min(cell_width, cell_height),
                    markeredgecolor='darkred', markeredgewidth=2, zorder=6)[0]
        )
        coord_labels.append(
            ax.text(cell_cx, cell_cy + 0.3 * cell_height,
                    f"({int(x)},{int(y)})",
                    ha='center', va='center', fontsize=9, color='black')
        )

    def update(frame):
        _clear_dynamic()
        for k in range(frame + 1):
            _place_point(k)

        (r, c) = sampling_points[frame]
        (x, y) = real_points[frame]
        phase = "黑格" if (r + c) % 2 == 0 else "白格"

        # 规则检查：相邻性（只对比同相位点，黑格只查黑格、白格只查白格。
        # 跨相位相邻是全覆盖规划的必然结构，不计为违反，避免白格阶段必然全红）
        violations = []
        if frame > 0:
            prev_same_phase = [p for p in sampling_points[:frame] if (p[0] + p[1]) % 2 == (r + c) % 2]
            if any(opt.is_adjacent((r, c), p) for p in prev_same_phase):
                violations.append("相邻性")
        if not opt._in_allowed_region(x, y):
            violations.append("不在允许区域")
        if opt._in_lajin(x, y):
            violations.append("落在拉筋")

        if violations:
            status = f"⚠️ 规则违反: {'、'.join(violations)}"
            color = 'red'
        else:
            status = "✅ 规则通过"
            color = 'green'
        status_text.set_text(status)
        status_text.set_color(color)

        title = (f"第 {frame + 1}/{total} 点 · 小区({r},{c}) · 坐标({int(x)},{int(y)})mm · "
                 f"阶段:{phase}\n{status}")
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    # 需要采样小区坐标：由 real_points 推导不可行，改为调用方传入。
    # 这里使用 opt.plan_regions() 重建小区序列（确定性），
    # 与 get_automatic_sampling_points 的调用顺序一致。
    sampling_points = opt.plan_regions()

    anim = FuncAnimation(fig, update, frames=total, interval=interval, blit=False)
    buf = BytesIO()
    anim.save(buf, writer=PillowWriter(fps=fps), dpi=100)
    buf.seek(0)
    gif_base64 = "data:image/gif;base64," + base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return gif_base64
```

注意：`update` 中通过 `opt.plan_regions()` 重建小区序列。因为 `plan_regions` 是确定性的（shuffle 关闭时），且真实坐标来自 `realize_regions(opt.plan_regions())`，两者一一对应。若调用方传入 `shuffle_regions=True` 的配置，需保证调用动画前已固定同一批 `regions` —— 本函数不接受 regions 参数，依赖确定性；这是设计约定（见 spec 第 4 节）。若你需要支持任意 regions，说明理由。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_auto_sampling.py -v`
Expected: 全部通过（22 passed）

- [ ] **Step 5: 提交**

```bash
git add src/sampling_visualization.py tests/test_auto_sampling.py
git commit -m "feat: add step-by-step sampling point GIF animation"
```

---

## 完成标准

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过（22 passed）。
- `python -m py_compile src/sampling_visualization.py` 通过。
- 手动验证：`python -c "from src.sampling_visualization import render_sampling_animation; from src.auto_sampling import CoalSamplingOptimizer; opt = CoalSamplingOptimizer(length=11000, width=5500, ljs=[[2000,100,2200,5400]], yx=[100,100,10900,5400]); regions = opt.plan_regions(); real_points = [opt.sample_point_in_region(r,c) for r,c in regions]; print(render_sampling_animation(opt, real_points)[:30])"` 输出 `data:image/gif;base64,...` 前缀。
