# 采样点逐点放置 GIF 动画设计

日期：2026-08-02
分支：`refactor/auto-sampling`

## 背景

`src/auto_sampling.py` 重构后，`plan_regions()` 确定性输出 18 个采样小区（前 9 个黑格互不相邻，后 9 个白格填满），`realize_regions()` 生成真实坐标。用户需要逐点放置动画（每点 1 帧，共 18 帧），直观检查是否违反规则（相邻性、拉筋/允许区域）以及寻找优化点。

## 目标

1. 在 `src/sampling_visualization.py` 新增 `render_sampling_animation()`，返回 base64 GIF。
2. 每帧展示：网格、大区底色、拉筋/允许区域、已放置的点（小区红圈+真实坐标绿点）、点序号与坐标、规则检查结果、进度与黑/白格阶段。
3. 复用现有绘制逻辑，不复制代码；`render_sampling_preview` 行为不变。
4. 补充 pytest 单测。

## 设计

### 1. 函数签名与产物

```python
def render_sampling_animation(opt, real_points, fps: int = 1, interval: int = 1000) -> str:
    """生成逐点放置采样点的 GIF 动画，返回 base64 GIF 字符串。"""
```

- `opt`：`CoalSamplingOptimizer` 实例。
- `real_points`：长度 = `opt.rows * opt.cols` 的真实坐标列表。
- `fps`：GIF 帧率（保存时用）；`interval`：FuncAnimation 帧间隔毫秒。
- 返回 `data:image/gif;base64,...` 字符串。
- 输入校验：`len(real_points) != opt.rows * opt.cols` 时抛 `ValueError`。

### 2. 共享底层绘制

抽取模块级帮助函数，`render_sampling_preview` 与动画共用：

```python
def _draw_base(ax, opt) -> tuple[float, float]:
    """绘制静态底层（网格、大区底色、拉筋、允许区域、格子标签、图例、坐标轴）。

    返回 (width_scale, length_scale) 供坐标换算。
    """
```

`render_sampling_preview` 改为调用 `_draw_base` 后再画点。行为必须与重构前一致（有回归测试保护）。

### 3. 动画帧内容

- 共 18 帧（每点 1 帧），`FuncAnimation(fig, update, frames=len(real_points), interval=interval, blit=False)`。
- `update(k)`（k 从 0 开始）：
  - 重画前 k+1 个点的标记（每次清空动态 artist 再重画，保证正确）。
  - 计算并显示当前点 `p_k` 的规则检查结果：
    - 相邻性：**只对比同相位（同奇偶性）的之前小区**用 `opt.is_adjacent` 检查（黑格只查黑格、白格只查白格），若相邻 → 红字警告。跨相位相邻是全覆盖规划的必然结构（黑格先行、白格填满时每个白格必与某黑格相邻），不计为违反，避免动画后 9 帧必然全红而误导。
    - 允许区域/拉筋：当前真实坐标不满足 `opt._in_allowed_region` 或落在 `opt._in_lajin` → 红字警告。
    - 正常 → 绿字"规则通过"。
  - 标题：`第 {k+1}/18 点 · 阶段: 黑格/白格`，标注当前格 `(row, col)`、真实坐标 `(x, y) mm`。
- 保存：`PillowWriter(fps=fps)` → `savefig` 到 `BytesIO` → base64。`plt.close(fig)` 释放资源。

### 4. 黑格/白格阶段判定

当前点小区 `(r, c)`：`(r + c) % 2 == 0` → 黑格阶段（前 9 点）；否则白格阶段（后 9 点）。

### 5. 测试（`tests/test_auto_sampling.py`）

- `test_render_sampling_animation_returns_base64_gif`：生成动画，断言以 `data:image/gif;base64,` 开头，base64 解码后字节以 `GIF89a` 开头。
- `test_render_sampling_animation_wrong_point_count_raises`：`real_points` 长度错误时抛 `ValueError`。
- 现有预览回归测试继续通过（`_draw_base` 重构后 `render_sampling_preview` 不变）。

## 不做的事

- 不新增动画 API 到 main.py / FastAPI。
- 不改动 `auto_sampling.py` 的 `__main__` demo。
- 不引入 imageio 等新依赖（用 matplotlib 自带 `PillowWriter`）。

## 验证方式

- `pytest tests/test_auto_sampling.py` 全部通过。
- `python -m py_compile src/sampling_visualization.py` 通过。
- 手动示例：`python -c` 生成 GIF 到临时文件可打开。
