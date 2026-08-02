# 自动采样模块重构设计

日期：2026-08-02
分支：`refactor/auto-sampling`

## 背景

`src/auto_sampling.py`（626 行）存在硬编码、死代码、重复逻辑、类型标注错误和算法不确定性等问题。本次在 `main` 分支基础上新建 `refactor/auto-sampling` 分支进行重构。

## 目标

1. 消除硬编码，网格参数与算法参数集中为可配置项。
2. 用确定性棋盘式选点替代随机重试算法，保证可行且可复现。
3. 清理死代码与重复逻辑，修正类型标注。
4. 对外入口函数签名保持向后兼容（追加可选 `config` 参数）。
5. 补充 pytest 单元测试。

## 问题清单（现状）

- 硬编码：`rows=3, cols=6, row_span=3, col_span=2` 在 3 个入口函数重复；`get_region_coordinates` 内 `//6 %6` 魔法数 6；`is_adjacent_constraint=i*num_regions<=6` 魔法数 6；`max_try_times=100`、`max_attempts=50`、`max_try_count=100`。
- 死代码：`get_point`（无限循环、未使用）、`get_region_coordinates`、`visualize_sampling_points_animated`、`visualize_sampling_points_step_by_step`、`get_point1` 的 `delta=0` 死参数。
- 重复：3 个入口函数各自实现"重试 100 次"逻辑。
- 坏味道：`os.getenv('SHUFFLE_REGION')` 内联读环境变量；类型标注错误（`get_min_regions -> list[list[int,int]]` 实际返回元组列表；`optimize_sampling -> np.ndarray` 实际返回 tuple）。
- 算法：区域选点依赖随机重试（区内 50 次/区，失败整体重来 100 次），不可复现、可能失败。

## 设计

### 1. `SamplingConfig` 配置类

新增 dataclass，放于 `src/auto_sampling.py`：

```python
@dataclass(frozen=True)
class SamplingConfig:
    grid_rows: int = 3
    grid_cols: int = 6
    region_row_span: int = 3
    region_col_span: int = 2
    shuffle_regions: bool = False      # 替代 SHUFFLE_REGION 环境变量
    seed: int | None = None            # 随机种子，可复现
    max_coordinate_attempts: int = 100 # 坐标生成尝试上限
```

校验（`__post_init__`）：
- `grid_rows % region_row_span == 0` 且 `grid_cols % region_col_span == 0`
- `num_regions = (rows//row_span) * (cols//col_span)`，要求 `grid_rows * grid_cols % num_regions == 0`

### 2. 确定性选点算法（替代 `get_min_regions`）

关键洞察：棋盘式着色 `(r+c)%2`。黑格集（9 个）两两不相邻，且每个大区恰好 3 个黑格（区域为 2 列×3 行）。

新方法 `plan_regions() -> list[tuple[int, int]]`：
- 轮次 `rounds = grid_rows * grid_cols // num_regions`（=6），每轮每个大区放 1 点。
- 前 `num_regions` 轮（=3）：每区分配黑格（保证互不相邻）→ 前 9 点。
- 后 `num_regions` 轮（=3）：每区分配白格填满 → 后 9 点，实现 18 格全覆盖、无重复。
- `shuffle_regions=True` 时，每轮按 `seed` 打乱大区轮序。
- 不再抛出 `ValueError` 重试：算法确定可行。

保留的辅助：`generate_region_mask`、`is_adjacent`（语义化相邻判定）。

### 3. 坐标生成

- `get_point1` 重命名为 `sample_point_in_region(row, col)`，去掉 `delta=0` 死参数，直接用行列计算格边界。
- 保留允许区域（`yx`）与拉筋（`ljs`）约束，失败抛自定义 `SamplingError`（含区域、尝试次数信息）。
- 尝试次数由 `config.max_coordinate_attempts` 控制。

### 4. 入口函数（向后兼容）

保留函数名与位置，追加可选 `config` 参数：

```python
def get_automatic_sampling_points(car_length, car_width, car_lj=(), car_kx=(), config=None) -> tuple
def get_automatic_sampling_regions(config=None) -> list
def get_automatic_sampling_points_from_regions(car_length, car_width, car_lj=(), car_kx=(), regions=(), config=None) -> tuple
```

- 去掉各自的重试循环（确定性算法不再需要）。
- `config=None` 时使用默认 `SamplingConfig()`。
- 修复返回类型标注。

### 5. 可视化拆分

- `visualize_sampling_points` 保留为规划结果可视化（返回 base64）。
- 死代码 `visualize_sampling_points_animated`、`visualize_sampling_points_step_by_step` 删除。
- 不做独立文件拆分（保持文件内聚，避免过度拆分），但删除 `animation` 相关 import。

### 6. 删除项

- `get_point`（无限循环、未使用）
- `get_region_coordinates`（含 `//6 %6` 魔法数）
- `optimize_sampling`、`optimize_sampling_points_from_regions`（合并进入口逻辑，如仍需要则保留一个内部方法）
- `os.getenv('SHUFFLE_REGION')` 读取逻辑
- `main` 块中的 `pickle` 示例代码（改为简洁 demo）

### 7. main.py 适配

main.py 的三个接口（`/api/auto_sampling` 废弃、`/api/auto_sampling_regions`、`/api/auto_sampling_points`）保持调用方式不变，仅内部可透传默认 `SamplingConfig()`。若入口签名追加 `config` 可选参数，则 main.py 无需改动或仅最小改动。

### 8. 测试（`tests/test_auto_sampling.py`）

用 pytest 验证：
- 分区划分：默认配置下 3 个区域、各 6 格。
- `plan_regions` 返回 18 个点、全覆盖无重复、每区恰 6 点。
- 前 9 点两两不相邻。
- `seed` 相同时结果可复现；`seed=None` 时多次结果可不同。
- `sample_point_in_region` 生成的点落在允许区域内、避开拉筋。
- 非法配置（行/列不整除）抛错。
- 对外入口函数返回结构与重构前一致（`(points, image)` / `regions`）。

## 验证方式

- `pytest tests/` 通过。
- 运行 `python src/auto_sampling.py` demo 无异常。
- `python -m py_compile src/auto_sampling.py main.py` 通过。

## 风险与说明

- 确定性算法改变随机选点的"随机感"，但通过 `shuffle_regions` + `seed` 可复现多样性。
- 默认行为（shuffle 关闭、seed 无）下结果仍确定，与原先随机结果不同——这是有意的算法改进。
- main.py 中已废弃的 `/api/auto_sampling` 接口不删除（保持 API 兼容），仅内部走新逻辑。
