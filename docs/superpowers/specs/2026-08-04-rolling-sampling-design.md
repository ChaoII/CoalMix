# 跨车滚动采样设计

日期：2026-08-04
分支：`feature/rolling-sampling`

## 背景

同一批煤跨多车累进采样。每车实际采样点数由前端/操作员决定（`need`），18 个采样点滚动循环分配：优先采未采的点，18 个用完后循环回 1。真实坐标由现有 `sample_point_in_region` 生成（行为不变：随机 + 避开拉筋 + 落在允许区域）。

## 目标

1. 新增两个函数到 `src/auto_sampling.py`（不新增 FastAPI 接口，由用户自行配置）。
2. 方法1 `get_automatic_sampling_regions_rolling(used, need, config)`：跨车滚动分配采样点，返回 `(编号列表, 对应格子列表)`。
3. 方法2 复用现有 `get_automatic_sampling_points_from_regions(...)` 生成真实坐标。
4. 编号 1-18 固定映射格子，映射在服务启动时建立（模块级缓存，服务生命周期不变）。

## 需求确认（来自澄清）

### 接口行为

前端传 `used`（当前轮已用编号列表）+ `need`（本车要采点数），返回本车应采的编号序列。

用户例子（4 轮）：
| 轮 | used | need | 返回 |
|----|------|------|------|
| 1 | [] | 5 | 1,2,3,4,5 |
| 2 | [1..5] | 6 | 6,7,8,9,10,11 |
| 3 | [1..11] | 5 | 12,13,14,15,16 |
| 4 | [1..16] | 6 | 17,18 + 换轮补 1,2,3,4 |

### 已确认规则

- **编号固定映射格子（列优先、从右往左）**：编号 1-18 按"列从右到左、每列内行从小到大"固定映射格子（确定性，与 seed 无关）。编号 1-6 = 最右大区（列 4,5），7-12 = 中间大区（列 2,3），13-18 = 最左大区（列 0,1）。模块级缓存，跨车不变。
- **used 语义**：当前轮已用编号，无重复、最多 18 个。
- **换轮规则**：未用编号数 ≥ need 时按编号序取前 need 个；未用数 < need 时先取完全部未用，再清空 used 从编号 1 重新取剩余（返回可跨轮，如 `[17,18,1,2,3,4]`）。
- **返回顺序**：要返回的编号按**当次 `plan_regions()` 格子序**排序（棋盘格不相邻处理，返回编号可能不是连续递增）。
- **返回结构**：`(编号列表, 对应格子列表)`，编号→格子映射在服务端，前端无需理解。
- **方法2**：复用现有 `get_automatic_sampling_points_from_regions(car_length, car_width, car_lj, car_kx, regions)`，接收格子列表生成真实坐标。

## 设计

### 方法1：`get_automatic_sampling_regions_rolling`

```python
_NUMBERING: dict[int, tuple[int, int]] | None = None  # 编号 -> 格子（启动时建立）


def get_automatic_sampling_regions_rolling(
        used: list[int] | None = None,
        need: int = 0,
        config: SamplingConfig | None = None) -> tuple[list[int], list[list[int]]]:
    """跨车滚动采样：返回 (本车应采的编号列表, 对应格子列表)。

    used: 当前轮已用编号（无重复，1-18 范围）。need: 本车要采点数。
    未用编号数不足 need 时，先取全部未用，再清空 used 从编号 1 换轮重取。
    返回编号按当次 plan_regions() 格子序排序。
    """
```

逻辑：
1. 首次调用建立 `_NUMBERING`（若为 None）：按"列从右到左、每列内行从小到大"构建编号 1-18 → 格子映射（确定性，不依赖 `plan_regions()` 随机顺序）。
2. `used` 转 set 去重（防御性）；`need <= 0` 返回空。
3. 未用编号 = `[i for i in 1..18 if i not in used]`，按编号序。
4. 若 `len(未用) >= need`：分配 = 未用[:need]。
5. 否则：分配 = 未用 + [1, 2, ..., need - len(未用)]（换轮，重新从 1 编号）。
6. 当次 `plan_regions()` → 格子序 `current`；按格子在 `current` 中的先后对分配编号排序。
7. 返回 `(排序后的编号, [编号→格子])`。

### 方法2：复用现有函数

`get_automatic_sampling_points_from_regions(...)`（已在 `src/auto_sampling.py`），无需修改。

### 编号映射缓存线程安全性

FastAPI 多线程场景下，`_NUMBERING` 首次赋值可能竞态。用模块级 `_NUMBERING_LOCK = threading.Lock()` 保护首次初始化。

## 测试（`tests/test_auto_sampling.py`）

- `test_rolling_round1`: used=[] need=5 → 返回 5 个编号，集合 `{1,2,3,4,5}`，格子长度 5。
- `test_rolling_round2`: used=[1..5] need=6 → 返回编号集合 `{6..11}`。
- `test_rolling_round3`: used=[1..11] need=5 → 返回编号集合 `{12..16}`。
- `test_rolling_round4_wrap`: used=[1..16] need=6 → 返回编号集合 `{1,2,3,4,17,18}`（含跨轮）。
- `test_rolling_empty_need`: need=0 → 空。
- `test_rolling_used_none`: used=None need=3 → 返回 `{1,2,3}`。
- `test_rolling_returns_matching_cells`: 返回编号与格子一一对应（通过 `_NUMBERING`）。
- `test_rolling_numbering_fixed_across_calls`: 两次调用，编号→格子映射一致（缓存生效）。
- `test_rolling_ordering_follows_current_plan`: 返回编号按当次 plan 格子序排序（验证排序逻辑）。
- `test_rolling_numbering_column_first_right_to_left`: 编号映射 = 列优先从右往左；1-6 最右大区(列4,5)、7-12 中间(列2,3)、13-18 最左(列0,1)。

## 不做的事

- 不新增 FastAPI 接口。
- 不修改 `plan_regions`、`sample_point_in_region`、`get_automatic_sampling_points_from_regions`。
- 不引入数据库/持久化状态（前端维护 used）。

## 验证方式

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过。
- 手动：复现用户 4 轮例子逐轮断言。
- `python -m py_compile src/auto_sampling.py` 通过。
