# 行交替启发式消除同车相邻设计

日期：2026-08-05
分支：`feature/rolling-sampling`

## 背景

跨车滚动采样中，同一车返回的多个采样点可能出现物理相邻（如编号 2=(1,5) 和 5=(1,4) 同大区相邻）。原因是同一大区的黑格与白格在空间上可能相邻，若落到同一车窗口则产生相邻采样点。

## 分析（已验证）

- 同一大区（2列×3行）内，黑格与白格**只有空间相邻的才对相邻**（如 12=(2,2) 黑 与 7=(0,3) 白 不相邻）。
- 当前 `_generate_batch_order` 每个大区的黑/白格内部**随机 shuffle**，导致同车窗口可能取到空间相邻的格子。
- 实测：当前实现 300 次滚动中，同车相邻平均 2.22 对/次，100% 运行至少出现 1 对。

## 设计：行交替启发式

每个大区的黑格/白格内部排列从"随机 shuffle"改为"**按行交错排列**"：

```python
def _arrange_by_row(lst, numbering):
    """把格子按行分组，然后按行交错排列，使同列上下相邻的格子拉开距离。"""
    by_row = {}
    for n in lst:
        by_row.setdefault(numbering[n][0], []).append(n)
    rows = sorted(by_row)
    if len(rows) <= 1:
        return list(lst)
    res = []
    for col_idx in range(max(len(v) for v in by_row.values())):
        for r in rows:
            if col_idx < len(by_row[r]):
                res.append(by_row[r][col_idx])
    return res
```

效果：同大区某相位的格子按 行0→行1→行2→行0→... 交错，同列上下相邻格子在顺序中拉开。

## 验证结果（300 次随机批次）

| 约束 | 当前实现 | 行交替启发式 |
|------|---------|-------------|
| 同车任意两点不相邻 | 300/300 有相邻（均值2.22） | **300/300 无相邻（均值0）** |
| 大区轮转 [2,1,0]*6 | ✓ | ✓ |
| 前9黑后9白 | ✓ | ✓ |
| 前9黑格互不相邻 | ✓ | ✓ |
| 18 格全覆盖 | ✓ | ✓ |

## 保留的随机性

- 先黑后白 vs 先白后黑：随机
- 起始大区（换轮衔接）：随机
- 行内（同行格子）相对顺序：保留 shuffle

## 改动范围

- 仅 `_generate_batch_order` 内部排列逻辑（新增 `_arrange_by_row` 帮助函数）。
- 跨车滚动、换轮衔接、编号映射均不变。

## 测试新增

- `test_sampling_order_same_car_non_adjacent`：批次顺序按 need 窗口切分，任意窗口内两两不相邻。
- 既有约束测试（大区轮转、黑白结构、全覆盖、跨车连续）保持通过。

## 验证方式

- `python -m pytest tests/test_auto_sampling.py -v` 全部通过。
- 手动：300 次随机批次，同车窗口内无相邻。
- 更新 Excel 展示效果。
