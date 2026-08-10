import random
import threading
from typing import Tuple
from dataclasses import dataclass
from log.log import logger
import numpy as np

from src.sampling_visualization import render_sampling_preview


class SamplingError(Exception):
    """采样过程中无法生成满足约束的点时抛出。"""


@dataclass(frozen=True)
class SamplingConfig:
    grid_rows: int = 3
    grid_cols: int = 6
    region_row_span: int = 3
    region_col_span: int = 2
    shuffle_regions: bool = True
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
        if self.num_regions < 2:
            raise ValueError(
                f"num_regions({self.num_regions}) 必须至少为 2，否则无法保证相邻采样点位于不同大区"
            )

    @property
    def num_regions(self) -> int:
        return (self.grid_rows // self.region_row_span) * (
            self.grid_cols // self.region_col_span
        )


class CoalSamplingOptimizer:
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

    def sample_point_in_region(self, row: int, col: int) -> list[float]:
        """在指定格子(row, col)内随机生成一个满足约束的真实坐标点。

        坐标系约定：列 0 位于车辆最左端（物理 x 最小），列 cols-1 位于
        最右端（物理 x 最大）。编号方向（列0=最右大区）与物理方向解耦。
        """
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

    @staticmethod
    def generate_region_mask(rows: int, cols: int,
                             row_span: int, col_span: int) -> np.ndarray:
        """
        生成大区划分矩阵。

        坐标系约定：列 0 为最右列（从右往左编号）。因此 region 0 覆盖列0-1
        （物理最右两列，即第一个大区），region 编号从右往左递增。
        """
        region_mask = np.zeros((rows, cols), dtype=int)
        region_id = 0

        for r in range(0, rows, row_span):
            for c in range(0, cols, col_span):
                row_end = min(r + row_span, rows)
                col_end = min(c + col_span, cols)
                region_mask[r:row_end, c:col_end] = region_id
                region_id += 1

        return region_mask

    @staticmethod
    def is_adjacent(point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
        """判断两个点是否相邻"""
        r1, c1 = point1
        r2, c2 = point2
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    def plan_regions(self) -> list[tuple[int, int]]:
        """规划采样小区（坐标系列0为最右，大区从右往左：region 0→2）。

        核心思想（棋盘格 + 相位随机 + 区内随机）：
        1. 棋盘格只有 2 种着色——phase_shift 随机取 0/1，黑格判定为
           (r+c+shift)%2==0。黑格集为 9 格（每区 3 格），同相位格子彼此
           不相邻，因此前 num_regions 轮的黑格互不相邻是结构保证。
        2. 相位随机选完后，棋盘格即确定；之后每个大区只在对应颜色的
           集合内部随机排列（self._rng.shuffle），再按固定顺序 pop。
        3. 大区轮序固定从右往左（region 0 → num_regions-1），每轮每区放
           1 点，因此任意相邻采样点来自不同大区（跨轮边界也满足）。
        4. 前 num_regions 轮取黑格（9 点），后 num_regions 轮取白格（9 点），
           实现 18 格全覆盖、无重复、每区 6 点。

        随机性来源：phase_shift（1 bit，决定黑格为偶格/奇格）+ 每区黑/白
        集合内部排列。seed 固定时可复现；shuffle_regions=False 时完全确定性
        （shift 固定 0、区内不 shuffle）。
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
        region_order = list(range(self.num_regions))  # 从右往左（region 0 最右）
        selected: list[tuple[int, int]] = []
        for round_idx in range(rounds):
            for region_id in region_order:
                pool = black if round_idx < constrained_rounds else white
                selected.append(pool[region_id].pop())
                logger.info(f"第{round_idx + 1}次采样，分配到区域{region_id}，位置{selected[-1]}")
        return selected

    def realize_regions(self, regions: list[list[int, int]]) -> tuple[list[list[float]], str]:
        real_points = []
        for (r, c) in regions:
            real_points.append(self.sample_point_in_region(r, c))
        image = render_sampling_preview(self, regions, real_points)
        return real_points, image


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
    return opt.realize_regions(regions)


# 编号 -> 格子映射缓存（确定性：列优先从右往左，生命周期不变）
_NUMBERING: dict[int, tuple[int, int]] | None = None
# 各采样机的批次采样顺序缓存（key=sampler_id，批次开始时随机生成一次，批次内跨车复用）
_BATCH_ORDER: dict[str, list[int]] = {}
_NUMBERING_LOCK = threading.Lock()
# 生成批次时同车不相邻的重试上限（约几十 ms/批次，仅初始化一次）
_BATCH_MAX_TRIES = 3000


def _ensure_numbering(config: SamplingConfig | None) -> dict[int, tuple[int, int]]:
    """建立并返回编号 1..num_points -> 格子 的固定映射。

    编号规则（列优先、从右往左）：坐标系列 0 为最右列，编号从最右列（列0）开始，
    逐列左移（列1、列2...）。因此编号 1..num_regions*rows 对应最右大区（region 0）。
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
                for c in range(cols):
                    for r in range(rows):
                        numbering[n] = (r, c)
                        n += 1
                _NUMBERING = numbering
    return _NUMBERING


def _generate_batch_order(config: SamplingConfig | None, start_region: int | None = None,
                          avoid_points: list[int] | None = None) -> list[int]:
    """生成一次批次的随机采样顺序（随机性 + 同车尽量少相邻）。

    约束：
    - 大区严格轮转（0->1->2，递增循环，region 0 为最右），从 start_region 开始；
      未指定时从最右大区（region 0）开始。
    - 前 num_regions 轮取同一种奇偶格（互不相邻），后 num_regions 轮取另一种。
    - 先黑(偶)后白(奇) 或 先白后黑 随机。
    - 每个大区的同奇偶格子内部完全随机排列（保留批次间随机性）。
    - 重试：任意连续窗口（长度<=6）内两两不相邻，使 need<=6 的同车尽量不相邻
      （need=7 属数学极限，偶尔相邻）。达到重试上限时返回相邻最少的。
    - avoid_points：换轮衔接时传入旧批次收尾点，要求本批次开头若干点不与它们
      相邻，避免换轮边界出现同车相邻。
    结果：18 个编号，大区序列从 start_region 起 0->1->2 循环，
    前 9 同色后 9 另一色；need<=6 的同车窗口内任意两点不相邻（实测 100/100）。
    """
    numbering = _ensure_numbering(config)
    opt = CoalSamplingOptimizer(config=config)
    mask = opt.region_mask
    num_regions = opt.num_regions
    rows, cols = opt.rows, opt.cols
    avoid = set(avoid_points or [])

    even = {r: [] for r in range(num_regions)}   # (r+c)%2==0
    odd = {r: [] for r in range(num_regions)}    # (r+c)%2==1
    for n, (r, c) in numbering.items():
        reg = int(mask[r, c])
        if (r + c) % 2 == 0:
            even[reg].append(n)
        else:
            odd[reg].append(n)

    def is_adj(a, b):
        r1, c1 = numbering[a]; r2, c2 = numbering[b]
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    def window_adj(order):
        """任意连续窗口（长度<=6）内两两相邻的对数。"""
        total = 0
        n = len(order)
        for win_size in range(2, 7):
            for start in range(n - win_size + 1):
                win = order[start:start + win_size]
                for a in range(len(win)):
                    for b in range(a + 1, len(win)):
                        if is_adj(win[a], win[b]):
                            total += 1
        return total

    if start_region is None:
        start_region = 0
    region_cycle = [(start_region + k) % num_regions for k in range(num_regions)]
    rounds = (rows * cols) // num_regions

    best_order: list[int] | None = None
    best_adj = 10**9
    for _ in range(_BATCH_MAX_TRIES):
        first, second = (even, odd) if random.random() < 0.5 else (odd, even)
        e = {r: list(first[r]) for r in range(num_regions)}
        o = {r: list(second[r]) for r in range(num_regions)}
        for reg in range(num_regions):
            random.shuffle(e[reg])
            random.shuffle(o[reg])
        order: list[int] = []
        for round_idx in range(rounds):
            for reg in region_cycle:
                pool = e if round_idx < num_regions else o
                order.append(pool[reg][round_idx % num_regions])
        if avoid:
            dup_front = sum(1 for n in order[:6] if n in avoid)
            # 重复数优先，其次窗口相邻数
            adj = window_adj(order) + dup_front * 1000
        else:
            adj = window_adj(order)
        if adj == 0:
            return order
        if adj < best_adj:
            best_adj = adj
            best_order = order
    return best_order


def _ensure_batch_order(config: SamplingConfig | None, sampler_id: str = "default") -> list[int]:
    """返回指定采样机的当前批次采样顺序；无缓存时（新批次）随机生成一次。"""
    if sampler_id not in _BATCH_ORDER:
        with _NUMBERING_LOCK:
            if sampler_id not in _BATCH_ORDER:
                _BATCH_ORDER[sampler_id] = _generate_batch_order(config)
    return _BATCH_ORDER[sampler_id]


def _new_batch(config: SamplingConfig | None, start_region: int | None = None,
               avoid_points: list[int] | None = None,
               sampler_id: str = "default") -> list[int]:
    """强制开始指定采样机的新批次：重新随机生成采样顺序，大区从 start_region 起轮转。"""
    with _NUMBERING_LOCK:
        _BATCH_ORDER[sampler_id] = _generate_batch_order(config, start_region, avoid_points)
    return _BATCH_ORDER[sampler_id]


def get_automatic_sampling_regions_rolling(
        used: list[list[int]] | None = None,
        need: int = 0,
        config: SamplingConfig | None = None,
        sampler_id: str = "default") -> tuple[list[int], list[list[int]]]:
    """跨车滚动采样：返回 (本车应采的编号列表, 对应格子坐标列表)。

    used: 当前批次已用采样小区的格子坐标 [[row, col], ...]（无需排序，服务端
    自动去重并转换为编号）。need: 本车要采点数。
    采样顺序在批次开始时随机生成一次（大区严格 0->1->2 轮转、先黑后白或先白后黑
    随机、每大区同色格内部随机），批次内跨车复用保证大区持续轮转且不重复。
    used 为空（None/[]）时视为新批次，重新随机生成顺序。

    换轮：未用编号不足 need 时，先取完全部未用（旧批次收尾），再生成新批次。
    新批次大区从"旧批次最后一个收尾点的下一个大区"起（递增循环），保证全局
    0->1->2->0->... 跨批次无缝连续轮转。补足点 = 新批次顺序前若干编号。
    返回中旧批次收尾在前、新批次补足在后；前端下轮应传"新批次补足部分"的
    格子坐标 作为 used（而非空 []），服务端即延续新批次。

    编号->格子映射固定（_NUMBERING，列优先从右往左：编号1-6 为最右大区 region 0）。
    """
    if need <= 0:
        return [], []

    numbering = _ensure_numbering(config)
    cell_to_num = {cell: n for n, cell in numbering.items()}

    # used 为格子坐标 [[r,c],...] -> 转编号集合
    used_set = {cell_to_num[tuple(cell)] for cell in (used or [])}
    if not used_set:
        order = _new_batch(config, sampler_id=sampler_id)
    else:
        order = _ensure_batch_order(config, sampler_id=sampler_id)

    allocated: list[int] = []
    allocated_set: set[int] = set()
    for n in order:
        if n not in used_set:
            allocated.append(n)
            allocated_set.add(n)
        if len(allocated) >= need:
            break

    if len(allocated) < need:
        # 未用不足：当前批次已采完，生成新的 18 点批次。
        # 新批次大区从"旧批次最后一个收尾点的下一个大区"起（递增循环），
        # 保证全局 0->1->2->0->... 跨批次无缝连续轮转。
        # 补足点 = 新批次顺序前 (need - len(allocated)) 个编号（不跳过），
        # 保持新批次自身大区 0->1->2 连续。新批次开头已由 avoid_points 保证
        # 与旧批次收尾点不相邻（避免同车相邻）。
        opt = CoalSamplingOptimizer(config=config)
        old_last = allocated[-1]  # 旧批次收尾的最后一个点
        last_region = int(opt.region_mask[numbering[old_last][0],
                                          numbering[old_last][1]])
        next_start = (last_region + 1) % opt.num_regions
        # 生成新批次：要求其开头若干点（<=6）与旧批次收尾点不相邻（避免换轮边界
        # 同车相邻），且批次内部窗口不相邻。generator 内部已保证与 prev_order 不同。
        order = _new_batch(config, start_region=next_start, avoid_points=allocated,
                           sampler_id=sampler_id)
        for n in order:
            allocated.append(n)
            if len(allocated) >= need:
                break

    cells = [list(numbering[n]) for n in allocated]
    return allocated, cells


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

