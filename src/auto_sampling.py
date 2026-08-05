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

    @staticmethod
    def generate_region_mask(rows: int, cols: int,
                             row_span: int, col_span: int) -> np.ndarray:
        """
        生成大区划分矩阵
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
        """规划采样小区（从右往左）。

        核心思想（棋盘格 + 相位随机 + 区内随机）：
        1. 棋盘格只有 2 种着色——phase_shift 随机取 0/1，黑格判定为
           (r+c+shift)%2==0。黑格集为 9 格（每区 3 格），同相位格子彼此
           不相邻，因此前 num_regions 轮的黑格互不相邻是结构保证。
        2. 相位随机选完后，棋盘格即确定；之后每个大区只在对应颜色的
           集合内部随机排列（self._rng.shuffle），再按固定顺序 pop。
        3. 大区轮序固定从右往左（region num_regions-1 → 0），每轮每区放
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
        region_order = list(range(self.num_regions - 1, -1, -1))  # 从右往左
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
# 当前批次的采样顺序缓存（批次开始时随机生成一次，批次内跨车复用）
_BATCH_ORDER: list[int] | None = None
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


def _generate_batch_order(config: SamplingConfig | None) -> list[int]:
    """生成一次批次的随机采样顺序。

    约束：
    - 大区严格 2->1->0 轮转（每轮大区 num_regions-1 -> 0）。
    - 前 num_regions 轮取同一种奇偶格（互不相邻），后 num_regions 轮取另一种。
    - 先黑(偶)后白(奇) 或 先白后黑 随机。
    - 每个大区的同奇偶格子内部随机排列。
    结果：18 个编号，大区序列严格 [2,1,0]*6，前 9 同色后 9 另一色。
    """
    numbering = _ensure_numbering(config)
    opt = CoalSamplingOptimizer(config=config)
    mask = opt.region_mask
    num_regions = opt.num_regions
    rows, cols = opt.rows, opt.cols

    even = {r: [] for r in range(num_regions)}   # (r+c)%2==0
    odd = {r: [] for r in range(num_regions)}    # (r+c)%2==1
    for n, (r, c) in numbering.items():
        reg = int(mask[r, c])
        if (r + c) % 2 == 0:
            even[reg].append(n)
        else:
            odd[reg].append(n)
    for reg in range(num_regions):
        random.shuffle(even[reg])
        random.shuffle(odd[reg])

    # 随机决定先取偶格还是奇格
    first, second = (even, odd) if random.random() < 0.5 else (odd, even)

    order: list[int] = []
    rounds = (rows * cols) // num_regions
    region_order = list(range(num_regions - 1, -1, -1))
    for round_idx in range(rounds):
        for reg in region_order:
            pool = first if round_idx < num_regions else second
            order.append(pool[reg][round_idx % num_regions])
    return order


def _ensure_batch_order(config: SamplingConfig | None) -> list[int]:
    """返回当前批次采样顺序；无缓存时（新批次）随机生成一次。"""
    global _BATCH_ORDER
    if _BATCH_ORDER is None:
        with _NUMBERING_LOCK:
            if _BATCH_ORDER is None:
                _BATCH_ORDER = _generate_batch_order(config)
    return _BATCH_ORDER


def _new_batch(config: SamplingConfig | None) -> list[int]:
    """强制开始新批次：重新随机生成采样顺序。"""
    global _BATCH_ORDER
    with _NUMBERING_LOCK:
        _BATCH_ORDER = _generate_batch_order(config)
    return _BATCH_ORDER


def get_automatic_sampling_regions_rolling(
        used: list[int] | None = None,
        need: int = 0,
        config: SamplingConfig | None = None) -> tuple[list[int], list[list[int]]]:
    """跨车滚动采样：返回 (本车应采的编号列表, 对应格子列表)。

    used: 当前批次已用编号（无重复，1-18 范围）。need: 本车要采点数。
    采样顺序在批次开始时随机生成一次（大区 2->1->0 轮转、先黑后白或先白后黑随机、
    每大区同色格内部随机），批次内跨车复用保证大区持续轮转且不重复。
    used 为空（None/[]）时视为新批次，重新随机生成顺序。
    未用编号不足 need 时，先取完全部未用，再换轮从顺序开头补足。
    编号->格子映射固定（_NUMBERING，列优先从右往左）。
    """
    if need <= 0:
        return [], []

    numbering = _ensure_numbering(config)

    used_set = set(used or [])
    if not used_set:
        order = _new_batch(config)
    else:
        order = _ensure_batch_order(config)

    allocated: list[int] = []
    allocated_set: set[int] = set()
    for n in order:
        if n not in used_set:
            allocated.append(n)
            allocated_set.add(n)
        if len(allocated) >= need:
            break

    if len(allocated) < need:
        # 未用不足：先取全部未用，再换轮从顺序开头补足（跳过已分配）
        for n in order:
            if n not in allocated_set:
                allocated.append(n)
                allocated_set.add(n)
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

