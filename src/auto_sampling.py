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


# 编号 -> 格子映射缓存（服务启动时首次 plan_regions() 建立，生命周期不变）
_NUMBERING: dict[int, tuple[int, int]] | None = None
_NUMBERING_LOCK = threading.Lock()


def _ensure_numbering(config: SamplingConfig | None) -> dict[int, tuple[int, int]]:
    """建立并返回编号 1..num_points -> 格子 的固定映射。"""
    global _NUMBERING
    if _NUMBERING is None:
        with _NUMBERING_LOCK:
            if _NUMBERING is None:
                opt = CoalSamplingOptimizer(config=config)
                cells = opt.plan_regions()
                _NUMBERING = {i + 1: cell for i, cell in enumerate(cells)}
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

