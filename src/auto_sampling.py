import base64
import random
from io import BytesIO
from typing import List, Tuple
from log.log import logger
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


class CoalSamplingOptimizer:
    def __init__(self, rows: int = 3, cols: int = 6,
                 row_span: int = 3, col_span: int = 2, length: int = 0, width: int = 0, ljs=tuple(),
                 yx=tuple()):
        """
        初始化采样优化器
        """
        self.length = length
        self.width = width
        self.ljs = ljs
        self.yx = yx
        self.rows = rows
        self.cols = cols
        self.row_span = row_span
        self.col_span = col_span

        # 生成大区划分
        self.region_mask = self.generate_region_mask(rows, cols, row_span, col_span)
        self.num_regions = int(np.max(self.region_mask)) + 1

        # 构建各种映射关系
        self.region_to_cells = {}
        self.row_counts = {i: 0 for i in range(rows)}
        self.col_counts = {j: 0 for j in range(cols)}
        self.region_counts = {r: 0 for r in range(self.num_regions)}

        for region_id in range(self.num_regions):
            self.region_to_cells[region_id] = []
            for i in range(rows):
                for j in range(cols):
                    if self.region_mask[i, j] == region_id:
                        self.region_to_cells[region_id].append((i, j))

    def get_region_coordinates(self, region_id: int):
        current_row = region_id // 6
        current_col = region_id % 6
        return self.get_region_coordinates1(current_row, current_col)

    def get_region_coordinates1(self, current_row: int, current_col: int):
        region_length = self.length / self.cols
        region_width = self.width / self.rows
        current_region_x0 = current_col * region_length
        current_region_y0 = current_row * region_width
        current_region_x1 = current_col * region_length + region_length
        current_region_y1 = current_row * region_width + region_width
        return [current_region_x0, current_region_y0, current_region_x1, current_region_y1]

    def get_point(self, region_id: int) -> list:
        region = self.get_region_coordinates(region_id)
        while True:
            x = random.uniform(region[0], region[1])
            y = random.uniform(region[2], region[3])
            for lj in self.ljs:
                if lj[0] <= x <= lj[1] and lj[2] <= y <= lj[3]:
                    continue
            if self.yx[0] < x < self.yx[1] and self.yx[2] < y < self.yx[3]:
                return [x, y]

    def get_point1(self, current_row: int, current_col: int) -> list:
        region = self.get_region_coordinates1(current_row, current_col)
        max_try_count = 100
        delta = 100
        while max_try_count:
            x = random.uniform(region[0] + delta, region[2] - delta)
            y = random.uniform(region[1] + delta, region[3] - delta)

            # 检查是否在拉筋区域内
            in_lajin = False
            for lj in self.ljs:
                if lj[0] <= x <= lj[2] and lj[1] <= y <= lj[3]:
                    in_lajin = True
                    break  # 如果在拉筋内，直接跳出循环

            # 如果在拉筋区域内，重新生成点
            if in_lajin:
                max_try_count -= 1
                continue

            # 检查是否在允许区域内
            if self.yx[0] < x < self.yx[2] and self.yx[1] < y < self.yx[3]:
                return [x, y]

            max_try_count -= 1

        raise ValueError("无法在给定区域中找到满足条件的点,请确认拉筋和允许区域设置正确，或请重新执行")

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

    def is_adjacent_any(self, point: Tuple[int, int], points: List[Tuple[int, int]]):
        for existing_point in points:
            if self.is_adjacent(point, existing_point):
                return True
        return False

    def get_valid_choice_from_region(self, region_id: int, used_points: List[Tuple[int, int]],
                                     max_attempts: int = 50, is_adjacent_constraint: bool = True) -> Tuple[int, int]:
        """从指定区域获取一个有效的、不与已有点相邻的采样点"""
        available_cells = [cell for cell in self.region_to_cells[region_id]
                           if cell not in used_points]

        if not available_cells:
            raise ValueError(f"区域 {region_id} 中没有可用的未使用单元格")

        attempts = 0
        while attempts < max_attempts:
            choice = random.choice(available_cells)
            if is_adjacent_constraint:
                if not self.is_adjacent_any(choice, used_points):
                    return choice
            else:
                return choice
            attempts += 1
        raise ValueError(f"区域 {region_id} 无法找到不相邻的采样点，已尝试{max_attempts}次")

    def optimize_sampling(self, num_points: int) -> np.ndarray:
        """
        优化的采样点分配算法
        每3个采样点为一组，均匀分布在3个大区中，确保不相邻
        """
        selected_points = []
        region_order = list(range(self.num_regions))  # [0, 1, 2] for 3 regions
        for i in range(num_points // self.num_regions):
            # 确定当前采样点应该分配到哪个区域
            random.shuffle(region_order)
            for region_id in region_order:
                # 从目标区域获取一个有效的采样点
                valid_point = self.get_valid_choice_from_region(
                    region_id=region_id,
                    used_points=selected_points,
                    is_adjacent_constraint=i * self.num_regions <= 6
                )
                selected_points.append(valid_point)
                logger.info(f"第{i + 1}次采样，分配到区域{region_id}，位置{valid_point}")
        # 注意，选择的采样点为[row,col] 索引，也就是[height,width],也就是[y,x]
        result = np.array(selected_points)
        real_points = []
        for point in result:
            real_points.append(self.get_point1(*point))
        # self.visualize_sampling_points_animated(result, num_points)
        fig = self.visualize_sampling_points(result, real_points, num_points)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        buf.seek(0)
        image_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return real_points, image_base64

    def visualize_sampling_points(self, sampling_points: np.ndarray, real_points: list, num_points: int):
        """可视化采样点"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制网格
        for i in range(self.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(self.cols + 1):
            ax.axvline(j, color='black', linewidth=2)

        # 绘制大区背景色
        colors = plt.get_cmap('Pastel1', self.num_regions)
        for region_id in range(self.num_regions):
            region_color = colors(region_id)
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.region_mask[i, j] == region_id:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                 edgecolor='none', facecolor=region_color, alpha=0.7)
                        ax.add_patch(rect)

        # 绘制拉筋
        width_scale = self.width / self.rows
        length_scale = self.length / self.cols
        for lj in self.ljs:
            ax.add_patch(patches.Rectangle((lj[0] / length_scale, lj[1] / width_scale),
                                           (lj[2] - lj[0]) / length_scale,
                                           (lj[3] - lj[1]) / width_scale,
                                           edgecolor='red', facecolor='red', linewidth=2, alpha=0.3))
        # 绘制允许区域
        ax.add_patch(patches.Rectangle((self.yx[0] / length_scale, self.yx[1] / width_scale),
                                       (self.yx[2] - self.yx[0]) / length_scale,
                                       (self.yx[3] - self.yx[1]) / width_scale,
                                       edgecolor='green', facecolor='None', linewidth=2))

        # 绘制小区的坐标
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j + 0.5, i + 0.5 + 0.2, f"unit:({i},{j})",
                        ha='center', va='center', fontsize=12, color='black')

        # 绘制真实坐标
        for i in range(len(sampling_points)):
            current_row = sampling_points[i][0]
            current_col = sampling_points[i][1]
            real_point = real_points[i]
            ax.text(current_col + 0.5, current_row + 0.5 + 0.3, f"real:({int(real_point[0])},{int(real_point[1])})",
                    ha='center', va='center', fontsize=12, color='black')
        # 标记采样区域序号
        for idx, (i, j) in enumerate(sampling_points):
            ax.plot(j + 0.5, i + 0.5, 'ro', markersize=36,
                    markeredgecolor='darkred', markeredgewidth=1, zorder=5)
            ax.text(j + 0.5, i + 0.5, f'{idx + 1}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=24, zorder=10)

        # 标记采样点
        for idx, (x, y) in enumerate(real_points):
            real_x = x / length_scale
            real_y = y / width_scale
            ax.plot(real_x, real_y, 'go', markersize=12,
                    markeredgecolor='darkred', markeredgewidth=1, zorder=6)

        # 设置图形属性
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        title = f'汽车煤采样点规划 (共{num_points}个采样点)\n'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # 绘制legend
        legend_elements = []
        for region_id in range(self.num_regions):
            color = colors(region_id)
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black',
                              label=f'大区{region_id + 1}')
            )

        legend_elements.append(
            patches.Patch(facecolor="red", alpha=0.3, edgecolor='black',
                          label=f'拉筋区域')
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2, label='区域序号')
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2, label='采样点')
        )

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        # plt.show()
        return fig

    def visualize_sampling_points_animated(self, sampling_points: np.ndarray, num_points: int,
                                           save_path: str = "sampling_animation1.gif"):
        """动画形式可视化采样点，每画一个点是一帧，帧率为1"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制网格
        for i in range(self.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(self.cols + 1):
            ax.axvline(j, color='black', linewidth=2)

        # 绘制大区背景色
        colors = plt.cm.get_cmap('Pastel1', self.num_regions)
        for region_id in range(self.num_regions):
            region_color = colors(region_id)
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.region_mask[i, j] == region_id:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                 edgecolor='none', facecolor=region_color, alpha=0.7)
                        ax.add_patch(rect)

        # 绘制小区的坐标
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j + 0.5, i + 0.5, f"({i},{j})",
                        ha='center', va='center', fontsize=12, color='black')

        # 创建空的列表用于存储图形元素
        all_scatter_points = []  # 存储所有scatter对象
        all_text_labels = []  # 存储所有text对象

        def animate(frame):
            # 每次调用时清除之前的所有点
            for point in all_scatter_points:
                point.remove()
            for text in all_text_labels:
                text.remove()

            all_scatter_points.clear()
            all_text_labels.clear()

            # 绘制到当前帧为止的所有采样点
            for idx in range(frame + 1):
                if idx < len(sampling_points):
                    i, j = sampling_points[idx]
                    # 绘制采样点
                    scatter = ax.plot(j + 0.5, i + 0.5, 'ro', markersize=36,
                                      markeredgecolor='darkred', markeredgewidth=1, zorder=5)[0]
                    # 绘制数字标签
                    text = ax.text(j + 0.5, i + 0.5, f'{idx + 1}',
                                   ha='center', va='center', color='white',
                                   fontweight='bold', fontsize=24, zorder=10)

                    all_scatter_points.append(scatter)
                    all_text_labels.append(text)

            # 更新标题
            title = f'汽车煤采样点规划 (共{num_points}个采样点)\n当前显示: 第{frame + 1}/{num_points}个采样点'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # 创建动画 - 设置帧数为采样点数量，interval为1000ms(1秒)
        anim = animation.FuncAnimation(fig, animate, frames=len(sampling_points),
                                       interval=1000, repeat=False, blit=False)

        # 设置图形属性
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # 绘制legend
        legend_elements = []
        for region_id in range(self.num_regions):
            color = colors(region_id)
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black',
                              label=f'大区{region_id + 1}')
            )
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2, label='采样点')
        )

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        plt.tight_layout()
        plt.gca().invert_yaxis()

        # 保存为GIF
        if save_path:
            print(f"正在保存动画到 {save_path}...")
            anim.save(save_path, writer='pillow', fps=1, dpi=100)
            print(f"动画保存完成！")

        plt.show()
        return anim

    def visualize_sampling_points_step_by_step(self, sampling_points: np.ndarray, num_points: int):
        """逐步显示采样点（非动画，手动控制）"""
        colors = plt.cm.get_cmap('Pastel1', self.num_regions)

        for step in range(len(sampling_points)):
            fig, ax = plt.subplots(figsize=(14, 8))

            # 绘制网格
            for i in range(self.rows + 1):
                ax.axhline(i, color='black', linewidth=2)
            for j in range(self.cols + 1):
                ax.axvline(j, color='black', linewidth=2)

            # 绘制大区背景色
            for region_id in range(self.num_regions):
                region_color = colors(region_id)
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.region_mask[i, j] == region_id:
                            rect = patches.Rectangle((j, i), 1, 1,
                                                     edgecolor='none', facecolor=region_color, alpha=0.7)
                            ax.add_patch(rect)

            # 绘制小区的坐标
            for i in range(self.rows):
                for j in range(self.cols):
                    ax.text(j + 0.5, i + 0.5, f"({i},{j})",
                            ha='center', va='center', fontsize=12, color='black')

            # 标记已选择的采样点
            for idx in range(step + 1):
                i, j = sampling_points[idx]
                ax.plot(j + 0.5, i + 0.5, 'ro', markersize=36,
                        markeredgecolor='darkred', markeredgewidth=1, zorder=5)
                ax.text(j + 0.5, i + 0.5, f'{idx + 1}',
                        ha='center', va='center', color='white', fontweight='bold', fontsize=24, zorder=10)

            # 设置图形属性
            ax.set_xlim(0, self.cols)
            ax.set_ylim(0, self.rows)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            title = f'汽车煤采样点规划 (共{num_points}个采样点)\n当前显示: 第{step + 1}/{num_points}个采样点'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            # 绘制legend
            legend_elements = []
            for region_id in range(self.num_regions):
                color = colors(region_id)
                legend_elements.append(
                    patches.Patch(facecolor=color, edgecolor='black',
                                  label=f'大区{region_id + 1}')
                )
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                           markersize=12, markeredgecolor='darkred', markeredgewidth=2, label='采样点')
            )

            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
            plt.tight_layout()
            plt.gca().invert_yaxis()

            plt.show()
            plt.close()  # 关闭当前图形

            # 等待用户输入继续
            if step < len(sampling_points) - 1:
                input(f"按回车键继续显示第{step + 2}个采样点...")


def get_automatic_sampling_points(car_length: int, car_width: int, car_lj: tuple = tuple(), car_kx: tuple = tuple):
    logger.info(f"汽车长度：{car_length}")
    logger.info(f"汽车宽度：{car_width}")
    logger.info(f"拉筋区域：{car_lj}")
    logger.info(f"允许区域：{car_kx}")
    opt = CoalSamplingOptimizer(rows=3, cols=6, row_span=3, col_span=2, length=car_length, width=car_width,
                                ljs=car_lj, yx=car_kx)
    max_try_times = 100
    while True:
        try:
            return opt.optimize_sampling(18)
        except ValueError as e:
            logger.warning(e)
            max_try_times -= 1
            if max_try_times == 0:
                error_msg = "尝试次数过多，请检查输入参数"
                logger.error(error_msg)
                raise ValueError(error_msg)
