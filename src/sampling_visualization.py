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
