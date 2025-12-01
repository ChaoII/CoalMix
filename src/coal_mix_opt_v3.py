"""
author:AiChao
date:2025-06-29
针对鄂州项目做了一些调整优化，比如鄂州电厂没有煤价，没有负荷调度计划，也就没有计划煤量
"""
import json
from functools import reduce
from math import lcm
from log.log import logger
import cvxpy as cp
import numpy as np

epsilon = 0.009
I_COL = 0  # 煤场索引
W_COL = 1  # 库存量
Q_COL = 2  # 热值     # Qnet,ar
S_COL = 3  # 硫分     # St,ar
A_COL = 4  # 灰分     # Aar
V_COL = 5  # 挥发分   # Vdaf
M_COL = 6  # 全水     # Mt,ar
T_COL = 7  # 灰熔点   # Taf


def normalize_to_list(v):
    """把 None/空/ndarray 转成 list"""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return [v]


def safe_sum_with_index(x, indices, axis=0):
    """安全求和：当索引全为False时返回0"""
    if np.any(indices):
        return cp.sum(x[indices, :], axis=axis)
    else:
        return 0


def coal_mixed_integer_optimization_v3(coal_info, unit_constraint, container_constraint, mix_ratio,
                                       coal_quality, mix_coal_num, max_scheme_count, opt_flag):
    # ------------------------------数据初始化-------------------------------------------------
    # 求每一行混煤率的最大公约数
    gcd_s = 1
    for i in range(mix_ratio.shape[1]):
        gcd_s = np.gcd(gcd_s, mix_ratio[:, i])
    mix_ratio_normalized = mix_ratio / gcd_s.reshape((-1, 1))
    # 每一行混煤率的和
    sum_mix_ratio = mix_ratio_normalized.sum(axis=1).reshape((-1, 1))
    # 混煤率和的最小公倍数(单仓煤仓煤量)
    max_ele = reduce(lambda c_, d_: lcm(int(c_), int(d_)), sum_mix_ratio.flatten(order="C").tolist())
    # 混煤比例元素集合
    ele_s = np.unique(mix_ratio_normalized / np.tile(sum_mix_ratio, (1, mix_ratio_normalized.shape[1])) * max_ele)
    # 煤仓数
    m = container_constraint.shape[0]
    # 煤种数
    n = coal_info.shape[0]
    # 大M法（Big M Method）中使用的一个足够大的常数，用于将逻辑约束转化为线性约束
    M = max_ele * 2
    # 低负荷下煤仓启动索引
    container_low_index = container_constraint[:, 0] != 0
    # 高负荷下煤仓启用索引
    container_high_index = container_constraint[:, 1] != 0
    # 处理特殊情况：当所有索引都为False时的默认值
    # 低负荷下总煤量单位
    if not np.any(container_low_index):
        logger.warning("所有低负荷煤仓索引都为False，使用默认值")
        # 低负荷下总煤量单位（使用默认值1避免除0错误）
        total_quality_low = 1
    else:
        total_quality_low = np.sum(container_constraint[:, 0]) * max_ele
    # 高负荷下总煤量单位
    if not np.any(container_high_index):
        logger.warning("所有高负荷煤仓索引都为False，使用默认值")
        # 高负荷下总煤量单位（使用默认值1避免除0错误）
        total_quality_high = 1
    else:
        total_quality_high = np.sum(container_constraint[:, 1]) * max_ele

    # -----------------------------开始建模-------------------------------------------------
    # 待约束变量(整数)
    x = cp.Variable((m, n), integer=True)
    # 二元{0，1}辅助变量(x > 0  时z=1，x==0 时 z=0)
    z0 = cp.Variable((m, n), boolean=True)
    z1 = cp.Variable((m * n, ele_s.shape[0]), boolean=True)
    # 辅助变量use_coal，是否使用某种煤
    z2 = cp.Variable(n, boolean=True)
    z3 = cp.Variable(ele_s.shape[0], boolean=True)
    # 连续变量，X-A >= y && A-X >= y && y > 0    abs(x-A) > 0 的线性变换写法
    y = cp.Variable((m, n))
    # 约束0：正整数约束，煤仓存煤量非负
    constraint0 = [x >= 0]
    # 约束1：给煤机出力一致性约束
    constraint1 = []
    if np.any(container_low_index):
        constraint1.append(cp.sum(x[container_low_index, :], axis=1) == max_ele)
    if np.any(container_high_index):
        constraint1.append(cp.sum(x[container_high_index, :], axis=1) == max_ele)
    # 约束2：单仓上煤总数约束(构造二元辅助变量，计算二元辅助变量的值间接计算非零整数),如果指定煤种比例，则该仓煤种比例需要小于等于指定比例数量
    constraint2 = [
        x >= 0,  # 假设 x 非负整数(前面已经有相关约束)
        x <= M * z0,  # 若 z=0 -> x<=0 (结合非负即 x=0)
        x >= 1 * z0  # 若 z=1 -> x>=1
    ]
    # 每行非零整数个数 <= 2
    for i in range(m):
        constraint2.append(cp.sum(z0[i, :]) <= np.max(np.sum(mix_ratio_normalized > 0, axis=1)))

    # 约束3：煤仓上煤比例约束在固定集合{ele_s} 中
    constraint3 = [cp.sum(z1, 1) == 1, z1 @ ele_s == x.flatten(order="C")]
    # 约束4：机组煤质约束
    # 低负荷约束
    constraint4 = []
    if np.any(container_low_index):
        constraint4_low = [(unit_constraint[0][:, 0] <= cp.sum(x[container_low_index, :], axis=0) @
                            coal_info[:, 2:-1] / total_quality_low),
                           (cp.sum(x[container_low_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_low <=
                            unit_constraint[0][:, 1])]
        constraint4.extend(constraint4_low)

        # 高负荷约束（只有存在高负荷煤仓时才添加）
    if np.any(container_high_index):
        constraint4_high = [(unit_constraint[1][:, 0] <= cp.sum(x[container_high_index, :], axis=0) @
                             coal_info[:, 2:-1] / total_quality_high),
                            (cp.sum(x[container_high_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_high <=
                             unit_constraint[1][:, 1])]
        constraint4.extend(constraint4_high)


    # 约束5煤量约束(煤量都是0)
    constraint5 = []
    if coal_quality[0] < epsilon:
        coal_quality[0] = 1
        logger.warning(f"煤量[coal_quality_low: {coal_quality[0]}]小于epsilon,可能为0，已设置为1")
    if coal_quality[1] < epsilon:
        coal_quality[1] = 1
        logger.warning(f"煤量[coal_quality_high: {coal_quality[1]}]小于epsilon，可能为0，已设置为1")
    if np.any(container_low_index) and total_quality_low > 0:
        constraint5_low = [cp.sum(x, axis=0) / total_quality_low <= (coal_info[:, W_COL] / coal_quality[0])]
        constraint5.extend(constraint5_low)
    if np.any(container_high_index) and total_quality_high > 0:
        constraint5_height = [cp.sum(x, axis=0) / total_quality_high <= (coal_info[:, W_COL] / coal_quality[1])]
        constraint5.extend(constraint5_height)

    # 约束6：煤仓煤质约束
    lb_index = np.arange(7, 19, 2)
    ub_index = lb_index + 1
    temp = x @ coal_info[:, 2: - 1] / max_ele
    constraint6 = [(container_constraint[:, lb_index] <= temp), (temp <= container_constraint[:, ub_index])]

    # 约束7：指定挥发分约束(暂无)
    constraint7 = [x >= -1]  # Implement as needed

    # 约束8：煤仓启用约束(基本没用)
    r = np.any([container_constraint[:, 0].reshape(-1, 1), container_constraint[:, 1].reshape(-1, 1)], axis=0)
    r1, _ = np.where(r == 0)
    # r2, _ = np.where(container_constraint[:, 1].reshape(-1, 1) == 0)
    constraint8 = [cp.sum(x[r1, :]) == 0]
    # 约束9：煤仓固定比例约束
    constraint9 = []  # Implement as needed
    container_coal = container_constraint[:, 5: 7]
    for i in range(container_coal.shape[0]):
        coal_kind = normalize_to_list(container_coal[i, 0])
        coal_rate = normalize_to_list(container_coal[i, 1])
        # 规则 1：煤种和比例都未指定 -> 跳过
        if len(coal_kind) == 0 and len(coal_rate) == 0:
            logger.warning(f"煤仓[{i}]未指定煤种和比例，跳过")
            continue
        # 规则 2：煤种和比例都指定 -> 长度必须相等
        total_rate = sum(coal_rate)
        scale_factor = 1 if total_rate == 0 else max_ele / total_rate
        for kind, rate in zip(coal_kind, coal_rate):
            if len(coal_kind) != len(coal_rate):
                raise ValueError(
                    f"第 {i} 行煤种和比例长度不一致: {coal_kind} vs {coal_rate}"
                )
            constraint9.append(x[i, kind] == rate * scale_factor)
        # 规则 3：煤种指定，比例未指定
        if coal_kind and not coal_rate:
            constraint9.append(cp.sum(z3) == 1)  # 只能选择一个方案
            for kind in coal_kind:
                constraint9.append(z3 @ ele_s == x[i, kind])
                constraint9.append(x[i, kind] >= 1)
            continue

        # 规则 4：煤种未指定，比例指定
        if coal_rate and not coal_kind:
            scaled_rates = [rate * scale_factor for rate in coal_rate]  # e.g., [1, 2]
            K = len(scaled_rates)
            zs = cp.Variable((n, K), boolean=True)
            # 每个比例值分配给恰好一个煤种
            for k in range(K):
                constraint9.append(cp.sum(zs[:, k]) == 1)
            # 约束1: 每个位置最多选择一个非零值
            for j in range(n):
                constraint9.append(cp.sum(zs[j, :]) <= 1)
            # 约束2: 定义x[i, j]的值
            for j in range(n):
                # x[i, j] = sum(允许的非零值 * 对应的二进制变量)
                constraint9.append(x[i, j] == scaled_rates @ zs[j, :])

            # 总共使用K个煤种
            constraint9.append(cp.sum(zs) == K)

    # 约束10：最大煤种约束
    # 如果z2[j] = 0 -> 列和必须 ≤ 0，而列和 ≥ 0（自然数保证），所以列和只能等于0。
    # 如果z2[j] = 1 -> 列和 ≤ m*M，允许正数。
    constraint10 = [cp.sum(x, axis=0) <= m * M * z2, cp.sum(z2) <= mix_coal_num]

    # 约束11：不同方案的最大数量限制 -------------------
    # 限制不同配煤方案的种类数量（例如最多 3 种）
    K = max_scheme_count
    # 定义布尔变量
    use_scheme = cp.Variable(K, boolean=True)  # 哪些方案槽位被启用
    assign = cp.Variable((m, K), boolean=True)  # 每个煤仓分配到哪个方案槽
    scheme_pattern = cp.Variable((K, n), integer=True)  # 每种方案的具体配煤方案（n维整数）

    constraint11 = []
    # 每个煤仓必须分配到一个方案槽
    for i in range(m):
        constraint11.append(cp.sum(assign[i, :]) == 1)
    # 每个煤仓只能分配给已启用的方案槽
    for i in range(m):
        for k in range(K):
            constraint11.append(assign[i, k] <= use_scheme[k])
    # 限制最多启用 K 种方案
    constraint11.append(cp.sum(use_scheme) <= K)
    # Big-M 逻辑：当 assign[i, k] == 1 时，强制 x[i, :] == scheme_pattern[k, :]
    for i in range(m):
        for k in range(K):
            for j in range(n):
                constraint11.append(x[i, j] - scheme_pattern[k, j] <= M * (1 - assign[i, k]))
                constraint11.append(scheme_pattern[k, j] - x[i, j] <= M * (1 - assign[i, k]))

    # 构建约束列表
    constraints = []
    constraints.extend(constraint0)
    constraints.extend(constraint1)
    constraints.extend(constraint2)
    constraints.extend(constraint3)
    constraints.extend(constraint4)
    constraints.extend(constraint5)
    constraints.extend(constraint6)
    constraints.extend(constraint7)
    constraints.extend(constraint8)
    constraints.extend(constraint9)
    constraints.extend(constraint10)
    constraints.extend(constraint11)

    # 目标函数
    obj = None
    # 目标：无价格时的“综合寻优指标”设计
    # 我们希望：
    #  1.尽量让机组的配煤热值接近目标（≠过高或过低）；
    #  2.避免高硫煤、保证环保；
    #  3.保留高热值煤种不轻易使用（节约资源）；
    #  4.整体热值利用率合理、差异小。
    if opt_flag == 0:
        # 限制逻辑：未使用的煤比例为 0
        for j in range(n):
            for i in range(m):
                constraints.append(x[i, j] <= M * z2[j])

        # 1. 各煤仓平均热值（与目标热值的偏差越小越好）
        target_q = np.mean([
            unit_constraint[0, 0, 1],
            unit_constraint[1, 0, 1],
            np.mean(container_constraint[container_high_index, 7])
        ])

        # 混煤热值
        # x 是 m×n 矩阵，coal_info[:, Q_COL] 是 n×1 向量
        mix_q = x @ coal_info[:, Q_COL]  # 结果应该是 m×1 向量
        # 热值偏差 - 使用绝对值而不是平方，避免二次规划
        q_dev = cp.sum(cp.abs(mix_q - target_q))
        # 2. 各煤仓平均硫分（越低越好）
        mix_sulfur = x @ coal_info[:, S_COL]  # m×1 向量
        avg_sulfur = cp.sum(mix_sulfur) / m
        # 3. 保留高热值煤（希望高热值煤尽量不被使用）
        # coal_info[:, 1] 是 n×1 向量，(1 - z2) 是 n×1 向量
        remain_w = cp.multiply(coal_info[:, 1], (1 - z2))  # 逐元素乘法
        remain_q = remain_w @ coal_info[:, Q_COL]  # 标量
        # ---------------------- 综合目标函数 ----------------------
        # alpha/beta/gamma 为权重，可根据业务重要性调整
        alpha = 0.5  # 热值偏差权重
        beta = 0.3  # 环保（硫分）权重
        gamma = 0.2  # 保留高热值煤权重
        # 最终目标：最小化热值偏差 + 硫分 - 剩余热值
        obj = alpha * q_dev + beta * avg_sulfur - gamma * remain_q
    # 煤价最低
    elif opt_flag == 1:
        obj = cp.sum(x, axis=0) @ coal_info[:, -1]
    # 最环保（硫分最合理）
    elif opt_flag == 2:
        lower_s = np.min(
            [unit_constraint[0, 1, 1], unit_constraint[1, 1, 1], np.min(container_constraint[container_high_index, 9])])
        obj = lower_s * total_quality_high - cp.sum(x, axis=0) @ coal_info[:, S_COL]
    # 给煤机最小出力（热值最合理）
    elif opt_flag == 3:
        lower_q = np.min(
            [unit_constraint[0, 0, 1], unit_constraint[1, 0, 1], np.min(container_constraint[container_high_index, 7])])
        obj = lower_q * total_quality_high - cp.sum(x, axis=0) @ coal_info[:, Q_COL]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.SCIPY)
    # 求解目标
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        logger.info(f"找到最优解或近似最优解（可能由于数值精度），求解器状态{problem.status}")
        solution = x.value
        mix_info_low = np.sum(solution[container_low_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_low
        mix_info_high = np.sum(solution[container_high_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_high
        mix_case_low = coal_quality[0] / total_quality_low * solution[container_low_index, :]
        mix_case_high = coal_quality[1] / total_quality_high * solution[container_high_index, :]
        result = np.around(solution).astype(int)
        mix_price_low = np.sum(result[container_low_index, :], axis=0) @ coal_info[:, -1] / total_quality_low
        mix_price_high = np.sum(result[container_high_index, :], axis=0) @ coal_info[:, -1] / total_quality_high
        with np.printoptions(precision=2, formatter={'float_kind': '{:0.2f}'.format}):
            print("*************optimize result is:*************")
            print(f"- mix_integer:\n{result}")
            print(f"-mix_case_low:\n{mix_case_low}\n-mix_case_high:\n{mix_case_high}\n")
            print(f"-mix_info_low:\n{mix_info_low}\n-mix_info_high:\n{mix_info_high}\n")
            print(f"-mix_price_low:\n{mix_price_low}\n-mix_price_high:\n{mix_price_high}")
        # 可以继续使用这个解
    elif problem.status == cp.INFEASIBLE:
        raise Exception("问题不可行，请检查约束条件,尝试放松某些约束")
        # 可以尝试放松某些约束
    elif problem.status == cp.UNBOUNDED:
        raise Exception("问题无界，请检查目标函数")
    else:
        raise Exception(f"求解状态: {problem.status}")
        # 等效于abs(x-x.value)>=0但是abs()>=0是一个非凸的问题,需要构造连续辅助变量进行调整
    return result.tolist(), [mix_case_low.tolist(), mix_case_high.tolist()], \
        [mix_info_low.tolist(), mix_info_high.tolist()], [
        mix_price_low.tolist(), mix_price_high.tolist()]


if __name__ == '__main__':
    mat_data = json.load(open("../test_data/mix_coal/input_v3_3.json"))
    # 使用示例
    coal_info = mat_data["coal_info"]  # 替换为实际的数据
    unit_constraint = mat_data["unit_constraint"]
    container_constraint = mat_data["container_constraint"]
    mix_ratio = mat_data["mix_ratio"]
    coal_quality = mat_data["coal_quality"]  # 替换为实际的数据
    mix_coal_num = mat_data["mix_coal_num"]
    max_scheme_count = mat_data["max_scheme_count"]
    opt_flag = mat_data["opt_flag"]
    c = coal_mixed_integer_optimization_v3(np.array(coal_info), np.array(unit_constraint),
                                           np.array(container_constraint, dtype=object),
                                           np.array(mix_ratio, dtype=int),
                                           coal_quality, mix_coal_num, max_scheme_count, opt_flag)
