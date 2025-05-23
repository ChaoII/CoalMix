"""
author:AiChao
date:2023-11-24
"""

import numpy as np
import cvxpy as cp
from math import lcm
import scipy.io as scio
from functools import reduce

epsilon = 0.009


def coal_mixed_integer_optimization(coal_info, unit_constraint, container_constraint, feeder_capacity, mix_ratio,
                                    mutex_coal, standard_coal_qty, max_mix_coal, opt_flag, top_k):
    # ------------------------------数据初始化-------------------------------------------------
    # 求每一行混煤率的最大公约数
    mix_ratio = mix_ratio / np.gcd(mix_ratio[:, 0], mix_ratio[:, 1]).reshape((-1, 1))
    # 每一行混煤率的和
    sum_mix_ratio = mix_ratio.sum(axis=1).reshape((-1, 1))
    # 混煤率和的最小公倍数(单仓煤仓煤量)
    max_ele = reduce(lambda c_, d_: lcm(int(c_), int(d_)), sum_mix_ratio.flatten(order="C").tolist())
    # 混煤比例元素集合
    ele_s = np.unique(mix_ratio / np.tile(sum_mix_ratio, (1, 2)) * max_ele)
    # 煤仓数
    m = container_constraint.shape[0]
    # 煤种数
    n = coal_info.shape[0]
    # 总煤量单位
    total_quality = np.sum(container_constraint[:, 0]) * max_ele
    # 煤仓启用标志
    container_enable_flag = container_constraint[:, 0] != 0

    # -----------------------------开始建模-------------------------------------------------
    # 待约束变量(整数)
    x = cp.Variable((m, n), integer=True)
    # 二元{0，1}辅助变量(x > 0  时z=1，x==0 时 z=0)
    z0 = cp.Variable((m, n), boolean=True)
    z1 = cp.Variable((m * n, ele_s.shape[0]), boolean=True)
    z2 = cp.Variable(n, boolean=True)
    # 连续变量，X-A >= y && A-X >= y && y > 0 abs(x-A) > 0 的线性变换写法
    y = cp.Variable((m, n))
    # 约束0：正整数约束，煤仓存煤量非负
    constraint0 = [x >= 0]
    # 约束01：给煤机出力一致性约束
    constraint1 = [cp.abs(cp.sum(x[container_enable_flag, :], axis=1) - max_ele) <= epsilon]
    # 约束2：单仓上煤总数约束(构造二元辅助变量，计算二元辅助变量的值间接计算非零整数)
    constraint2 = []

    for i in range(m):
        constraint2.append(x[i, :] <= max_ele * z0[i, :])
        constraint2.append(x[i, :] >= -max_ele * (1 - z0[i, :]))
        constraint2.append(cp.sum(z0[i, :]) <= 2)

    # 约束3：煤仓上煤比例约束在固定集合{ele_s} 中
    constraint3 = [cp.sum(z1, 1) == 1, z1 @ ele_s == x.flatten(order="C")]

    # 约束4：煤量约束，所有仓的煤量的总数小于存煤量
    constraint4 = []
    # constraint4 = [cp.sum(x, axis=0) / total_quality <= (coal_info[:, 1] / feeder_capacity)]

    # 约束5：机组煤质约束
    constraint5 = [(unit_constraint[:, 0] <= cp.sum(x, axis=0) @ coal_info[:, 2:-1] / total_quality),
                   (cp.sum(x, axis=0) @ coal_info[:, 2:-1] / total_quality <= unit_constraint[:, 1])]
    # 约束6：煤仓煤质约束
    lb_index = np.arange(6, 18, 2)
    ub_index = lb_index + 1
    temp = x @ coal_info[:, 2: - 1] / max_ele
    constraint6 = [(container_constraint[:, lb_index] * container_constraint[:, 0].reshape((-1, 1)) <= temp), (
            temp <= container_constraint[:, ub_index] * container_constraint[:, 0].reshape((-1, 1)))]

    # 约束7：指定挥发分约束(暂无)
    constraint7 = [x >= -1]  # Implement as needed

    # 约束8：煤仓启用约束(基本没用)
    r, _ = np.where(container_constraint[:, 0].reshape(-1, 1) == 0)
    constraint8 = [cp.sum(x[r, :]) == 0]

    # 约束9：煤仓固定比例约束
    constraint9 = []  # Implement as needed
    container_coal = container_constraint[:, 2: 6]
    r, _ = np.where(np.sum(container_coal, 1).reshape(-1, 1) != 0)
    r_len = r.shape[0]
    for i in range(r_len):
        ttt = max_ele / sum(container_coal[r[i], [2, 3]])
        constraint9.append(x[r[i], container_coal[r[i], [0, 1]]] == container_coal[r[i], [2, 3]] * ttt)

    # 约束10：煤种互斥约束
    constraint10 = []
    for index, g in enumerate(mutex_coal):
        for i in range(m):
            constraint10.append(x[i, g] - max_ele * z0[i, g] <= 0)
            constraint10.append(z0[i, g] - x[i, g] <= 0)
            constraint10.append(cp.sum(z0[i, g]) <= 1)

    # 约束11：热值守恒约束(属于脱裤子放屁约束)
    constraint11 = []  # Implement as needed
    # 约束12：最大煤种约束
    # 需要构造二元辅助变量，将非零的变量的数量小于某个值的问题
    constraint12 = [cp.sum(x, axis=0) <= max_ele * m * z2,
                    cp.sum(x, axis=0) >= -1 * max_ele * m * (1 - z2),
                    cp.sum(z2) <= max_mix_coal]

    # 目标函数
    # 煤价最低
    obj = None
    if opt_flag == 1:
        obj = cp.sum(x, axis=0) @ coal_info[:, -1]
    # 最环保（硫分最合理）
    elif opt_flag == 2:
        lower_s = np.min([unit_constraint[1, 1], np.min(container_constraint[:, 9])])
        obj = lower_s * total_quality - cp.sum(x, axis=0) @ coal_info[:, 4]
    # 给煤机最小出力（热值最合理）
    elif opt_flag == 3:
        lower_q = np.min([unit_constraint[0, 1], np.min(container_constraint[:, 7])])
        obj = lower_q * total_quality - cp.sum(x, axis=0) @ coal_info[:, 3]

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
    constraints.extend(constraint12)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.SCIPY)
    # 求解目标
    # mix_cases = []
    # mix_infos = []
    # mix_prices = []
    # for i in range(top_k):
    mix_case = []
    mix_info = []
    mix_price = []
    if problem.status == cp.OPTIMAL:
        solution = x.value
        # Mix info
        mix_info = np.sum(solution, axis=0) @ coal_info[:, 2:-1] / total_quality
        # mix_infos.append(mix_info.tolist())
        q = mix_info[0]
        coal_mass = standard_coal_qty * 7000 / q
        mix_case = coal_mass / total_quality * solution
        # mix_cases.append(mix_case.tolist())
        result = np.around(solution).astype(int)
        mix_price = np.sum(result, axis=0) @ coal_info[:, -1] / total_quality
        # mix_prices.append(mix_price.tolist())

        with np.printoptions(precision=2, formatter={'float_kind': '{:0.2f}'.format}):
            print("*************optimize result is:*************")
            print("\n- mix_integer:")
            print(result)
            print("\n- mix_case:")
            print(mix_case)
            print("\n- mix_info:")
            print(mix_info)
            print("\n- mix_price:")
            print(f"{mix_price:.2f}")
    else:
        raise Exception("Optimization failed!")
        # 等效于abs(x-x.value)>=0但是abs()>=0是一个非凸的问题,需要构造连续辅助变量进行调整
    # constraints.extend([x - x.value <= y, x.value - x <= y, y >= 0])
    return mix_case, mix_info, mix_price


if __name__ == '__main__':
    mat_data = scio.loadmat("../test_data/mix_coal/new.mat")
    # 使用示例
    coalInfo = mat_data["coalInfo"]  # 替换为实际的数据
    unitConstraint = mat_data["unitConstraint"]
    containerConstraint = mat_data["containerConstraint"]
    feederCapacity = mat_data["feederCapacity"]  # 替换为实际的数据
    mixRatio = mat_data["mixRatio"]
    mutexCoal = mat_data["mutexCoal"]
    standardCoalQty = mat_data["standardCoalQty"]  # 替换为实际的数据
    maxMixCoal = mat_data["maxMixCoal"]
    optFlag = mat_data["optFlag"]  # 替换为实际的数据
    topK = mat_data["topK"]  # 替换为实际的数据
    c = coal_mixed_integer_optimization(coalInfo, unitConstraint, containerConstraint, feederCapacity, mixRatio,
                                        mutexCoal, standardCoalQty, np.array([[3]]), optFlag, topK)
    cp.installed_solvers()
