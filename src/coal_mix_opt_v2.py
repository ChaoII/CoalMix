"""
author:AiChao
date:2023-11-24
"""
import json
from functools import reduce
from math import lcm
from log.log import logger
import cvxpy as cp
import numpy as np

epsilon = 0.009


def coal_mixed_integer_optimization_v2(coal_info, unit_constraint, container_constraint, mix_ratio,
                                       coal_quality, mix_coal_num, opt_flag):
    # ------------------------------数据初始化-------------------------------------------------
    # 求每一行混煤率的最大公约数
    gcd_s = 1
    for i in range(mix_ratio.shape[1]):
        gcd_s = np.gcd(gcd_s, mix_ratio[:, i])
    mix_ratio = mix_ratio / gcd_s.reshape((-1, 1))
    # 每一行混煤率的和
    sum_mix_ratio = mix_ratio.sum(axis=1).reshape((-1, 1))
    # 混煤率和的最小公倍数(单仓煤仓煤量)
    max_ele = reduce(lambda c_, d_: lcm(int(c_), int(d_)), sum_mix_ratio.flatten(order="C").tolist())
    # 混煤比例元素集合
    ele_s = np.unique(mix_ratio / np.tile(sum_mix_ratio, (1, mix_ratio.shape[1])) * max_ele)
    # 煤仓数
    m = container_constraint.shape[0]
    # 煤种数
    n = coal_info.shape[0]
    # 低负荷下总煤量单位
    total_quality_low = np.sum(container_constraint[:, 0]) * max_ele
    # 高负荷下总煤量单位
    total_quality_high = np.sum(container_constraint[:, 1]) * max_ele
    # 低负荷下煤仓启动索引
    container_low_index = container_constraint[:, 0] != 0
    # 高负荷下煤仓启用索引
    container_high_index = container_constraint[:, 1] != 0
    # -----------------------------开始建模-------------------------------------------------
    # 待约束变量(整数)
    x = cp.Variable((m, n), integer=True)
    # 二元{0，1}辅助变量(x > 0  时z=1，x==0 时 z=0)
    z0 = cp.Variable((m, n), boolean=True)
    z1 = cp.Variable((m * n, ele_s.shape[0]), boolean=True)
    z2 = cp.Variable(n, boolean=True)
    z3 = cp.Variable(ele_s.shape[0], boolean=True)
    # 连续变量，X-A >= y && A-X >= y && y > 0 abs(x-A) > 0 的线性变换写法
    y = cp.Variable((m, n))
    # 约束0：正整数约束，煤仓存煤量非负
    constraint0 = [x >= 0]
    # 约束01：给煤机出力一致性约束
    constraint1 = [cp.abs(cp.sum(x[container_low_index, :], axis=1) - max_ele) <= epsilon,
                   cp.abs(cp.sum(x[container_high_index, :], axis=1) - max_ele) <= epsilon]
    # 约束2：单仓上煤总数约束(构造二元辅助变量，计算二元辅助变量的值间接计算非零整数),如果指定煤种比例，则该仓煤种比例需要小于等于指定比例数量
    constraint2 = []
    for i in range(m):
        constraint2.append(x[i, :] <= max_ele * z0[i, :])
        constraint2.append(x[i, :] >= -max_ele * (1 - z0[i, :]))
        constraint2.append(cp.sum(z0[i, :]) <= np.max(np.sum(mix_ratio > 0, axis=1)))
    # 约束3：煤仓上煤比例约束在固定集合{ele_s} 中
    constraint3 = [cp.sum(z1, 1) == 1, z1 @ ele_s == x.flatten(order="C")]
    # 约束4：机组煤质约束
    # 低负荷约束
    constraint4 = [(unit_constraint[0][:, 0] <= cp.sum(x[container_low_index, :], axis=0) @
                    coal_info[:, 2:-1] / total_quality_low),
                   (cp.sum(x[container_low_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_low <=
                    unit_constraint[0][:, 1])]
    # 高负荷约束
    constraint4_high = [(unit_constraint[1][:, 0] <= cp.sum(x[container_high_index, :], axis=0) @
                         coal_info[:, 2:-1] / total_quality_high),
                        (cp.sum(x[container_high_index, :], axis=0) @ coal_info[:, 2:-1] / total_quality_high <=
                         unit_constraint[1][:, 1])]
    constraint4.extend(constraint4_high)

    # 约束5煤量约束
    constraint5_ = []
    constraint5_low = [cp.sum(x, axis=0) / total_quality_low <= (coal_info[:, 1] / coal_quality[0])]
    constraint5_height = [cp.sum(x, axis=0) / total_quality_high <= (coal_info[:, 1] / coal_quality[1])]
    constraint5_.extend(constraint5_low)
    constraint5_.extend(constraint5_height)

    # 约束5：煤仓煤质约束
    lb_index = np.arange(7, 19, 2)
    ub_index = lb_index + 1
    temp = x @ coal_info[:, 2: - 1] / max_ele
    constraint5 = [(container_constraint[:, lb_index] <= temp), (temp <= container_constraint[:, ub_index])]

    # 约束6：指定挥发分约束(暂无)
    constraint6 = [x >= -1]  # Implement as needed

    # 约束7：煤仓启用约束(基本没用)
    r = np.any([container_constraint[:, 0].reshape(-1, 1), container_constraint[:, 1].reshape(-1, 1)], axis=0)
    r1, _ = np.where(r == 0)
    # r2, _ = np.where(container_constraint[:, 1].reshape(-1, 1) == 0)
    constraint7 = [cp.sum(x[r1, :]) == 0]

    # 约束8：煤仓固定比例约束
    constraint8 = []  # Implement as needed
    container_coal = container_constraint[:, 5: 7]
    r, _ = np.where(np.sum(container_coal, 1).reshape(-1, 1) != 0)
    for i in r:
        coal_kind = container_coal[i, 0]
        coal_rate = container_coal[i, 1]
        if not isinstance(coal_rate, list):
            coal_rate = [coal_rate]
        if not isinstance(coal_kind, list):
            coal_kind = [coal_kind]
        if len(coal_rate) != len(coal_kind):
            raise ValueError("每种和每种比例元素的长度比必须相等")
        # 注意当只指定煤种，但是不指定比例时要做额外的操作
        ttt = 1 if sum(coal_rate) == 0 else max_ele / sum(coal_rate)
        for kind, rate in zip(coal_kind, coal_rate):
            if sum(coal_rate) == 0:
                logger.warning(
                    f"指定了煤种[{kind}]但是未指定煤种比例即煤种比例给出的是{rate}，这可能导致求解失败")
                constraint8.extend([cp.sum(z3) == 1, z3 @ ele_s == x[i, kind], x[i, kind] >= 1])
            else:
                constraint8.append(x[i, kind] == rate * ttt)

    # 约束9：最大煤种约束
    # 需要构造二元辅助变量，将非零的变量的数量小于某个值的问题
    constraint9 = [cp.sum(x, axis=0) <= max_ele * m * z2,
                   cp.sum(x, axis=0) >= -1 * max_ele * m * (1 - z2),
                   cp.sum(z2) <= mix_coal_num]

    # 目标函数
    # 煤价最低
    obj = None
    if opt_flag == 1:
        obj = cp.sum(x, axis=0) @ coal_info[:, -1]
    # 最环保（硫分最合理）
    elif opt_flag == 2:
        lower_s = np.min(
            [unit_constraint[0, 1, 1], unit_constraint[1, 1, 1], np.min(container_constraint[container_high_index, 9])])
        obj = lower_s * total_quality_high - cp.sum(x, axis=0) @ coal_info[:, 4]
    # 给煤机最小出力（热值最合理）
    elif opt_flag == 3:
        lower_q = np.min(
            [unit_constraint[0, 0, 1], unit_constraint[1, 0, 1], np.min(container_constraint[container_high_index, 7])])
        obj = lower_q * total_quality_high - cp.sum(x, axis=0) @ coal_info[:, 3]

    # 构建约束列表
    constraints = []
    constraints.extend(constraint0)
    constraints.extend(constraint1)
    constraints.extend(constraint2)
    constraints.extend(constraint3)
    constraints.extend(constraint4)
    constraints.extend(constraint5_)
    constraints.extend(constraint5)
    constraints.extend(constraint6)
    constraints.extend(constraint7)
    constraints.extend(constraint8)
    constraints.extend(constraint9)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.SCIPY)
    # 求解目标
    if problem.status == cp.OPTIMAL:
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
    else:
        raise Exception("Optimization failed!")
        # 等效于abs(x-x.value)>=0但是abs()>=0是一个非凸的问题,需要构造连续辅助变量进行调整
    return result.tolist(), [mix_case_low.tolist(), mix_case_high.tolist()], \
        [mix_info_low.tolist(), mix_info_high.tolist()], [
        mix_price_low.tolist(), mix_price_high.tolist()]


if __name__ == '__main__':
    mat_data = json.load(open("../test_data/mix_coal/input_v2.json"))
    # 使用示例
    coal_info = mat_data["coal_info"]  # 替换为实际的数据
    unit_constraint = mat_data["unit_constraint"]
    container_constraint = mat_data["container_constraint"]
    mix_ratio = mat_data["mix_ratio"]
    coal_quality = mat_data["coal_quality"]  # 替换为实际的数据
    mix_coal_num = mat_data["mix_coal_num"]
    c = coal_mixed_integer_optimization_v2(np.array(coal_info), np.array(unit_constraint),
                                           np.array(container_constraint), np.array(mix_ratio),
                                           coal_quality, mix_coal_num)
