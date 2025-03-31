"""
author:AiChao
date:2025-03-13
"""

import numpy as np
import cvxpy as cp


def coal_mixed_optimization_simple(coal_info, unit_constraint, total_qty, max_mix_coal_num, opt_flag):
    # ------------------------------数据初始化-------------------------------------------------
    # 煤种数
    n = coal_info.shape[0]
    # 总煤量单位
    # -----------------------------开始建模-------------------------------------------------
    # 待约束变量(整数)假设x1,x2,x3...xi...xn中任意的0=<xi<=10,并且x1+x2+x3+...+xn=10
    x = cp.Variable(n, integer=True)
    z = cp.Variable(n, boolean=True)
    # 约束0：配煤比例为非负
    constraint0 = [x >= 0, x <= 10]
    # 约束01：配煤比例和为1
    constraint1 = [cp.sum(x) == 10]
    # 约束2：机组煤质约束
    constraint2 = [(unit_constraint[:, 0] <= x @ coal_info[:, 2:-1] / 10),
                   (x @ coal_info[:, 2:-1] / 10 <= unit_constraint[:, 1])]

    # 约束3：最大煤种约束, 需要构造二元辅助变量，将非零的变量的数量小于某个值的问题
    # x<=z 可以理解为，如果x=0，那么z取的值为0和1，如果x>0 那么z取值一定是1，对z进行两个约束即可得到想要的约束条件
    constraint3 = [x <= 10 * z, cp.sum(z) <= max_mix_coal_num]
    # 约束4 煤量约束
    constraint4 = [x @ coal_info[:, 1] / 10 >= total_qty]

    # 目标函数
    # 煤价最低
    obj = None
    if opt_flag == 1:
        obj = x @ coal_info[:, -1]
    # 最环保（硫分最合理）
    elif opt_flag == 2:
        lower_s = np.min([unit_constraint[1, 1]])
        obj = lower_s - x @ coal_info[:, 4] / 10
    # 给煤机最小出力（热值最合理）
    elif opt_flag == 3:
        lower_q = np.min([unit_constraint[0, 1]])
        obj = lower_q - x @ coal_info[:, 3] / 10

    # 构建约束列表
    constraints = []
    constraints.extend(constraint0)
    constraints.extend(constraint1)
    constraints.extend(constraint2)
    constraints.extend(constraint3)
    constraints.extend(constraint4)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.SCIPY)
    if problem.status == cp.OPTIMAL:
        solution = x.value
        mix_case = np.around(solution) / 10
        mix_price = solution @ coal_info[:, -1] / 10
        mix_info = solution @ coal_info[:, 2:-1] / 10
        with np.printoptions(precision=2, formatter={'float_kind': '{:0.2f}'.format}):
            print("*************optimize result is:*************")
            print("- mix_integer:")
            print(solution)
            print("\n- mix_info:")
            print(mix_info)
            print("\n- mix_price:")
            print(mix_price)
    else:
        raise Exception("Optimization failed!")
    return mix_case, mix_info, mix_price


if __name__ == '__main__':
    cp.installed_solvers()
