"""
author:AiChao
date:2025-03-24
"""

import numpy as np
import cvxpy as cp


def output_opt_impl(container_coal_info, output_constraint, unit_constraint, total_qty):
    container_index = np.unique(container_coal_info[:, 0]).astype(int)
    container_ids = container_coal_info[:, 0].astype(int)
    layer_indices = container_coal_info[:, 1].astype(int)
    container_num = len(container_coal_info[:, 0])
    # 表示某一层煤消耗量
    x = cp.Variable(container_num)
    # 二进制变量，表示某一层煤是否被消耗
    y = cp.Variable(container_num, boolean=True)

    # 煤仓出力约束
    constraint0 = []
    for container_id in container_index:
        indices = np.where(container_ids == container_id)[0]
        constraint0.append(cp.sum(x[indices]) >= output_constraint[container_id, 1])
        constraint0.append(cp.sum(x[indices]) <= output_constraint[container_id, 2])

    # 机组约束
    constraint1 = [x @ container_coal_info[:, 3:-1] / total_qty >= unit_constraint[:, 0],
                   x @ container_coal_info[:, 3:-1] / total_qty <= unit_constraint[:, 1]]

    # 煤量约束
    constraint2 = [cp.sum(x) == total_qty]
    # 每层煤的出煤量不能为负
    constraint3 = [x >= 0, x <= container_coal_info[:, 2]]
    # 每层煤是否被消耗的约束
    constraint4 = [x <= cp.multiply(y, cp.max(container_coal_info[:, 2]))]

    # 煤层顺序约束
    constraint5 = []
    for container_id in container_index:
        indices = np.where(container_ids == container_id)[0]
        layer_indices_sorted = layer_indices[indices]
        sorted_indices = indices[np.argsort(layer_indices_sorted)]
        for i in range(len(sorted_indices) - 1):
            constraint5.append(y[sorted_indices[i]] <= y[sorted_indices[i + 1]])

    constraints = []
    constraints.extend(constraint0)
    constraints.extend(constraint1)
    constraints.extend(constraint2)
    constraints.extend(constraint3)
    constraints.extend(constraint4)
    constraints.extend(constraint5)

    # 寻优目标
    cost = x @ container_coal_info[:, -1]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.SCIPY)
    if problem.status == cp.OPTIMAL:
        solution = x.value
        print("\n - out_info:")
        print(solution)
        return solution
    else:
        raise Exception("Optimization failed!")


if __name__ == '__main__':
    print(cp.installed_solvers())
