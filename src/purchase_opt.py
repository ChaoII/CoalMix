import cvxpy as cp
import numpy as np


def get_stock(coal_info, mount):
    rest = []
    stop_index = -1
    rem = mount
    for index, coal in enumerate(coal_info):
        if rem > coal_info[index, 5]:
            rest.append(coal_info[index, :])
            rem = rem - coal_info[index, 5]
        else:
            stop_index = index
            last_coal = np.copy(coal)
            last_coal[5] = rem
            rest.append(last_coal)
            if coal_info[index, 5] - rem == 0:
                stop_index += 1
            coal_info[index, 5] = coal_info[index, 5] - rem
            break
    remained = coal_info[stop_index:, :]
    return np.array(rest), remained


def purchase_opt_impl(market_coal: np.ndarray, stock_coal: np.ndarray, ending_inventory: np.ndarray,
                      burning_constraint: np.ndarray, total_purchase: float, replace_rate: float, max_purchase_kind=4):
    """
    :param market_coal: 市场存煤信息
    :param stock_coal: 煤场信息
    :param ending_inventory: 期末库存约束
    :param burning_constraint: 入炉燃烧约束
    :param total_purchase: 总采购量
    :param replace_rate: 置换率
    :return:
    """
    m, n = market_coal.shape
    # 入炉煤燃烧量
    burning_mount = burning_constraint[0]
    # 入炉的热值
    burning_q = burning_constraint[1]
    # 入炉的硫
    burning_s = burning_constraint[2]
    # 入炉的挥发会
    burning_h = burning_constraint[3]

    # 煤场存煤量
    stocking_mount = ending_inventory[0]
    # 入炉的热值
    stocking_q = ending_inventory[1]
    # 入炉的硫
    stocking_s = ending_inventory[2]
    # 入炉的挥发会
    stocking_h = ending_inventory[3]

    x = cp.Variable(m)
    z = cp.Variable(m, boolean=True)

    # 煤种选取约束
    constraint0 = [
        x <= total_purchase * z,  # 如果x[i] > 0, 则z[i] 必须为1
        cp.sum(z) <= max_purchase_kind  # 最多有4个x[i]可以大于0
    ]
    # 煤量和约束
    constraint1 = [cp.sum(x) == total_purchase]
    # 市场存煤限值约束
    constraint2 = [x >= market_coal[:, 6], x <= market_coal[:, 7]]
    # 固定煤量约束
    nonzero_indices = np.nonzero(market_coal[:, 5])[0]
    nonzero_values = market_coal[:, 5][nonzero_indices]
    constraint3 = [x[i] == value for i, value in zip(nonzero_indices, nonzero_values)]
    # 采购的煤一部分直接燃烧，一部分与煤场的旧煤进行置换
    # 烧煤场库存煤的量
    stocking_burning_mount = burning_mount * replace_rate
    # 烧市场采购煤量
    market_burning_mount = burning_mount * (1 - replace_rate)
    # 市场采购煤入库的量
    market_replacing_mount = total_purchase - market_burning_mount

    # 市场采购部分入炉百分比
    burning_rate = market_burning_mount / total_purchase
    # 市场采购部分置换的百分比
    replacing_rate = market_replacing_mount / total_purchase
    # 获取煤场烧旧煤的煤种结构，以及剩下的存煤结构
    if stocking_burning_mount > 0 and stock_coal.size > 0:
        old_coal_info, rem_coal_info = get_stock(stock_coal, stocking_burning_mount)
        stocking_burning = old_coal_info[:, [0, 5]]
        stock_q = np.sum(old_coal_info[:, 5] * old_coal_info[:, 2])
        stock_s = np.sum(old_coal_info[:, 5] * old_coal_info[:, 3])
        stock_h = np.sum(old_coal_info[:, 5] * old_coal_info[:, 4])
        stock_q_1 = np.sum(rem_coal_info[:, 5] * rem_coal_info[:, 2])
        stock_s_1 = np.sum(rem_coal_info[:, 5] * rem_coal_info[:, 3])
        stock_h_1 = np.sum(rem_coal_info[:, 5] * rem_coal_info[:, 4])
    elif stocking_burning_mount > 0 >= stock_coal.size:
        raise Exception("置换量错误，无库存，无法置换")
    elif stocking_burning_mount <= 0 < stock_coal.size:
        stock_q = 0
        stock_s = 0
        stock_h = 0
        stock_q_1 = np.sum(stock_coal[:, 5] * stock_coal[:, 2])
        stock_s_1 = np.sum(stock_coal[:, 5] * stock_coal[:, 3])
        stock_h_1 = np.sum(stock_coal[:, 5] * stock_coal[:, 4])
        stocking_burning = np.array([])
    else:
        stock_q = 0
        stock_s = 0
        stock_h = 0
        stock_q_1 = 0
        stock_s_1 = 0
        stock_h_1 = 0
        stocking_burning = np.array([])

    # ------------------ 入炉煤煤质约束-------------------------
    # 热值约束
    market_q = cp.sum(cp.multiply(burning_rate * x, market_coal[:, 2]))
    constraint4 = [(market_q + stock_q) / burning_mount >= burning_q]
    # 硫分约束
    market_s = cp.sum(cp.multiply(burning_rate * x, market_coal[:, 3]))
    constraint5 = [(market_s + stock_s) / burning_mount <= burning_s]
    # 挥发会约束
    market_h = cp.sum(cp.multiply(burning_rate * x, market_coal[:, 4]))
    constraint6 = [(market_h + stock_h) / burning_mount >= burning_h]
    # ---------------- 入场煤质约束-------------------------
    # 热值约束
    market_q_1 = cp.sum(cp.multiply(replacing_rate * x, market_coal[:, 2]))
    constraint7 = [(market_q_1 + stock_q_1) / stocking_mount >= stocking_q]
    # 硫分约束
    market_s_1 = cp.sum(cp.multiply(replacing_rate * x, market_coal[:, 3]))
    constraint8 = [(market_s_1 + stock_s_1) / stocking_mount <= stocking_s]
    # 挥发会约束
    market_h_1 = cp.sum(cp.multiply(replacing_rate * x, market_coal[:, 4]))
    constraint9 = [(market_h_1 + stock_h_1) / stocking_mount >= stocking_h]

    obj = x @ market_coal[:, 1]

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

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.SCIPY)
    if problem.status == cp.OPTIMAL:
        solution = x.value
        return solution, stocking_burning
    else:
        raise Exception("Optimization failed!")


if __name__ == '__main__':
    market_coal = np.loadtxt("test_data/purchase_data/market_coal.csv", delimiter=",", encoding="utf8")
    stock_coal = np.loadtxt("test_data/purchase_data/stock_coal.csv", delimiter=",", encoding="utf8")
    ending_inventory = np.loadtxt("test_data/purchase_data/ending_inventory.csv", delimiter=",", encoding="utf8")
    burning_constraint = np.loadtxt("test_data/purchase_data/burning_constraint.csv", delimiter=",", encoding="utf8")
    total_purchase = 12.0
    replace_rate = 0.5

    purchase_opt_impl(market_coal, stock_coal, ending_inventory, burning_constraint, total_purchase, replace_rate,
                      max_purchase_kind=10)
