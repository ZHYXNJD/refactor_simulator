import numpy as np

def LD(dispatch_observ, method, reject_nonpositive=False):


    dispatch_action = []
    if reject_nonpositive:
        dispatch_observ = [pair for pair in dispatch_observ if pair[2] > 0]
        if not dispatch_observ:
            return dispatch_action

    # 我们使用 set() 来自动去重，这比 np.unique() 更快
    order_ids_set = set()
    driver_ids_set = set()

    # 我们需要遍历一次列表来获取所有 ID
    for pair in dispatch_observ:
        order_ids_set.add(pair[0])
        driver_ids_set.add(pair[1])

    l_orders = np.array(list(sorted(order_ids_set)))  # 转换为 NumPy 数组
    l_drivers = np.array(list(sorted(driver_ids_set)))  # 转换为 NumPy 数组

    M = len(l_orders)  # the number of orders
    N = len(l_drivers)  # the number of drivers

    order_to_index = {order_id: i for i, order_id in enumerate(l_orders)}
    driver_to_index = {driver_id: i for i, driver_id in enumerate(l_drivers)}

    # coefficients and parameters, formulated as M * N matrix
    non_exist_link_value = -9999 # this value should be  smaller when use a trained q-table by large dataset
    matrix_reward = non_exist_link_value + np.zeros([M, N])  # reward     # this value should be smaller than any possible weights
    matrix_flag = np.zeros([M, N])  # pick up distance
    matrix_x_variables = np.zeros([M, N])  # 1 means there is potential match. otherwise, 0

    # 放弃 DataFrame，直接遍历原始输入 list (dispatch_observ)
    # 假设原始输入 dispatch_observ 是一个 list of lists: [order_id, driver_id, reward, flag]

    for pair in dispatch_observ:  # 注意：这里用的是函数的原始输入
        order_id = pair[0]
        driver_id = pair[1]
        reward = pair[2]
        flag = pair[3]

        # 用 O(1) 速度的字典查找替换 O(N*M) 的 np.where
        if order_id in order_to_index and driver_id in driver_to_index:
            m = order_to_index[order_id]
            n = driver_to_index[driver_id]

            matrix_reward[m, n] = 0. + reward
            matrix_flag[m, n] = flag
            matrix_x_variables[m, n] = 1

    # ------------------ [新代码开始] 方案二：高效的初始解构建 ------------------

    # 1. 从矩阵中提取所有有效的 "配对"
    #    我们不再使用 list[dict]，而是直接从已构建的矩阵中获取数据。
    m_indices, n_indices = np.where(matrix_x_variables == 1)

    num_valid_pairs = len(m_indices)

    # 2. 创建一个 NumPy 结构化数组 (一个高性能表格)
    #    这个数组包含 greedy 算法所需的所有信息：
    #    m (行索引), n (列索引), reward (奖励), flag (标志)
    pair_dtype = [('m', np.int32), ('n', np.int32), ('reward', np.float64), ('flag', np.float64)]
    pairs_array = np.empty(num_valid_pairs, dtype=pair_dtype)

    pairs_array['m'] = m_indices
    pairs_array['n'] = n_indices
    pairs_array['reward'] = matrix_reward[m_indices, n_indices]
    pairs_array['flag'] = matrix_flag[m_indices, n_indices]

    # 3. 高效排序 (替换原来的 if/elif 块)
    #    np.lexsort (字典序排序) 是实现多键排序的标准方法。
    #    注意：它的键是反向的 (最后一个键最先排)。
    if method in ['ir', 'rl', 'dynamic_matching','static_multi_choice']:
        # 键：(-reward) -> 按 reward 降序
        sort_keys = (-pairs_array['reward'],)
    elif method in ['ir_d', 'rl_d']:
        # 键：(flag, -reward) -> 按 reward 降序, 然后按 flag 升序
        sort_keys = (pairs_array['flag'], -pairs_array['reward'])
    elif method in ['d', 'd_rl']:
        # 键：(-flag, -reward) -> 按 reward 降序, 然后按 flag 降序
        sort_keys = (-pairs_array['flag'], -pairs_array['reward'])
    else:
        # 添加一个默认值, 以免出错
        # print(f"警告: 未知的初始解排序方法 '{method}', 默认使用 'ir' 规则。")
        sort_keys = (-pairs_array['reward'],)

    # 获取排序后的索引, 并应用它们
    sort_indices = np.lexsort(sort_keys)
    sorted_pairs = pairs_array[sort_indices]

    # 4. 高效的 Greedy 分配 (替换原来的 for od in ... 循环)
    #    我们使用布尔数组来跟踪分配, 这比 Python 的 set() 快得多。
    assigned_order_bool = np.zeros(M, dtype=bool)  # M = 订单数
    assigned_driver_bool = np.zeros(N, dtype=bool)  # N = 司机数

    initial_best_reward = 0.0
    initial_best_solution = np.zeros([M, N])  # 保持原样

    for pair in sorted_pairs:
        m = pair['m']  # 直接获取行索引
        n = pair['n']  # 直接获取列索引

        # 检查订单或司机是否已被分配
        if assigned_order_bool[m] or assigned_driver_bool[n]:
            continue
        if reject_nonpositive and pair['reward'] <= 0:
            continue

        # 标记为已分配
        assigned_order_bool[m] = True
        assigned_driver_bool[n] = True

        # [关键] 直接更新结果, 100% 避免了所有转换
        initial_best_solution[m, n] = 1
        initial_best_reward += pair['reward']

    # ------------------ [新代码结束] --------------------------------------------

    max_iterations = 25  # 25
    u = np.zeros(N)  # initialization
    Z_LB = initial_best_reward  # the lower bound of original problem that is initialized with the naive algorithm
    Z_UP = float('inf')  # infinity
    theta = 1.0
    # gap = 0.0001
    gap = 0.01

    # ---------------------------------------------Start iteration--------------------------------------------------
    for t in range(1, max_iterations + 1):
        matrix_x = np.zeros([M, N])
        QI = matrix_reward - u
        QI_masked = np.ma.masked_where(matrix_x_variables != 1, QI)
        idx_col_array = np.argmax(QI_masked, axis=1)
        idx_row_array = np.array(range(M))
        matrix_x[idx_row_array, idx_col_array] = 1

        # calculate Z_UP and Z_D
        Z_D = np.sum(u) + np.sum(matrix_reward * matrix_x)
        Z_UP = Z_D if Z_D < Z_UP else Z_UP

        # stage 1
        copy_matrix_reward = non_exist_link_value + np.zeros([M, N])
        copy_matrix_reward[idx_row_array, idx_col_array] = matrix_reward[idx_row_array, idx_col_array]
        copy_matrix_x = np.zeros([M, N])
        idx_col_array = np.array(range(N))
        idx_row_array = np.argmax(copy_matrix_reward, axis=0)
        con = copy_matrix_reward[idx_row_array, idx_col_array] > non_exist_link_value
        idx_col_array = idx_col_array[con]
        idx_row_array = idx_row_array[con]
        if len(idx_row_array) > 0:
            copy_matrix_x[idx_row_array, idx_col_array] = 1

        # stage 2
        index_existed_pair = np.where(copy_matrix_x == 1)
        index_drivers_with_order = np.unique(index_existed_pair[1])
        index_drivers_without_order = np.setdiff1d(np.array(range(N)), index_drivers_with_order)
        index_orders_with_driver = np.unique(index_existed_pair[0])
        index_orders_without_driver = np.setdiff1d(np.array(range(M)), index_orders_with_driver)

        if len(index_orders_without_driver) != 0:
            second_allocated_driver = []
            for m in index_orders_without_driver.tolist():
                con_second = np.isin(index_drivers_without_order, second_allocated_driver)
                if np.all(con_second):
                    break
                else:
                    reward_array = matrix_reward[m][index_drivers_without_order]
                    masked_reward_array = np.ma.masked_where(con_second, reward_array)
                    index_reward = np.argmax(masked_reward_array)
                    if masked_reward_array[index_reward] > 0:
                        index_driver = index_drivers_without_order[index_reward]
                        second_allocated_driver.append(index_driver)
                        copy_matrix_x[m][index_driver] = 1

        # stage 3
        new_Z_LB = np.sum(copy_matrix_x * matrix_reward)
        if new_Z_LB > Z_LB:
            Z_LB = new_Z_LB
            initial_best_solution = np.zeros([M, N])
            initial_best_solution[copy_matrix_x == 1] = 1

        # update u
        sum_m = np.sum(matrix_x, axis=0)
        sum = np.sum((1 - sum_m) ** 2)
        if sum == 0:
            sum = 0.00001  # given a small value
        k_t = theta * (Z_D - Z_LB) / sum

        u = u + k_t * (sum_m - 1) / t
        u[u < 0] = 0

        if (Z_UP == 0) or ((Z_UP - Z_LB) / Z_UP <= gap):
            matrix_x = initial_best_solution
            break
        if t == max_iterations:
            matrix_x = initial_best_solution
            break

    # solution
    index_existed = np.where(matrix_x == 1)
    for m, index_driver in zip(index_existed[0].tolist(), index_existed[1].tolist()):
        dispatch_action.append([l_orders[m], l_drivers[index_driver], matrix_reward[m][index_driver],
                                matrix_flag[m][index_driver]])

    return dispatch_action

if __name__ == '__main__':

    # 写一个脚本测试一下新的LD
    test_dt = [['a', 1, 120, 0],
                   ['b', 1, 110, 20],
                   ['c', 1, 130, 25],
                   ]
    dispatch_action = LD(test_dt, 'rl')
    print(f"{dispatch_action}")

