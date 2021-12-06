# 将value_list中的值映射成 [lower_bound , len(set(value_list)) + lower_bound)区间，并返回用于映射的map
def get_reindex_map(value_list, lower_bound):
    assert type(value_list) is list
    reindex_map = dict()
    for v in value_list:
        if v not in reindex_map.keys():
            reindex_map[v] = lower_bound
            lower_bound += 1
    return reindex_map


# 根据交互次数删除user
# def remove_users(inter_num):
