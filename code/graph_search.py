import numpy as np

def graph_search(state_arr, frontier, path=[], explored=[]):
    while True:
        path, frontier = find_low_cost_path(frontier, state_arr)
        explored.append(path)
        if state_arr.shape[0] == len(path):
            break

        actions = available_actions(path, state_arr)

        for a in actions:
            candidate_path = path[:]
            candidate_path.append([len(candidate_path), a])

            valid_path = False
            for leg in candidate_path:
                if valid_path:
                    break
                for prior_path in explored:
                    if leg not in prior_path:
                        frontier.append(candidate_path)
                        valid_path = True
                        break

    return path

def get_frontier(state_arr):
    actions = available_actions([], state_arr)
    frontier = [[[0, action]] for action in actions]
    return frontier

def available_actions(path, state_arr):
    action_space = set(range(state_arr.shape[1]))
    actions_taken = set(actions_from_path(path))
    return list(action_space - actions_taken)

def actions_from_path(path):
    if len(path) == 0:
        return []
    return [leg[1] for leg in path]

def find_low_cost_path(frontier, state_arr):
    path_costs = []
    for path in frontier:
        cost = sum([state_arr[leg[0]][leg[1]] for leg in path])
        path_costs.append(cost)

    low_cost_index = path_costs.index(min(path_costs))
    low_cost_path = frontier[low_cost_index]
    frontier.pop(low_cost_index)

    return low_cost_path, frontier
