'''
reinforcement learning simulation environment
'''

import sys
import os
import math
import time

import torch
from scipy.special import binom

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from new_reward_project.simulation_environment.Shortest_path import *
from new_reward_project.simulation_environment.network import *


class generate_synchronous_timestep_environment_with_directional_action():
    '''
    generate environment with synchronous timestep and directional action for multiple agents
    '''

    def __init__(
            self, experienced_travel_time: np.ndarray, node_id_to_grid_id: pd.DataFrame, ac_dim: int,
            action_interval: int, num_of_action_interval: int, num_of_cal_reward: int,
            epsilon: float, vehicle_num: int, seed: int,
    ):
        '''

        :param experienced_travel_time:
        :param node_id_to_grid_id: need contain column 'grid_id', 'node_id',
        'node_coordinate_row', 'node_coordinate_col'.
        :param ac_dim: the dimension of action, only support 4 or 8
        :param action_interval: seconds, every timestep's time interval
        :param num_of_action_interval: num_of_action_interval * action_interval = interval to cal reward
        :param num_of_cal_reward: number of timestep = num_of_cal_reward * num_of_action_interval
        :param epsilon: (0, 1]
        :param vehicle_num:
        :param seed:
        :return:
        '''

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        assert ac_dim == 4 or ac_dim == 8, '4 and 8 are ac_dim allowed value.'

        self.experienced_travel_time = experienced_travel_time
        self.max_experienced_travel_time = self.experienced_travel_time.max()
        self.node = node_id_to_grid_id[['node_id', 'grid_id', 'node_coordinate_row', 'node_coordinate_col']]
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
        self.node['node_cover'] = 0
        self.grid = pd.DataFrame({'grid_id': self.node['grid_id'].unique()})
        self.grid.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        self.node_id_min = self.node['node_id'].min()
        self.node_id_max = self.node['node_id'].max()
        self.ac_dim = ac_dim
        self.action_interval = action_interval

        self.num_of_action_interval = num_of_action_interval
        self.interval_to_cal_reward = num_of_action_interval * self.action_interval
        self.num_of_cal_reward = num_of_cal_reward
        self.past_time = 0
        self.cal_reward_time = 0
        self.episode_duration = self.interval_to_cal_reward * self.num_of_cal_reward
        self.epsilon = epsilon
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        self.vehicle_action_paths = []
        self.node_vehicle_id_cover_name = []
        self.grid_vehicle_id_cover_name = []
        self.binomial_coefficient_lookup_table = {}
        for i in range(self.vehicle_num):
            self.node_vehicle_id_cover_name.append('node_vehicle_id_cover{}'.format(i))
            self.grid_vehicle_id_cover_name.append('grid_vehicle_id_cover{}'.format(i))
            self.node['node_vehicle_id_cover{}'.format(i)] = 0
            self.grid['grid_vehicle_id_cover{}'.format(i)] = 0
            self.vehicle_states.append([
                -1,  # location1
                -1,  # location2
                0,  # remaining_time
            ])
            self.vehicle_action_paths.append(- 10000)
            self.binomial_coefficient_lookup_table[i] = binom(self.vehicle_num, i)
        self.binomial_coefficient_lookup_table[self.vehicle_num] = binom(self.vehicle_num, self.vehicle_num)
        self.vehicle_states = np.array(self.vehicle_states)
        self.got_reward = 0

    def grid_vehicle_id_cover_count(self):
        grid_vehicle_id_cover = self.node.groupby(['grid_id'], as_index=False)[self.node_vehicle_id_cover_name].sum()
        grid_vehicle_id_cover.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        return grid_vehicle_id_cover

    def map_grid_vehicle_id_cover_to_node(self):
        if 'grid_vehicle_id_cover0' in self.node.columns.values:
            self.node.drop(columns=self.grid_vehicle_id_cover_name, inplace=True)
        self.node = pd.merge(
            self.node, self.grid[['grid_id'] + self.grid_vehicle_id_cover_name], on=['grid_id'])
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)

    def p_hat_statistic(self):
        if self.cal_reward_time == 0:
            self.grid['p_hat{}'.format(self.cal_reward_time)] = self.epsilon
        else:
            grid_cover_or_not = self.grid[self.grid_vehicle_id_cover_name].values.sum(axis=1)
            grid_cover_or_not[np.where(grid_cover_or_not > 0)] = 1
            self.grid['p_hat{}'.format(self.cal_reward_time)] = (1 - 1 / (self.cal_reward_time + 1)) * self.grid[
               'p_hat{}'.format(self.cal_reward_time - 1)].values + grid_cover_or_not / (self.cal_reward_time + 1)
        self.grid['p_hat'] = self.grid['p_hat{}'.format(self.cal_reward_time)]

    def reset_grid_weight(self, grid_weight: Union[pd.DataFrame, None]):
        if grid_weight is not None:
            self.grid = grid_weight
        else:
            self.grid['weight'] = np.random.dirichlet(
                np.ones(shape=len(self.grid), dtype=np.int), size=1).flatten()

    def map_grid_info_to_node(self, add_col_names: list):
        '''

        :param add_col_names: items in add_col_name only support 'grid_vehicle_id_cover', 'weight' and 'p'
        :return:
        '''
        repetitive_col_names = list(set(self.node.columns.values) & set(add_col_names))
        self.node.drop(columns=repetitive_col_names, inplace=True)
        self.node = pd.merge(
            self.node, self.grid[['grid_id'] + add_col_names], on=['grid_id'])
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)

    def cal_reward(self):
        weight = self.grid['weight'].values
        grid_vehicle_id_cover = self.grid[self.grid_vehicle_id_cover_name].values.sum(axis=1).astype(int)
        p_hat = self.grid['p_hat{}'.format(self.cal_reward_time)].values
        cover_where = np.where(grid_vehicle_id_cover > 0)
        no_cover_where = np.where(grid_vehicle_id_cover == 0)
        # cal s
        s = 1 - (1 - p_hat) ** (1 / self.vehicle_num)
        # cal p
        p = np.zeros_like(weight, dtype=np.float32)
        p[no_cover_where] = p_hat[no_cover_where]
        binomial_coefficient = grid_vehicle_id_cover.copy()
        for unique_value in np.unique(grid_vehicle_id_cover):
            binomial_coefficient[np.where(grid_vehicle_id_cover == unique_value)] = \
                self.binomial_coefficient_lookup_table[unique_value]
        p[cover_where] = binomial_coefficient[cover_where] * (s[cover_where] ** grid_vehicle_id_cover[cover_where]) * \
            ((1 - s[cover_where]) ** (self.vehicle_num - grid_vehicle_id_cover[cover_where])) / p_hat[cover_where]
        # cal I
        I = np.log(p)
        I[cover_where] = - I[cover_where]
        # cal reward
        reward = weight * I
        reward = reward.sum()
        self.got_reward += reward
        return reward

    def reset(self, grid_weight):
        '''
        reset the agents(vehicles) position and grid link_weight(rewards)
        :return:
        '''
        self.past_time = 0
        self.cal_reward_time = 0
        loc = np.random.randint(low=self.node_id_min, high=self.node_id_max + 1, size=self.vehicle_num)
        self.vehicle_states[:, 1] = loc  # location2
        self.vehicle_states[:, 0] = self.vehicle_states[:, 1].copy()  # location1
        self.vehicle_states[:, 2] = 0  # remaining_time
        self.reset_grid_weight(
            grid_weight=grid_weight,
        )
        self.got_reward = 0
        self.p_hat_statistic()
        self.map_grid_info_to_node(add_col_names=['weight', 'p_hat'])

        vehicle_loc = np.zeros((len(self.node), self.vehicle_num))
        vehicle_loc[(loc, range(self.vehicle_num))] = 1
        self.node[self.node_vehicle_id_cover_name] = vehicle_loc
        self.grid[self.grid_vehicle_id_cover_name] = \
            self.grid_vehicle_id_cover_count()[self.node_vehicle_id_cover_name].values
        self.map_grid_vehicle_id_cover_to_node()

        self.vehicle_action_paths = [- 10000] * self.vehicle_num

        node_weight = self.node['weight'].values
        vehicle_states = copy.deepcopy(self.vehicle_states)
        grid_vehicle_id_cover = self.node[self.grid_vehicle_id_cover_name].values
        p_hat = self.node['p_hat'].values

        return vehicle_states, node_weight, grid_vehicle_id_cover, p_hat

    def cal_angle(self, point_1: Union[tuple, list], point_2: Union[tuple, list], point_3: Union[tuple, list]):
        """
        calculating the Angle between the vertices at point_2
        :param point_1:
        :param point_2: Tuple, the coordinate of point_2
        :param point_3:
        :return: Float, the Angle between the vertices at point_2
        """

        a = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (
                point_2[1] - point_3[1]))
        b = math.sqrt((point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (
                point_1[1] - point_3[1]))
        c = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
                point_1[1] - point_2[1]))
        # A = math.degrees(math.acos((a * a - b * b - c * c)/(-2 * b * c)))
        middle_cal_result = round((b * b - a * a - c * c) / (-2 * a * c), 4)
        B = math.degrees(math.acos(middle_cal_result))
        # C = math.degrees(math.acos((c * c - a * a - b * b)/(-2 * a * b)))

        return B

    def determine_path(self, vehicle_state: Union[tuple, list, np.ndarray], action: int) -> list:
        '''
        deciding the final node destination
        :param vehicle_state: agent's state
        :param action: East, south, west, north four directions or plus
        northwest, northeast, southeast, southwest 8 directions totally
        :return:
        '''

        vehicle_loc = vehicle_state[1]
        remaining_time = vehicle_state[2]

        arrive_path = []
        start = vehicle_loc
        start_time = self.action_interval - remaining_time
        if start_time < 0:
            start_time = 0
        came_from, node_list = dijkstra_search(
            cost=self.experienced_travel_time,
            start=start,
            start_time=start_time,
            end_time=start_time + self.max_experienced_travel_time + 60,
            node_length=self.ac_dim,
        )
        for node in node_list:
            path = reconstruct_path(
                came_from=came_from,
                start=start,
                goal=node,
            )
            arrive_path.append(path)

        # calculate the Angle between the action vector and the destination vector,
        # and choose destination with the smallest Angle as final action destination
        vehicle_coord = self.node.loc[vehicle_loc, ['node_coordinate_row', 'node_coordinate_col']].values.astype(int)
        if self.ac_dim == 4:
            if action == 0:
                action_direction = (vehicle_coord[0] - 1, vehicle_coord[1])
            elif action == 1:
                action_direction = (vehicle_coord[0], vehicle_coord[1] - 1)
            elif action == 2:
                action_direction = (vehicle_coord[0], vehicle_coord[1] + 1)
            else:
                action_direction = (vehicle_coord[0] + 1, vehicle_coord[1])
        else:  # elif self.ac_dim == 8:
            if action == 0:
                action_direction = (vehicle_coord[0] - 1, vehicle_coord[1])
            elif action == 1:
                action_direction = (vehicle_coord[0] - 1, vehicle_coord[1] + 1)
            elif action == 2:
                action_direction = (vehicle_coord[0], vehicle_coord[1] + 1)
            elif action == 3:
                action_direction = (vehicle_coord[0] + 1, vehicle_coord[1] + 1)
            elif action == 4:
                action_direction = (vehicle_coord[0] + 1, vehicle_coord[1])
            elif action == 5:
                action_direction = (vehicle_coord[0] + 1, vehicle_coord[1] - 1)
            elif action == 6:
                action_direction = (vehicle_coord[0], vehicle_coord[1] - 1)
            else:
                action_direction = (vehicle_coord[0] - 1, vehicle_coord[1] - 1)

        angle_list = []
        for path in arrive_path:
            destination_loc = path[-1]
            destination_coord = self.node.loc[destination_loc, ['node_coordinate_row', 'node_coordinate_col']].values.astype(int)
            angle_list.append(self.cal_angle(list(action_direction), list(vehicle_coord), list(destination_coord)))
        index = angle_list.index(min(angle_list))

        return arrive_path[index]

    def execute_action(self, left_time, episode_time_cost):
        '''

        :param left_time:
        :param episode_time_cost:
        :return:
        '''

        done = False

        action_interval = self.action_interval
        if action_interval >= left_time:
            action_interval = left_time
            done = True

        for path_i, path in enumerate(self.vehicle_action_paths):
            path_index = 1
            remaining_time = action_interval
            remaining_time -= self.vehicle_states[path_i, 2]
            starting_node = self.vehicle_states[path_i, 0]
            ending_node = self.vehicle_states[path_i, 1]
            while remaining_time > 0:
                self.node.loc[ending_node, 'node_cover'] += 1
                starting_node = ending_node
                ending_node = int(path[path_index])
                remaining_time -= self.experienced_travel_time[starting_node, ending_node]
                path_index += 1
            self.vehicle_states[path_i, 2] = -remaining_time
            self.vehicle_states[path_i, 1] = ending_node
            if remaining_time == 0:
                self.node.loc[ending_node, 'node_vehicle_id_cover{}'.format(path_i)] = 1
                self.vehicle_states[path_i, 0] = self.vehicle_states[path_i, 1].copy()
            else:
                self.vehicle_states[path_i, 0] = starting_node

        self.vehicle_action_paths = [- 10000] * self.vehicle_num

        self.past_time += action_interval

        episode_time_cost += self.action_interval

        if self.past_time % self.interval_to_cal_reward == 0:
            self.grid[self.grid_vehicle_id_cover_name] = \
                self.grid_vehicle_id_cover_count()[self.node_vehicle_id_cover_name].values
            self.p_hat_statistic()
            self.map_grid_info_to_node(add_col_names=['p_hat'])
            reward = self.cal_reward()
            vehicle_loc = np.zeros((len(self.node), self.vehicle_num))
            vehicle_loc[(self.vehicle_states[:, 0], range(self.vehicle_num))] = 1
            self.node[self.node_vehicle_id_cover_name] = vehicle_loc
            self.grid[self.grid_vehicle_id_cover_name] = \
                self.grid_vehicle_id_cover_count()[self.node_vehicle_id_cover_name].values
            self.map_grid_vehicle_id_cover_to_node()
            self.cal_reward_time += 1
        else:
            self.grid[self.grid_vehicle_id_cover_name] = \
                self.grid_vehicle_id_cover_count()[self.node_vehicle_id_cover_name].values
            self.map_grid_vehicle_id_cover_to_node()
            reward = 0

        node_weight = self.node['weight'].values
        vehicle_states = copy.deepcopy(self.vehicle_states)
        grid_vehicle_id_cover = self.node[self.grid_vehicle_id_cover_name].values
        p_hat = self.node['p_hat'].values

        return vehicle_states, node_weight, grid_vehicle_id_cover, p_hat, reward, done, episode_time_cost

    def step(self, ac_dict: dict, episode_time_cost):
        '''
        Env receives all agents' action and make one timestep forward
        :param ac_dict: dict, key allowed in list(range(self.vehicle_num)), key value allowed 0, 1, 2, 3 or 0 - 7
        :param episode_time_cost:
        :return:
        '''

        left_time = self.episode_duration - self.past_time
        assert left_time > 0, 'you need reset the environment first'

        for i in range(self.vehicle_num):
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=ac_dict[i]
            )
            self.vehicle_action_paths[i] = path

        vehicle_states, node_weight, grid_vehicle_id_cover, p_hat, reward, done, episode_time_cost = \
            self.execute_action(
                left_time=left_time,
                episode_time_cost=episode_time_cost,
            )

        return vehicle_states, node_weight, grid_vehicle_id_cover, p_hat, reward, done, episode_time_cost


# class generate_asynchronous_timestep_environment():
#     '''
#     generate environment with asynchronous timestep for multiple agents
#     '''
#
#     def __init__(
#             self, experienced_travel_time: np.ndarray, node_id_to_grid_id: pd.DataFrame,
#             ts_duration: Union[int, float], num_of_ts: int, BETA: Union[int, float],
#             vehicle_num: int, seed: int,
#     ):
#         '''
#
#         :param experienced_travel_time:
#         :param node_id_to_grid_id: need contain column 'grid_id', 'node_id'.
#         :param ts_duration: timestep unit is second
#         :param num_of_ts: number of timestep
#         :param BETA:
#         :param vehicle_num:
#         :param seed:
#         :return:
#         '''
#
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#
#         self.experienced_travel_time = experienced_travel_time
#         self.node = node_id_to_grid_id[['node_id', 'grid_id']]
#         self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
#         self.node['node_cover'] = 0
#         self.grid = pd.DataFrame({'grid_id': self.node['grid_id'].unique()})
#         self.grid.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
#         self.node_id_min = self.node['node_id'].min()
#         self.node_id_max = self.node['node_id'].max()
#
#         self.ts_duration = ts_duration
#         self.num_of_ts = num_of_ts
#         self.past_time = 0
#         self.ts = 0
#         self.episode_duration = self.ts_duration * self.num_of_ts
#         self.BETA = BETA
#         self.vehicle_num = vehicle_num
#         self.vehicle_states = []
#         for i in range(self.vehicle_num):
#             self.vehicle_states.append([
#                 -1,  # location1
#                 -1,  # location2
#                 0,  # remaining_time
#             ])
#         self.vehicle_states = np.array(self.vehicle_states)
#         # calculate node allowed action
#         self.node_allowed_action = {}
#         self.max_actions_num = 0
#         for i in range(self.experienced_travel_time.shape[0]):
#             self.node_allowed_action[i] = np.where(self.experienced_travel_time[i] < np.inf)[0]
#             if self.max_actions_num < len(self.node_allowed_action[i]):
#                 self.max_actions_num = len(self.node_allowed_action[i])
#
#         self.grid_cover_col_names = (pd.Series(['grid_cover'] * self.num_of_ts) +
#                                      pd.Series(range(self.num_of_ts)).astype(str)).tolist()
#         self.p_col_names = (pd.Series(['p'] * self.num_of_ts) + pd.Series(range(self.num_of_ts)).astype(str)).tolist()
#
#     def grid_cover_count(self):
#         grid_cover = self.node.groupby(['grid_id'], as_index=False)[['node_cover']].sum()
#         grid_cover.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
#         return grid_cover
#
#     def map_grid_cover_to_node(self):
#         if 'grid_cover' in self.node.columns.values:
#             del self.node['grid_cover']
#         self.node = pd.merge(
#             self.node, self.grid[['grid_id', 'grid_cover']], on=['grid_id'])
#         self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
#
#     def ts_grid_cover_statistic(self):
#         self.grid['grid_cover{}'.format(self.ts)] = self.grid['grid_cover'].values
#         grid_cover_or_not = copy.deepcopy(self.grid['grid_cover'].values)
#         grid_cover_or_not[np.where(grid_cover_or_not > 0)] = 1
#         if self.ts > 0:
#             self.grid['ts_grid_cover{}'.format(self.ts)] = self.grid[
#                 'ts_grid_cover{}'.format(self.ts - 1)].values + grid_cover_or_not
#         else:
#             self.grid['ts_grid_cover{}'.format(self.ts)] = grid_cover_or_not
#         p = self.grid['ts_grid_cover{}'.format(self.ts)].values / (self.ts + 1)
#         p[np.where(p == 0)] = self.BETA / (self.ts + 1)
#         self.grid['p'] = p
#         self.grid['p{}'.format(self.ts + 1)] = p
#
#     def reset_grid_weight(self, grid_weight: Union[pd.DataFrame, None]):
#         if grid_weight is not None:
#             self.grid = grid_weight
#         else:
#             self.grid['weight'] = np.random.dirichlet(
#                 np.ones(shape=len(self.grid), dtype=np.int), size=1).flatten()
#
#     def map_grid_info_to_node(self):
#         if 'grid_cover' in self.node.columns.values:
#             del self.node['grid_cover']
#         if 'weight' in self.node.columns.values:
#             del self.node['weight']
#         if 'p' in self.node.columns.values:
#             del self.node['p']
#         self.node = pd.merge(
#             self.node, self.grid[['grid_id', 'grid_cover', 'weight', 'p']], on=['grid_id'])
#         self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
#
#     def cal_reward(self):
#         weight = self.grid['weight'].values
#         weight = np.repeat(weight, self.num_of_ts).reshape((len(weight), self.num_of_ts))
#         grid_cover = self.grid[self.grid_cover_col_names].values
#         p = self.grid[self.p_col_names].values
#         nn0_where = np.where(grid_cover != 0)
#         n0_where = np.where(grid_cover == 0)
#         reward = np.zeros_like(grid_cover, dtype=np.float32)
#         reward[nn0_where] = - weight[nn0_where] * grid_cover[nn0_where] * np.log(
#             1 - (1 - p[nn0_where]) ** grid_cover[nn0_where])
#         reward[n0_where] = weight[n0_where] * np.log(p[n0_where])
#         return reward.sum()
#
#     def reset(self, grid_weight):
#         '''
#         reset the agents(vehicles) position and grid link_weight(rewards)
#         :return:
#         '''
#         self.past_time = 0
#         self.ts = 0
#         coordinate = np.random.randint(low=self.node_id_min, high=self.node_id_max + 1, size=self.vehicle_num)
#         self.vehicle_states[:, 1] = coordinate  # location2
#         self.vehicle_states[:, 0] = self.vehicle_states[:, 1].copy()  # location1
#         self.vehicle_states[:, 2] = 0  # remaining_time
#         self.reset_grid_weight(
#             grid_weight=grid_weight,
#         )
#         self.node['node_cover'] = 0
#         self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
#         self.ts_grid_cover_statistic()
#         self.grid['p0'] = self.grid['p'].values
#         self.map_grid_info_to_node()
#
#         node_weight = self.node['weight'].values
#         vehicle_states = copy.deepcopy(self.vehicle_states)
#         grid_cover = self.node['grid_cover'].values
#         p = self.node['p'].values
#         need_move = list(range(self.vehicle_num))
#
#         return vehicle_states, node_weight, grid_cover, p, need_move
#
#     def execute_action(self, actions: dict, episode_time_cost):
#         '''
#
#         :param actions:
#         :param episode_time_cost:
#         :return:
#         '''
#
#         done = False
#         for vehicle_id in actions.keys():
#             action = actions[vehicle_id]
#             vehicle_origin = self.vehicle_states[vehicle_id, 0]
#             self.vehicle_states[vehicle_id, 1] = action
#             vehicle_destination = self.vehicle_states[vehicle_id, 1]
#             self.vehicle_states[vehicle_id, 2] = self.experienced_travel_time[vehicle_origin, vehicle_destination]
#
#         min_time_to_next_node = self.vehicle_states[:, 2].min()
#         ts_left_time = (self.ts + 1) * self.ts_duration - self.past_time
#
#         while min_time_to_next_node >= ts_left_time:
#
#             self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
#             self.ts_grid_cover_statistic()
#             self.map_grid_info_to_node()
#             self.node['node_cover'] = 0
#             self.ts += 1
#
#             self.past_time += ts_left_time
#             self.vehicle_states[:, 2] = self.vehicle_states[:, 2] - ts_left_time
#             episode_time_cost += ts_left_time
#
#             ts_left_time = (self.ts + 1) * self.ts_duration - self.past_time
#             min_time_to_next_node = self.vehicle_states[:, 2].min()
#
#             if self.ts == self.num_of_ts:
#                 done = True
#                 break
#
#         if min_time_to_next_node > 0:
#             self.past_time += min_time_to_next_node
#             self.vehicle_states[:, 2] = self.vehicle_states[:, 2] - min_time_to_next_node
#             episode_time_cost += min_time_to_next_node
#
#         need_move = np.where(self.vehicle_states[:, 2] == 0)[0].tolist()
#         self.vehicle_states[(need_move, [0] * len(need_move))] = self.vehicle_states[(need_move, [1] * len(need_move))]
#         arrival_loc = self.vehicle_states[(need_move, [0] * len(need_move))].tolist()
#         if len(set(arrival_loc)) == len(arrival_loc):
#             self.node.loc[arrival_loc, 'node_cover'] += 1
#         else:
#             for loc in arrival_loc:
#                 self.node.loc[loc, 'node_cover'] += 1
#         self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
#         self.map_grid_cover_to_node()
#
#         node_weight = self.node['weight'].values
#         vehicle_states = copy.deepcopy(self.vehicle_states)
#         grid_cover = self.node['grid_cover'].values
#         p = self.node['p'].values
#
#         if done:
#             reward = self.cal_reward()
#         else:
#             reward = 0
#
#         return vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost
#
#     def step_by_action_probs(
#             self, ac_probs_dict: dict, episode_time_cost):
#         '''
#         Env receives all agents' action probability, sampling from it, and make one timestep forward
#         :param ac_probs_dict: dict, key allowed in list(range(self.vehicle_num)), key value are torch.tensor like
#         [0.2512, 0.2487, 0.2500, 0.2501]
#         :param episode_time_cost:
#         :return:
#         '''
#
#         left_time = self.episode_duration - self.past_time
#         assert left_time > 0, 'you need reset the environment first'
#
#         actions = {}
#         for i in ac_probs_dict.keys():
#             ac_allowed = self.node_allowed_action[self.vehicle_states[i, 1]]
#             actions[i] = ac_allowed[torch.multinomial(ac_probs_dict[i][ac_allowed], num_samples=1).item()]
#
#         vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = self.execute_action(
#             actions=actions,
#             episode_time_cost=episode_time_cost,
#         )
#
#         return actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost


if __name__ == '__main__':

    # generate network
    seed = 4000
    height = 20
    width = 20
    experienced_travel_time, node_id_to_grid_id = generate_rectangle_network(
        height=height,
        width=width,
        low_second=30,
        high_second=300,
        per_grid_height=2,
        per_grid_width=2,
        seed=seed,
    )

    # # initial asynchronous env
    # ts_duration = 600
    # num_of_ts = 2
    # BETA = 0.5
    # vehicle_num = 2
    # env = generate_asynchronous_timestep_environment(
    #     experienced_travel_time=experienced_travel_time,
    #     node_id_to_grid_id=node_id_to_grid_id,
    #     ts_duration=ts_duration,
    #     num_of_ts=num_of_ts,
    #     BETA=BETA,
    #     vehicle_num=vehicle_num,
    #     seed=seed,
    # )
    # # reset env
    # grid_weight = None
    # vehicle_states, node_weight, grid_cover, p, need_move = env.reset(grid_weight=grid_weight)
    # # env forward
    # ac_probs_dict = {}
    # for vehicle_id in range(vehicle_num):
    #     ac_probs_dict[vehicle_id] = torch.tensor([1 / (height * width)] * (height * width))
    # st = time.time()
    # for episode in range(100):
    #     done = False
    #     while not done:
    #         ac_probs_dict_ = {}
    #         for v_id in need_move:
    #             ac_probs_dict_[v_id] = ac_probs_dict[v_id]
    #         actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = \
    #             env.step_by_action_probs(
    #                 ac_probs_dict=ac_probs_dict_,
    #                 episode_time_cost=0,
    #             )
    #     vehicle_states, node_weight, grid_cover, p, need_move = env.reset(grid_weight=grid_weight)
    # et = time.time()
    # print(et - st)

    # initial synchronous env
    ac_dim = 8
    action_interval = 180
    num_of_action_interval = 4
    num_of_cal_reward = 5
    epsilon = 0.5
    vehicle_num = 2
    env = generate_synchronous_timestep_environment_with_directional_action(
        experienced_travel_time=experienced_travel_time,
        node_id_to_grid_id=node_id_to_grid_id,
        ac_dim=ac_dim,
        action_interval=action_interval,
        num_of_action_interval=num_of_action_interval,
        num_of_cal_reward=num_of_cal_reward,
        epsilon=epsilon,
        vehicle_num=vehicle_num,
        seed=seed,
    )
    # reset env
    grid_weight = None
    vehicle_states, node_weight, grid_vehicle_id_cover, p_hat = env.reset(grid_weight=grid_weight)
    # env forward
    ac_probs_dict = {}
    for vehicle_id in range(vehicle_num):
        ac_probs_dict[vehicle_id] = torch.tensor([1 / 8] * 8)
    st = time.time()
    for episode in range(100):
        done = False
        while not done:
            ac_dict = {}
            for v_id in range(vehicle_num):
                ac_dict[v_id] = torch.multinomial(ac_probs_dict[v_id], num_samples=1).item()
            vehicle_states, node_weight, grid_vehicle_id_cover, p_hat, reward, done, episode_time_cost = env.step(
                    ac_dict=ac_dict,
                    episode_time_cost=0,
                )
            # print(np.where(grid_vehicle_id_cover != 0))
            # print(p_hat)
            # print(reward)
        print('*********************************************************')
        vehicle_states, node_weight, grid_vehicle_id_cover, p_hat = env.reset(grid_weight=grid_weight)
    et = time.time()
    print(et - st)