'''
reinforcement learning simulation environment
'''

import sys
import os
import math
import time
import copy
import random
import pickle
from typing import Union

import pandas as pd
import torch

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from grid_score_base_project.simulation_environment.Shortest_path import *


# class generate_environment_with_grid_score_directional_action_edge_base_features():
#     '''
#     generate environment with synchronous timestep and directional action for multiple agents
#     '''
#
#     def __init__(
#             self,
#             vehicle_speed: float,
#             adjacency_matrix_with_edge_id: np.ndarray,
#             edge_info: pd.DataFrame,
#             endpoints_of_each_edge: pd.DataFrame,
#             edge_id_to_grid_id: pd.DataFrame,
#             node_info: pd.DataFrame,
#             grid_weight: pd.DataFrame,
#             taxi_cover_times: Union[pd.DataFrame, None],
#             ac_dim: int,
#             action_interval: int,
#             T: int,
#             vehicle_num: int,
#             seed: int,
#     ):
#         '''
#
#         :param vehicle_speed:
#         :param adjacency_matrix_with_edge_id:
#         :param edge_info:
#         :param endpoints_of_each_edge:
#         :param edge_id_to_grid_id: need contain column 'grid_id', 'edge_id'.
#         :param node_info: need contain column 'node_id', 'node_coordinate_x', 'node_coordinate_y'.
#         :param grid_weight: need contain column 'grid_id', 'grid_weight'.
#         :param taxi_cover_times: need contain column 'grid_id', '9', '10', '11', '12', '13', '14'.
#         :param ac_dim: the dimension of action, only support 4 or 8
#         :param action_interval: seconds, every timestep's time interval
#         :param T: one episode has T timesteps
#         :param vehicle_num:
#         :param seed:
#         :return:
#         '''
#
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#
#         assert ac_dim == 4 or ac_dim == 8, '4 and 8 are ac_dim allowed value.'
#
#         self.adjacency_matrix_with_edge_id = adjacency_matrix_with_edge_id
#         edge_info['length'] /= vehicle_speed
#         self.experienced_travel_time = np.ones_like(self.adjacency_matrix_with_edge_id, dtype=np.float32)
#         self.experienced_travel_time[:] = np.inf
#         for row in range(self.adjacency_matrix_with_edge_id.shape[0]):
#             for col in range(self.adjacency_matrix_with_edge_id.shape[1]):
#                 edge_id = self.adjacency_matrix_with_edge_id[row, col]
#                 if edge_id != -1:
#                     self.experienced_travel_time[row, col] = edge_info.loc[edge_id, 'length']
#         self.max_experienced_travel_time = self.experienced_travel_time.max()
#         self.node = node_info[['node_id', 'node_coordinate_x', 'node_coordinate_y']]
#         self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
#         self.edge_id_to_grid_id = edge_id_to_grid_id
#         self.edge_id_to_grid_id.sort_values(by=['edge_id', 'grid_id'], inplace=True, ignore_index=True)
#         self.edge = pd.DataFrame({'edge_id': self.edge_id_to_grid_id['edge_id'].unique()})
#         self.edge.sort_values(by=['edge_id'], inplace=True, ignore_index=True)
#         self.edge['cover_times'] = 0
#         self.edge['t_vi_cover_times'] = 0
#         self.edge['edge_score'] = 0
#         self.grid = pd.DataFrame({'grid_id': self.edge_id_to_grid_id['grid_id'].unique()})
#         self.grid = pd.merge(self.grid, grid_weight[['grid_id', 'grid_weight']], how='left', on=['grid_id'])
#         self.grid['grid_weight'] /= self.grid['grid_weight'].sum()
#         self.grid.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
#         if taxi_cover_times is not None:
#             self.taxi_cover_times = pd.merge(taxi_cover_times, self.grid[['grid_id']], how='right', on=['grid_id'])
#             self.taxi_cover_times.fillna(value=0, inplace=True)
#             self.taxi_cover_times.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
#         else:
#             self.taxi_cover_times = None
#         self.node_id_min = self.node['node_id'].min()
#         self.node_id_max = self.node['node_id'].max()
#         self.ac_dim = ac_dim
#         self.action_interval = action_interval
#         self.T = T
#         self.past_time = 0
#         self.episode_duration = self.action_interval * self.T
#         self.time_of_reset_cover = 1
#         self.vehicle_num = vehicle_num
#         self.vehicle_states = []
#         self.vehicle_action_paths = []
#         for i in range(self.vehicle_num):
#             self.vehicle_states.append([
#                 -1,  # location1
#                 -1,  # location2
#                 0,  # remaining_time
#             ])
#             self.vehicle_action_paths.append(- 10000)
#         self.vehicle_states = np.array(self.vehicle_states)
#         self.got_reward = 0
#
#         # Invariant static feature
#         ivar_feat = endpoints_of_each_edge
#         # add endpoints coordinates of each edge
#         for end_point_name in ['one', 'two']:
#             ivar_feat.rename(columns={'node_id_{}'.format(end_point_name): 'node_id'}, inplace=True)
#             ivar_feat = pd.merge(ivar_feat, node_info, on=['node_id'])
#             del ivar_feat['node_id']
#         self.ivar_feat = pd.merge(ivar_feat, edge_info, on=['edge_id'])
#         self.ivar_feat.sort_values(by=['edge_id'], inplace=True, ignore_index=True)
#         self.ivar_feat.drop(columns=['edge_id'], inplace=True)
#         self.ivar_feat = self.ivar_feat.astype(np.float32)
#
#     def t_grid_cover_times_count(self):
#         grid_cover = pd.merge(self.edge, self.edge_id_to_grid_id, on=['edge_id'])
#         grid_cover = grid_cover.groupby(['grid_id'], as_index=False)[['cover_times', 't_vi_cover_times']].sum()
#         grid_cover.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
#         if self.taxi_cover_times is not None:
#             grid_cover['cover_times'] += self.taxi_cover_times['{}'.format(self.time_of_reset_cover + 8)].values
#         return grid_cover
#
#     def cal_grid_score(self, grid_cover):
#         self.grid['grid_score'] = ((grid_cover['cover_times'] + 1) ** 0.5 - grid_cover[
#             'cover_times'] ** 0.5).values * self.grid['grid_weight'].values
#
#     def cal_reward(self, grid_cover):
#         reward = (grid_cover['cover_times'] + grid_cover['t_vi_cover_times']) ** 0.5 - grid_cover['cover_times'] ** 0.5
#         reward = (reward.values * self.grid['grid_weight'].values).sum()
#         return reward
#
#     def map_grid_score_to_edge(self):
#         '''
#
#         :return:
#         '''
#         merge_result = pd.merge(
#             self.edge_id_to_grid_id[['edge_id', 'grid_id']], self.grid[['grid_id', 'grid_score']], on=['grid_id'])
#         edge_score = merge_result.groupby(by=['edge_id'], as_index=False)['grid_score'].sum()
#         edge_score.sort_values(by=['edge_id'], inplace=True, ignore_index=True)
#         self.edge['edge_score'] = edge_score['grid_score'].values.astype(np.float32)
#
#     def output_state_features(self):
#         '''
#         '''
#
#         edge_loc_features = np.zeros((self.vehicle_num, len(self.edge)), dtype=np.float32)
#         edge_all_v_loc_features = np.zeros((len(self.edge),), dtype=np.float32)
#         vehicle_node_loc = self.vehicle_states[:, : 2]
#         where = np.where(vehicle_node_loc[:, 0] == vehicle_node_loc[:, 1])
#         edge_id = []
#         if len(where[0]) > 0:
#             for v_i in range(len(vehicle_node_loc)):
#                 if v_i not in where[0]:
#                     v_i_loc = self.adjacency_matrix_with_edge_id[vehicle_node_loc[v_i, 0], vehicle_node_loc[v_i, 1]]
#                     edge_loc_features[v_i, v_i_loc] = 1
#                     edge_id.append(v_i_loc)
#                 else:
#                     v_i_loc = self.adjacency_matrix_with_edge_id[vehicle_node_loc[v_i, 0]][
#                         np.where(self.adjacency_matrix_with_edge_id[vehicle_node_loc[v_i, 0]] != -1)].tolist()
#                     edge_loc_features[([v_i] * len(v_i_loc), v_i_loc)] = 1
#                     edge_id += v_i_loc
#         else:
#             edge_id = self.adjacency_matrix_with_edge_id[(vehicle_node_loc[:, 0], vehicle_node_loc[:, 1])]
#             edge_loc_features[(list(range(self.vehicle_num)), edge_id)] = 1
#         edge_id = tuple(edge_id)
#         edge_all_v_loc_features[(edge_id, )] = 1
#
#         grid_cover = self.t_grid_cover_times_count()
#         self.cal_grid_score(grid_cover)
#         self.map_grid_score_to_edge()
#
#         the_same_features = np.concatenate((
#             edge_all_v_loc_features.reshape((edge_all_v_loc_features.shape[0], 1)),
#             self.ivar_feat.values,
#             self.edge['edge_score'].values.reshape((len(self.edge), 1))), axis=1)
#
#         return edge_loc_features, the_same_features
#
#     def reset(self):
#         '''
#         reset the agents(vehicles) position and grid link_score(rewards)
#         :return:
#         '''
#         self.past_time = 0
#         loc = np.random.randint(low=self.node_id_min, high=self.node_id_max + 1, size=self.vehicle_num)
#         self.vehicle_states[:, 1] = loc  # location2
#         self.vehicle_states[:, 0] = self.vehicle_states[:, 1].copy()  # location1
#         self.vehicle_states[:, 2] = 0  # remaining_time
#         self.time_of_reset_cover = 1
#         self.edge['cover_times'] = 0
#         self.edge['t_vi_cover_times'] = 0
#         self.vehicle_action_paths = [- 10000] * self.vehicle_num
#
#         edge_loc_features, the_same_features = self.output_state_features()
#
#         self.got_reward = 0
#
#         return edge_loc_features, the_same_features
#
#     def cal_angle(self, point_1: Union[tuple, list], point_2: Union[tuple, list], point_3: Union[tuple, list]):
#         """
#         calculating the Angle between the vertices at point_2
#         :param point_1:
#         :param point_2: Tuple, the coordinate of point_2
#         :param point_3:
#         :return: Float, the Angle between the vertices at point_2
#         """
#
#         a = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (
#                 point_2[1] - point_3[1]))
#         b = math.sqrt((point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (
#                 point_1[1] - point_3[1]))
#         c = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
#                 point_1[1] - point_2[1]))
#         # A = math.degrees(math.acos((a * a - b * b - c * c)/(-2 * b * c)))
#         middle_cal_result = round((b * b - a * a - c * c) / (-2 * a * c), 4)
#         B = math.degrees(math.acos(middle_cal_result))
#         # C = math.degrees(math.acos((c * c - a * a - b * b)/(-2 * a * b)))
#
#         return B
#
#     def determine_path(self, vehicle_state: Union[tuple, list, np.ndarray], action: int) -> list:
#         '''
#         deciding the final node destination
#         :param vehicle_state: agent's state
#         :param action: East, south, west, north four directions or plus
#         northwest, northeast, southeast, southwest 8 directions totally
#         :return:
#         '''
#
#         vehicle_loc = vehicle_state[1]
#         remaining_time = vehicle_state[2]
#
#         arrive_path = []
#         start = vehicle_loc
#         start_time = self.action_interval - remaining_time
#         if start_time < 0:
#             start_time = 0
#         came_from, node_list = dijkstra_search(
#             cost=self.experienced_travel_time,
#             start=start,
#             start_time=start_time,
#             end_time=start_time + self.max_experienced_travel_time + 60,
#             node_length=self.ac_dim,
#         )
#         for node in node_list:
#             path = reconstruct_path(
#                 came_from=came_from,
#                 start=start,
#                 goal=node,
#             )
#             arrive_path.append(path)
#
#         # calculate the Angle between the action vector and the destination vector,
#         # and choose destination with the smallest Angle as final action destination
#         vehicle_coord = self.node.loc[vehicle_loc, ['node_coordinate_x', 'node_coordinate_y']].values.astype(int)
#         if self.ac_dim == 4:
#             if action == 0:
#                 action_direction = (vehicle_coord[0] - 1, vehicle_coord[1])
#             elif action == 1:
#                 action_direction = (vehicle_coord[0], vehicle_coord[1] - 1)
#             elif action == 2:
#                 action_direction = (vehicle_coord[0], vehicle_coord[1] + 1)
#             else:
#                 action_direction = (vehicle_coord[0] + 1, vehicle_coord[1])
#         else:  # elif self.ac_dim == 8:
#             if action == 0:
#                 action_direction = (vehicle_coord[0] - 1, vehicle_coord[1])
#             elif action == 1:
#                 action_direction = (vehicle_coord[0] - 1, vehicle_coord[1] + 1)
#             elif action == 2:
#                 action_direction = (vehicle_coord[0], vehicle_coord[1] + 1)
#             elif action == 3:
#                 action_direction = (vehicle_coord[0] + 1, vehicle_coord[1] + 1)
#             elif action == 4:
#                 action_direction = (vehicle_coord[0] + 1, vehicle_coord[1])
#             elif action == 5:
#                 action_direction = (vehicle_coord[0] + 1, vehicle_coord[1] - 1)
#             elif action == 6:
#                 action_direction = (vehicle_coord[0], vehicle_coord[1] - 1)
#             else:
#                 action_direction = (vehicle_coord[0] - 1, vehicle_coord[1] - 1)
#
#         angle_list = []
#         for path in arrive_path:
#             destination_loc = path[-1]
#             destination_coord = self.node.loc[destination_loc, ['node_coordinate_x', 'node_coordinate_y']].values.astype(int)
#             angle_list.append(self.cal_angle(list(action_direction), list(vehicle_coord), list(destination_coord)))
#         index = angle_list.index(min(angle_list))
#
#         return arrive_path[index]
#
#     def execute_action(self, left_time, episode_time_cost):
#         '''
#
#         :param left_time:
#         :param episode_time_cost:
#         :return:
#         '''
#
#         done = False
#
#         action_interval = self.action_interval
#         if action_interval >= left_time:
#             action_interval = left_time
#             done = True
#
#         for path_i, path in enumerate(self.vehicle_action_paths):
#             path_index = 1
#             remaining_time = action_interval
#             remaining_time -= self.vehicle_states[path_i, 2]
#             starting_node = self.vehicle_states[path_i, 0]
#             ending_node = self.vehicle_states[path_i, 1]
#             while remaining_time > 0:
#                 if starting_node != ending_node:
#                     covered_edge_id = self.adjacency_matrix_with_edge_id[starting_node, ending_node]
#                     self.edge.loc[covered_edge_id, 't_vi_cover_times'] += 1
#                 starting_node = ending_node
#                 ending_node = int(path[path_index])
#                 remaining_time -= self.experienced_travel_time[starting_node, ending_node]
#                 path_index += 1
#             self.vehicle_states[path_i, 2] = -remaining_time
#             self.vehicle_states[path_i, 1] = ending_node
#             if remaining_time == 0:
#                 covered_edge_id = self.adjacency_matrix_with_edge_id[starting_node, ending_node]
#                 self.edge.loc[covered_edge_id, 't_vi_cover_times'] += 1
#                 self.vehicle_states[path_i, 0] = self.vehicle_states[path_i, 1].copy()
#             else:
#                 self.vehicle_states[path_i, 0] = starting_node
#
#         grid_cover = self.t_grid_cover_times_count()
#         cooperative_reward = self.cal_reward(grid_cover)
#         reward = [cooperative_reward] * self.vehicle_num
#         self.got_reward += cooperative_reward
#         self.edge['cover_times'] += self.edge['t_vi_cover_times']
#         self.edge['t_vi_cover_times'] = 0
#
#         self.vehicle_action_paths = [- 10000] * self.vehicle_num
#         self.past_time += action_interval
#         episode_time_cost += self.action_interval
#
#         edge_loc_features, the_same_features = self.output_state_features()
#
#         return edge_loc_features, the_same_features, reward, done, episode_time_cost
#
#     def step(self, ac_dict: dict, episode_time_cost):
#         '''
#         Env receives all agents' action and make one timestep forward
#         :param ac_dict: dict, key allowed in list(range(self.vehicle_num)), key value allowed 0, 1, 2, 3 or 0 - 7
#         :param episode_time_cost:
#         :return:
#         '''
#
#         left_time = self.episode_duration - self.past_time
#
#         # reset cover times(score)
#         if self.past_time >= self.time_of_reset_cover * 3600:
#             self.edge['cover_times'] = 0
#             self.edge['t_vi_cover_times'] = 0
#             self.time_of_reset_cover += 1
#         assert left_time > 0, 'you need reset the environment first'
#
#         for i in range(self.vehicle_num):
#             path = self.determine_path(
#                 vehicle_state=self.vehicle_states[i],
#                 action=ac_dict[i]
#             )
#             self.vehicle_action_paths[i] = path
#
#         edge_loc_features, the_same_features, reward, done, episode_time_cost = \
#             self.execute_action(
#                 left_time=left_time,
#                 episode_time_cost=episode_time_cost,
#             )
#
#         return edge_loc_features, the_same_features, reward, done, episode_time_cost


class generate_environment_with_grid_score_directional_action_node_base_features():
    '''
    generate environment with synchronous timestep and directional action for multiple agents
    '''

    def __init__(
        self,
        vehicle_speed: float,
        adjacency_matrix_with_edge_id: np.ndarray,
        edge_info: pd.DataFrame,
        node_info: pd.DataFrame,
        grid_weight: pd.DataFrame,
        taxi_cover_times: Union[pd.DataFrame, None],
        ac_dim: int,
        action_interval: int,
        T: int,
        vehicle_num: int,
        seed: int,
    ):
        '''

        :param vehicle_speed:
        :param adjacency_matrix_with_edge_id:
        :param edge_info:
        :param node_info: need contain column 'grid_id', 'node_id', 'node_coordinate_x', 'node_coordinate_y'.
        :param grid_weight: need contain column 'grid_id', 'grid_weight'.
        :param taxi_cover_times: need contain column 'grid_id', '9', '10', '11', '12', '13', '14'.
        :param ac_dim: the dimension of action, only support 4 or 8
        :param action_interval: seconds, every timestep's time interval
        :param T: one episode has T timesteps
        :param vehicle_num:
        :param seed:
        :return:
        '''

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        assert ac_dim == 4 or ac_dim == 8, '4 and 8 are ac_dim allowed value.'

        self.adjacency_matrix_with_edge_id = adjacency_matrix_with_edge_id
        edge_info['length'] /= vehicle_speed
        edge_info.sort_values(by=['edge_id'], inplace=True, ignore_index=True)
        self.experienced_travel_time = np.ones_like(self.adjacency_matrix_with_edge_id, dtype=np.float32)
        self.experienced_travel_time[:] = np.inf
        for row in range(self.adjacency_matrix_with_edge_id.shape[0]):
            for col in range(self.adjacency_matrix_with_edge_id.shape[1]):
                edge_id = self.adjacency_matrix_with_edge_id[row, col]
                if edge_id != -1:
                    self.experienced_travel_time[row, col] = edge_info.loc[edge_id, 'length']
        self.max_experienced_travel_time = self.experienced_travel_time.max()
        self.node = node_info[['node_id', 'grid_id', 'node_coordinate_x', 'node_coordinate_y']]
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
        self.node['node_coordinate_x'] = self.node['node_coordinate_x'].astype(np.float32).values
        self.node['node_coordinate_y'] = self.node['node_coordinate_y'].astype(np.float32).values
        self.node['node_coordinate_x'] += self.node['node_coordinate_x'].min() + 1
        self.node['node_coordinate_y'] += self.node['node_coordinate_y'].min() + 1
        self.node['cover_times'] = 0
        self.node['t_vi_cover_times'] = 0
        self.node['node_score'] = 0

        self.static_features = copy.deepcopy(self.node[['node_coordinate_x', 'node_coordinate_y']].values)
        self.static_features[:, 0] = (self.static_features[:, 0] - self.static_features[:, 0].min()) / (
            self.static_features[:, 0].max() - self.static_features[:, 0].min()
        )
        self.static_features[:, 1] = (self.static_features[:, 1] - self.static_features[:, 1].min()) / (
                self.static_features[:, 1].max() - self.static_features[:, 1].min()
        )

        self.grid = pd.DataFrame({'grid_id': self.node['grid_id'].unique()})
        self.grid = pd.merge(self.grid, grid_weight[['grid_id', 'grid_weight']], how='left', on=['grid_id'])
        self.grid['grid_weight'] /= self.grid['grid_weight'].sum()
        self.grid.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        self.grid['grid_weight'] += 0.01
        if taxi_cover_times is not None:
            self.taxi_cover_times = pd.merge(taxi_cover_times, self.grid[['grid_id']], how='right', on=['grid_id'])
            self.taxi_cover_times.fillna(value=0, inplace=True)
            self.taxi_cover_times.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        else:
            self.taxi_cover_times = None
        self.node_id_min = self.node['node_id'].min()
        self.node_id_max = self.node['node_id'].max()
        self.ac_dim = ac_dim
        self.action_interval = action_interval
        self.T = T
        self.past_time = 0
        self.episode_duration = self.action_interval * self.T
        self.time_of_reset_cover = 1
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        self.vehicle_action_paths = []
        for i in range(self.vehicle_num):
            self.vehicle_states.append([
                -1,  # location1
                -1,  # location2
                0,  # remaining_time
            ])
            self.vehicle_action_paths.append(- 10000)
        self.vehicle_states = np.array(self.vehicle_states)
        self.got_reward = 0

    def t_grid_cover_times_count(self):
        grid_cover = self.node.groupby(['grid_id'], as_index=False)[['cover_times', 't_vi_cover_times']].sum()
        grid_cover.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        return grid_cover

    def cal_grid_score(self, grid_cover):
        self.grid['grid_score'] = ((grid_cover['cover_times'] + 1) ** 0.5 - grid_cover[
            'cover_times'] ** 0.5).values * self.grid['grid_weight'].values

    def cal_reward(self, grid_cover):
        reward = (grid_cover['cover_times'] + grid_cover['t_vi_cover_times']) ** 0.5 - grid_cover['cover_times'] ** 0.5
        reward = (reward.values * self.grid['grid_weight'].values).sum()
        return reward

    def map_grid_score_to_node(self):
        '''

        :return:
        '''
        merge_result = pd.merge(
            self.node[['node_id', 'grid_id']], self.grid[['grid_id', 'grid_score']], on=['grid_id'])
        merge_result.sort_values(by=['node_id'], inplace=True, ignore_index=True)
        self.node['node_score'] = merge_result['grid_score'].values.astype(np.float32)

    def output_state_features(self):
        '''
        '''

        individual_loc_feat = np.zeros((self.vehicle_num, len(self.node)), dtype=np.float32)
        index1 = np.repeat(range(self.vehicle_num), 2)
        index2 = self.vehicle_states[:, : 2].flatten()
        individual_loc_feat[(index1, index2)] = 1
        loc_feat = np.zeros((len(self.node), ), dtype=np.float32)
        loc_feat[(index2, )] = 1

        grid_cover = self.t_grid_cover_times_count()
        self.cal_grid_score(grid_cover)
        self.map_grid_score_to_node()

        the_same_features = np.concatenate((
            loc_feat.reshape((loc_feat.shape[0], 1)),
            self.static_features,
            self.node[['node_score']].values,
        ), axis=1)

        return individual_loc_feat, the_same_features, \
               self.vehicle_states[:, : 2].flatten().copy().astype(np.float32)

    def reset(self):
        '''
        reset the agents(vehicles) position and grid link_score(rewards)
        :return:
        '''
        self.past_time = 0
        loc = np.random.randint(low=self.node_id_min, high=self.node_id_max + 1, size=self.vehicle_num)
        self.vehicle_states[:, 1] = loc  # location2
        self.vehicle_states[:, 0] = self.vehicle_states[:, 1].copy()  # location1
        self.vehicle_states[:, 2] = 0  # remaining_time
        self.time_of_reset_cover = 1
        self.node['t_vi_cover_times'] = 0
        if self.taxi_cover_times is not None:
            taxi_cover_times = self.taxi_cover_times[['grid_id', '{}'.format(self.time_of_reset_cover + 8)]].copy()
            taxi_cover_times.rename(columns={'{}'.format(self.time_of_reset_cover + 8): 'cover_times'}, inplace=True)
            self.node = pd.merge(self.node.drop(columns=['cover_times']), taxi_cover_times, on=['grid_id'])
            self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
        else:
            self.node['cover_times'] = 0
        self.vehicle_action_paths = [- 10000] * self.vehicle_num

        individual_loc_feat, the_same_features, loc_ID = self.output_state_features()

        self.got_reward = 0

        return individual_loc_feat, the_same_features, loc_ID

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
        vehicle_coord = self.node.loc[vehicle_loc, ['node_coordinate_x', 'node_coordinate_y']].values.astype(int)
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
            destination_coord = self.node.loc[destination_loc, ['node_coordinate_x', 'node_coordinate_y']].values.astype(int)
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
                self.node.loc[ending_node, 't_vi_cover_times'] += 1
                starting_node = ending_node
                ending_node = int(path[path_index])
                remaining_time -= self.experienced_travel_time[starting_node, ending_node]
                path_index += 1
            self.vehicle_states[path_i, 2] = -remaining_time
            self.vehicle_states[path_i, 1] = ending_node
            if remaining_time == 0:
                self.node.loc[ending_node, 't_vi_cover_times'] += 1
                self.vehicle_states[path_i, 0] = self.vehicle_states[path_i, 1].copy()
            else:
                self.vehicle_states[path_i, 0] = starting_node

        grid_cover = self.t_grid_cover_times_count()
        cooperative_reward = self.cal_reward(grid_cover)
        reward = [cooperative_reward] * self.vehicle_num
        self.got_reward += cooperative_reward
        self.node['cover_times'] += self.node['t_vi_cover_times']
        self.node['t_vi_cover_times'] = 0

        self.vehicle_action_paths = [- 10000] * self.vehicle_num
        self.past_time += action_interval
        episode_time_cost += self.action_interval

        individual_loc_feat, the_same_features, loc_ID = self.output_state_features()

        return individual_loc_feat, the_same_features, reward, done, episode_time_cost, loc_ID

    def step(self, ac_dict: dict, episode_time_cost):
        '''
        Env receives all agents' action and make one timestep forward
        :param ac_dict: dict, key allowed in list(range(self.vehicle_num)), key value allowed 0, 1, 2, 3 or 0 - 7
        :param episode_time_cost:
        :return:
        '''

        left_time = self.episode_duration - self.past_time

        # reset cover times(score)
        if self.past_time >= self.time_of_reset_cover * 3600:
            self.time_of_reset_cover += 1
            self.node['t_vi_cover_times'] = 0
            if self.taxi_cover_times is not None:
                taxi_cover_times = self.taxi_cover_times[['grid_id', '{}'.format(self.time_of_reset_cover + 8)]].copy()
                taxi_cover_times.rename(
                    columns={'{}'.format(self.time_of_reset_cover + 8): 'cover_times'}, inplace=True)
                self.node = pd.merge(self.node.drop(columns=['cover_times']), taxi_cover_times, on=['grid_id'])
                self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
            else:
                self.node['cover_times'] = 0
        assert left_time > 0, 'you need reset the environment first'

        for i in range(self.vehicle_num):
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=ac_dict[i]
            )
            self.vehicle_action_paths[i] = path

        individual_loc_feat, the_same_features, reward, done, episode_time_cost, loc_ID = \
            self.execute_action(
                left_time=left_time,
                episode_time_cost=episode_time_cost,
            )

        return individual_loc_feat, the_same_features, reward, done, episode_time_cost, loc_ID


# demo for generate_environment_with_grid_score_directional_action_edge_base_features
# if __name__ == '__main__':
#
#     # initial env
#     with open(project_path + '/grid_score_base_project/lq_network_data/edge_base_features/adjacency_matrix_with_edge_id.pickle', 'rb') as file:
#         adjacency_matrix_with_edge_id = pickle.load(file)
#     edge_info = pd.read_excel(project_path + '/grid_score_base_project/lq_network_data/edge_base_features/edge_info.xls')[[
#         'edge_id', 'length']]
#     endpoints_of_each_edge = pd.read_excel(
#         project_path + '/grid_score_base_project/lq_network_data/edge_base_features/endpoints_of_each_edge.xls')[[
#         'edge_id', 'node_id_one', 'node_id_two']]
#     edge_id_to_grid_id = pd.read_excel(
#         project_path + '/grid_score_base_project/lq_network_data/edge_base_features/edge_grid_pairs.xls')[[
#         'edge_id', 'grid_id']]
#     node_info = pd.read_excel(
#         project_path + '/grid_score_base_project/lq_network_data/edge_base_features/node_info.xls')[[
#         'node_id', 'project_x',	'project_y']]
#     node_info.rename(columns={'project_x': 'node_coordinate_x', 'project_y': 'node_coordinate_y'}, inplace=True)
#     grid_weight = pd.read_csv(
#         project_path + '/grid_score_base_project/lq_network_data/grid_static_feature.csv')[[
#         'grid_id', 'mean_gradient_variant']]
#     grid_weight.rename(columns={'mean_gradient_variant': 'grid_weight'}, inplace=True)
#     taxi_cover_times = pd.read_csv(
#         project_path + '/grid_score_base_project/lq_network_data/taxi_cover_times.csv')[[
#         'grid_id', '9', '10', '11', '12', '13', '14']]
#
#     vehicle_speed = 8
#     ac_dim = 8
#     action_interval = 180
#     T = 120
#     vehicle_num = 10
#     seed = 4000
#     env = generate_environment_with_grid_score_directional_action_edge_base_features(
#         vehicle_speed=vehicle_speed,
#         adjacency_matrix_with_edge_id=adjacency_matrix_with_edge_id,
#         edge_info=edge_info,
#         endpoints_of_each_edge=endpoints_of_each_edge,
#         edge_id_to_grid_id=edge_id_to_grid_id,
#         node_info=node_info,
#         grid_weight=grid_weight,
#         taxi_cover_times=taxi_cover_times,
#         ac_dim=ac_dim,
#         action_interval=action_interval,
#         T=T,
#         vehicle_num=vehicle_num,
#         seed=seed,
#     )
#     # reset env
#     edge_loc_features, the_same_features = env.reset()
#     # env forward
#     ac_probs_dict = {}
#     for vehicle_id in range(vehicle_num):
#         ac_probs_dict[vehicle_id] = torch.tensor([1 / ac_dim] * ac_dim)
#     st = time.time()
#     got_reward = []
#     for episode in range(100):
#         done = False
#         while not done:
#             ac_dict = {}
#             for v_id in range(vehicle_num):
#                 ac_dict[v_id] = torch.multinomial(ac_probs_dict[v_id], num_samples=1).item()
#             edge_loc_features, the_same_features, reward, done, episode_time_cost = env.step(
#                 ac_dict=ac_dict,
#                 episode_time_cost=0,
#             )
#         got_reward.append(env.got_reward)
#         edge_loc_features, the_same_features = env.reset()
#     et = time.time()
#     print(np.mean(got_reward))
#     # print(et - st)


# demo for generate_environment_with_grid_score_directional_action_node_base_features
if __name__ == '__main__':

    # initial env
    with open(project_path + '/grid_score_base_project/lq_network_data/node_base_features/adjacency_matrix_with_edge_id.pickle', 'rb') as file:
        adjacency_matrix_with_edge_id = pickle.load(file)
    edge_info = pd.read_excel(project_path + '/grid_score_base_project/lq_network_data/node_base_features/edge_info.xls')[[
        'edge_id', 'length']]
    endpoints_of_each_edge = pd.read_excel(
        project_path + '/grid_score_base_project/lq_network_data/node_base_features/endpoints_of_each_edge.xls')[[
        'edge_id', 'node_id_one', 'node_id_two']]
    node_info = pd.read_excel(
        project_path + '/grid_score_base_project/lq_network_data/node_base_features/node_info.xls')[[
        'node_id', 'grid_id', 'project_x',	'project_y']]
    node_info.rename(columns={'project_x': 'node_coordinate_x', 'project_y': 'node_coordinate_y'}, inplace=True)
    grid_weight = pd.read_csv(
        project_path + '/grid_score_base_project/lq_network_data/weight.csv')[[
        'grid_id', 'weight']]
    grid_weight.rename(columns={'weight': 'grid_weight'}, inplace=True)
    taxi_cover_times = pd.read_csv(
        project_path + '/grid_score_base_project/lq_network_data/taxi_cover_times.csv')[[
        'grid_id', '9', '10', '11', '12', '13', '14']]

    vehicle_speed = 8
    ac_dim = 8
    action_interval = 180
    T = 120
    vehicle_num = 2
    seed = 4000
    env = generate_environment_with_grid_score_directional_action_node_base_features(
        vehicle_speed=vehicle_speed,
        adjacency_matrix_with_edge_id=adjacency_matrix_with_edge_id,
        edge_info=edge_info,
        node_info=node_info,
        grid_weight=grid_weight,
        taxi_cover_times=taxi_cover_times,
        ac_dim=ac_dim,
        action_interval=action_interval,
        T=T,
        vehicle_num=vehicle_num,
        seed=seed,
    )
    # reset env
    individual_loc_feat, the_same_features, loc_ID = env.reset()
    # env forward
    ac_probs_dict = {}
    for vehicle_id in range(vehicle_num):
        ac_probs_dict[vehicle_id] = torch.tensor([1 / ac_dim] * ac_dim)
    st = time.time()
    got_reward = []
    for episode in range(100):
        done = False
        while not done:
            ac_dict = {}
            for v_id in range(vehicle_num):
                ac_dict[v_id] = torch.multinomial(ac_probs_dict[v_id], num_samples=1).item()
            individual_loc_feat, the_same_features, reward, done, episode_time_cost, loc_ID = env.step(
                ac_dict=ac_dict,
                episode_time_cost=0,
            )
        got_reward.append(env.got_reward)
        individual_loc_feat, the_same_features, loc_ID = env.reset()
    et = time.time()
    print(np.mean(got_reward))
    print(et - st)