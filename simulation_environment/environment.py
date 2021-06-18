'''
reinforcement learning simulation environment
'''

import sys
import os
import copy
import math
import random
import time
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_environment.Shortest_path import *
from simulation_environment.network import *


class generate_synchronous_timestep_environment_with_directional_action():
    '''
    generate environment with synchronous timestep and directional action for multiple agents
    '''

    def __init__(self):
        pass


class generate_asynchronous_timestep_environment():
    '''
    generate environment with asynchronous timestep for multiple agents
    '''

    def __init__(
            self, experienced_travel_time: np.ndarray, node_id_to_grid_id: pd.DataFrame,
            ts_duration: Union[int, float], num_of_ts: int, BETA: Union[int, float],
            vehicle_num: int, seed: int,
    ):
        '''

        :param experienced_travel_time:
        :param node_id_to_grid_id: need contain column 'grid_id', 'node_id'.
        :param ts_duration: timestep unit is second
        :param num_of_ts: number of timestep
        :param BETA:
        :param vehicle_num:
        :param seed:
        :return:
        '''

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.experienced_travel_time = experienced_travel_time
        self.node = node_id_to_grid_id
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)
        self.node['node_cover'] = 0
        self.grid = pd.DataFrame({'grid_id': self.node['grid_id'].unique()})
        self.grid.sort_values(by=['grid_id'], inplace=True, ignore_index=True)
        self.node_id_min = self.node['node_id'].min()
        self.node_id_max = self.node['node_id'].max()

        self.ts_duration = ts_duration
        self.num_of_ts = num_of_ts
        self.past_time = 0
        self.ts = 0
        self.episode_duration = self.ts_duration * self.num_of_ts
        self.BETA = BETA
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        for i in range(self.vehicle_num):
            self.vehicle_states.append([
                -1,  # location1
                -1,  # location2
                0,  # remaining_time
            ])
        self.vehicle_states = np.array(self.vehicle_states)
        # calculate node allowed action
        self.node_allowed_action = {}
        self.max_actions_num = 0
        for i in range(self.experienced_travel_time.shape[0]):
            self.node_allowed_action[i] = np.where(self.experienced_travel_time[i] < np.inf)[0]
            if self.max_actions_num < len(self.node_allowed_action[i]):
                self.max_actions_num = len(self.node_allowed_action[i])

        self.grid_cover_col_names = (pd.Series(['grid_cover'] * self.num_of_ts) +
                                     pd.Series(range(self.num_of_ts)).astype(str)).tolist()
        self.p_col_names = (pd.Series(['p'] * self.num_of_ts) + pd.Series(range(self.num_of_ts)).astype(str)).tolist()

    def grid_cover_count(self):
        grid_cover = self.node.groupby(['grid_id'], as_index=False)[['node_cover']].sum()
        return grid_cover

    def map_grid_cover_to_node(self):
        if 'grid_cover' in self.node.columns.values:
            del self.node['grid_cover']
        self.node = pd.merge(
            self.node, self.grid[['grid_id', 'grid_cover']], on=['grid_id'])
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)

    def ts_grid_cover_statistic(self):
        self.grid['grid_cover{}'.format(self.ts)] = self.grid['grid_cover'].values
        grid_cover_or_not = copy.deepcopy(self.grid['grid_cover'].values)
        grid_cover_or_not[np.where(grid_cover_or_not > 0)] = 1
        if self.ts > 0:
            self.grid['ts_grid_cover{}'.format(self.ts)] = self.grid[
                'ts_grid_cover{}'.format(self.ts - 1)].values + grid_cover_or_not
        else:
            self.grid['ts_grid_cover{}'.format(self.ts)] = grid_cover_or_not
        p = self.grid['ts_grid_cover{}'.format(self.ts)].values / (self.ts + 1)
        p[np.where(p == 0)] = self.BETA / (self.ts + 1)
        self.grid['p'] = p
        self.grid['p{}'.format(self.ts + 1)] = p

    def reset_grid_weight(self, grid_weight: Union[pd.DataFrame, None]):
        if grid_weight is not None:
            self.grid = grid_weight
        else:
            self.grid['weight'] = np.random.dirichlet(
                np.ones(shape=len(self.grid), dtype=np.int), size=1).flatten()

    def map_grid_info_to_node(self):
        if 'grid_cover' in self.node.columns.values:
            del self.node['grid_cover']
        if 'weight' in self.node.columns.values:
            del self.node['weight']
        if 'p' in self.node.columns.values:
            del self.node['p']
        self.node = pd.merge(
            self.node, self.grid[['grid_id', 'grid_cover', 'weight', 'p']], on=['grid_id'])
        self.node.sort_values(by=['node_id'], inplace=True, ignore_index=True)

    def cal_reward(self):
        weight = self.grid['weight'].values
        weight = np.repeat(weight, self.num_of_ts).reshape((len(weight), self.num_of_ts))
        grid_cover = self.grid[self.grid_cover_col_names].values
        p = self.grid[self.p_col_names].values
        nn0_where = np.where(grid_cover != 0)
        n0_where = np.where(grid_cover == 0)
        reward = np.zeros_like(grid_cover, dtype=np.float32)
        reward[nn0_where] = - weight[nn0_where] * grid_cover[nn0_where] * np.log(
            1 - (1 - p[nn0_where]) ** grid_cover[nn0_where])
        reward[n0_where] = weight[n0_where] * np.log(p[n0_where])
        return reward.sum()

    def reset(self, grid_weight):
        '''
        reset the agents(vehicles) position and grid link_weight(rewards)
        :return:
        '''
        self.past_time = 0
        self.ts = 0
        coordinate = np.random.randint(low=self.node_id_min, high=self.node_id_max + 1, size=self.vehicle_num)
        self.vehicle_states[:, 1] = coordinate  # location2
        self.vehicle_states[:, 0] = self.vehicle_states[:, 1].copy()  # location1
        self.vehicle_states[:, 2] = 0  # remaining_time
        self.reset_grid_weight(
            grid_weight=grid_weight,
        )
        self.node['node_cover'] = 0
        self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
        self.ts_grid_cover_statistic()
        self.grid['p0'] = self.grid['p'].values
        self.map_grid_info_to_node()

        node_weight = self.node['weight'].values
        vehicle_states = copy.deepcopy(self.vehicle_states)
        grid_cover = self.node['grid_cover'].values
        p = self.node['p'].values
        need_move = list(range(self.vehicle_num))

        return vehicle_states, node_weight, grid_cover, p, need_move

    def execute_action(self, actions: dict, episode_time_cost):
        '''

        :param actions:
        :param episode_time_cost:
        :return:
        '''

        done = False
        for vehicle_id in actions.keys():
            action = actions[vehicle_id]
            vehicle_origin = self.vehicle_states[vehicle_id, 0]
            self.vehicle_states[vehicle_id, 1] = action
            vehicle_destination = self.vehicle_states[vehicle_id, 1]
            self.vehicle_states[vehicle_id, 2] = self.experienced_travel_time[vehicle_origin, vehicle_destination]

        min_time_to_next_node = self.vehicle_states[:, 2].min()
        ts_left_time = (self.ts + 1) * self.ts_duration - self.past_time

        while min_time_to_next_node >= ts_left_time:

            self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
            self.ts_grid_cover_statistic()
            self.map_grid_info_to_node()
            self.node['node_cover'] = 0
            self.ts += 1

            self.past_time += ts_left_time
            self.vehicle_states[:, 2] = self.vehicle_states[:, 2] - ts_left_time
            episode_time_cost += ts_left_time

            ts_left_time = (self.ts + 1) * self.ts_duration - self.past_time
            min_time_to_next_node = self.vehicle_states[:, 2].min()

            if self.ts == self.num_of_ts:
                done = True
                break

        if min_time_to_next_node > 0:
            self.past_time += min_time_to_next_node
            self.vehicle_states[:, 2] = self.vehicle_states[:, 2] - min_time_to_next_node
            episode_time_cost += min_time_to_next_node

        need_move = np.where(self.vehicle_states[:, 2] == 0)[0].tolist()
        self.vehicle_states[(need_move, [0] * len(need_move))] = self.vehicle_states[(need_move, [1] * len(need_move))]
        arrival_loc = self.vehicle_states[(need_move, [0] * len(need_move))].tolist()
        if len(set(arrival_loc)) == len(arrival_loc):
            self.node.loc[arrival_loc, 'node_cover'] += 1
        else:
            for loc in arrival_loc:
                self.node.loc[loc, 'node_cover'] += 1
        self.grid['grid_cover'] = self.grid_cover_count()['node_cover'].values
        self.map_grid_cover_to_node()

        node_weight = self.node['weight'].values
        vehicle_states = copy.deepcopy(self.vehicle_states)
        grid_cover = self.node['grid_cover'].values
        p = self.node['p'].values

        if done:
            reward = self.cal_reward()
        else:
            reward = 0

        return vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost

    def step_by_action_probs(
            self, ac_probs_dict: dict, episode_time_cost):
        '''
        Env receives all agents' action probability, sampling from it, and make one timestep forward
        :param ac_probs_dict: dict, key allowed in list(range(self.vehicle_num)), key value are torch.tensor like
        [0.2512, 0.2487, 0.2500, 0.2501]
        :param episode_time_cost:
        :return:
        '''

        left_time = self.episode_duration - self.past_time
        assert left_time > 0, 'you need reset the environment first'

        actions = {}
        for i in ac_probs_dict.keys():
            ac_allowed = self.node_allowed_action[self.vehicle_states[i, 1]]
            actions[i] = ac_allowed[torch.multinomial(ac_probs_dict[i][ac_allowed], num_samples=1).item()]

        vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = self.execute_action(
            actions=actions,
            episode_time_cost=episode_time_cost,
        )

        return actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost


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

    # initial env
    ts_duration = 600
    num_of_ts = 2
    BETA = 0.5
    vehicle_num = 2
    env = generate_asynchronous_timestep_environment(
        experienced_travel_time=experienced_travel_time,
        node_id_to_grid_id=node_id_to_grid_id,
        ts_duration=ts_duration,
        num_of_ts=num_of_ts,
        BETA=BETA,
        vehicle_num=vehicle_num,
        seed=seed,
    )

    # reset env
    grid_weight = None
    vehicle_states, node_weight, grid_cover, p, need_move = env.reset(grid_weight=grid_weight)

    # env forward
    ac_probs_dict = {}
    for vehicle_id in range(vehicle_num):
        ac_probs_dict[vehicle_id] = torch.tensor([1 / (height * width)] * (height * width))
    st = time.time()
    for episode in range(100):
        done = False
        while not done:
            ac_probs_dict_ = {}
            for v_id in need_move:
                ac_probs_dict_[v_id] = ac_probs_dict[v_id]
            actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = \
                env.step_by_action_probs(
                    ac_probs_dict=ac_probs_dict_,
                    episode_time_cost=0,
                )
        vehicle_states, node_weight, grid_cover, p, need_move = env.reset(grid_weight=grid_weight)
    et = time.time()
    print(et - st)