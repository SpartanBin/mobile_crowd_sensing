'''
reinforcement learning simulation environment
'''

import sys
import os
import math
import random
import pickle

import torch

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from ITSC_project.simulation_environment.rectangular_network import *
from ITSC_project.simulation_environment.Shortest_path import *


class generate_rectangle_network_action_destination_env(generate_rectangle_network):
    '''
    Use a node destination as an action, with grid link_weight(rewards) and rectangle shape network.
    While agents decide the direction of further actions, the selection of destination node is made
    by the environment considering angle between the vertices at action's direction and final node's
    direction. And the path to final node's direction is the shortest path.
    '''

    def __init__(
            self, height: int, width: int, low_second: Union[int, float], high_second: Union[int, float],
            grid_height: int, grid_width: int, action_interval: Union[int, float],
            left_reward_to_stop: float, episode_duration: Union[int, float, None],
            link_weight_distribution: str, vehicle_num: int, seed: int,
    ):
        '''
        generate link matrix saving link relation between every network node, because of rectangle shape,
        per node link only around four nodes
        :param height: the number of row
        :param width: the number of column
        :param low_second: seconds, the lower bound of random experienced travel time
        :param high_second: seconds, the upper bound of random experienced travel time
        :param grid_height: per grid height
        :param grid_width: per grid width
        :param action_interval: seconds, every timestep's time interval
        :param episode_duration: seconds, episode duration time, if None, continuing until no reward
        :param link_weight_distribution: now only support gaussian distribution and uniform distribution
        :param vehicle_num: the number of agent(vehicle) in this simulation environment
        :return:
        '''

        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)

        super(generate_rectangle_network_action_destination_env, self).__init__(height, width)
        self.generate_random_experienced_travel_time(
            low_second=low_second,
            high_second=high_second,
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_interval = action_interval
        self.past_time = 0
        self.left_reward_to_stop = left_reward_to_stop
        self.episode_duration = episode_duration
        if self.episode_duration is None:
            self.episode_duration = float('inf')
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        self.vehicle_action_paths = []
        for i in range(self.vehicle_num):
            self.vehicle_states.append([
                -1, -1,  # location1
                -1, -1,  # location2
                0,  # remaining_time
            ])
            self.vehicle_action_paths.append(- 10000)
        self.vehicle_states = np.array(self.vehicle_states)
        self.reset(link_weight_distribution=link_weight_distribution)
        self.episode_got_scores = np.zeros(self.grid_weight.shape)
        # self.generate_grid(
        #     grid_height=self.grid_height,
        #     grid_width=self.grid_width,
        # )
        # self.grid_weight_ = copy.deepcopy(self.grid_weight)

    def reset(self, link_weight_distribution: str):
        '''
        reset the agents(vehicles) position and grid link_weight(rewards)
        :param link_weight_distribution: now only support gaussian distribution and uniform distribution
        :return:
        '''
        self.past_time = 0
        coordinate_row = np.random.randint(low=0, high=self.height, size=self.vehicle_num)
        coordinate_col = np.random.randint(low=0, high=self.width, size=self.vehicle_num)
        self.vehicle_states[:, 2] = coordinate_row  # location2 row
        self.vehicle_states[:, 3] = coordinate_col  # location2 col
        self.vehicle_states[:, 0] = self.vehicle_states[:, 2].copy()  # location1 row
        self.vehicle_states[:, 1] = self.vehicle_states[:, 3].copy()  # location1 col
        self.vehicle_states[:, 4] = 0  # remaining_time
        self.vehicle_action_paths = [- 10000] * self.vehicle_num
        self.generate_grid(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            link_weight_distribution=link_weight_distribution,
        )
        # self.grid_weight = copy.deepcopy(self.grid_weight_)
        self.cal_node_weight(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.left_reward = self.grid_weight.sum()

        node_weight = copy.deepcopy(self.node_weight.flatten())
        vehicle_states_start = (self.vehicle_states[:, 0] * self.width + self.vehicle_states[:, 1]).reshape((-1, 1))
        vehicle_states_end = (self.vehicle_states[:, 2] * self.width + self.vehicle_states[:, 3]).reshape((-1, 1))
        vehicle_states = np.hstack((vehicle_states_start, vehicle_states_end))
        self.episode_got_scores = np.zeros(self.grid_weight.shape)

        return [vehicle_states, node_weight]

    def return_allowed_action(self, ending_node):
        ac_allowed = {0, 1, 2, 3}
        if ending_node[0] == 0:
            ac_allowed -= {0}
        elif ending_node[0] == self.height - 1:
            ac_allowed -= {3}
        if ending_node[1] == 0:
            ac_allowed -= {1}
        elif ending_node[1] == self.width - 1:
            ac_allowed -= {2}
        return ac_allowed

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

    def determine_path(self, vehicle_state: Union[tuple, list, np.ndarray], action: int):
        '''
        deciding the final node destination when agent makes an allowed action else return - 10000 to reselect action
        :param vehicle_state: agent's state
        :param action: 0-up, 1-left, 2-right, 3-down
        :return: - 10000 or list
        '''

        vehicle_loc = vehicle_state[2: 4]
        remaining_time = vehicle_state[4]
        ac_allowed = self.return_allowed_action(ending_node=vehicle_loc)
        if action not in ac_allowed:
            return - 10000  # reselect the action

        arrive_path = []
        start = int(vehicle_loc[0] * self.width + vehicle_loc[1])
        start_time = self.action_interval - remaining_time
        if start_time < 0:
            start_time = 0
        came_from, node_list = dijkstra_search(
            cost=self.experienced_travel_time,
            start=start,
            start_time=start_time,
            end_time=start_time + self.experienced_travel_time.max() + 60,
            node_length=len(ac_allowed),
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
        if action == 0:
            action_direction = (vehicle_loc[0] - 1, vehicle_loc[1])
        elif action == 1:
            action_direction = (vehicle_loc[0], vehicle_loc[1] - 1)
        elif action == 2:
            action_direction = (vehicle_loc[0], vehicle_loc[1] + 1)
        else:
            action_direction = (vehicle_loc[0] + 1, vehicle_loc[1])
        angle_list = []
        for path in arrive_path:
            destination_row = path[-1] // self.width
            destination_col = int(path[-1] - destination_row * self.width)
            angle_list.append(self.cal_angle(action_direction, vehicle_loc, (destination_row, destination_col)))
        index = angle_list.index(min(angle_list))

        return arrive_path[index]

    def execute_action(self, left_time, reward_type, cooperative_weight, negative_constant_reward, episode_time_cost):
        '''

        :param left_time:
        :param reward_type: 'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance' are allowed value
        :param cooperative_weight: If reward_type is 'greedy_mean', final reward is equal to "greedy_reward +
        cooperative_weight * mean_of_all_greedy_rewards". And if reward_type is 'team_spirit', final reward is
        equal to "(1 - cooperative_weight) * greedy_reward + cooperative_weight * mean_of_all_greedy_rewards".
        :param negative_constant_reward: In past experiments, when reward_type is 'sum', negative_constant_reward is
        equal to -0.05
        :param episode_time_cost:
        :return:
        '''

        # start execute the action
        assert reward_type == 'greedy' or reward_type == 'sum' or reward_type == 'greedy_mean' or \
               reward_type == 'team_spirit' or reward_type == 'distance', \
               "'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance' are allowed reward_type value"
        done = False
        first_passed_node_vehicle = {}  # for save reward for every vehicle's own
        reward = np.zeros(self.vehicle_num, dtype=np.float32)
        #################################################################
        # for calculate distance reward coefficient
        all_vehicles_starting_node = copy.deepcopy(self.vehicle_states[:, 0: 2])
        all_distance_reward_coef = np.zeros((self.vehicle_num, self.vehicle_num), dtype=np.float32)
        #################################################################
        action_interval = self.action_interval
        if action_interval >= left_time:
            action_interval = left_time
            done = True
        for i, path in enumerate(self.vehicle_action_paths):
            path_index = 1
            remaining_time = action_interval
            remaining_time -= self.vehicle_states[i, 4]
            starting_node = self.vehicle_states[i, 0: 2]
            starting_code = int(starting_node[0] * self.width + starting_node[1])
            ending_node = self.vehicle_states[i, 2: 4]
            ending_code = int(ending_node[0] * self.width + ending_node[1])
            while remaining_time > 0:
                ending_row = ending_code // self.width
                ending_col = int(ending_code - ending_row * self.width)
                ending_node = (ending_row, ending_col)
                grid_row = int(ending_node[0] / self.grid_height)
                grid_col = int(ending_node[1] / self.grid_width)
                if (grid_row, grid_col) not in first_passed_node_vehicle.keys():
                    first_passed_node_vehicle[(grid_row, grid_col)] = {}
                    first_passed_node_vehicle[(grid_row, grid_col)]['vehicle'] = i
                    first_passed_node_vehicle[(grid_row, grid_col)]['remaining_time'] = remaining_time
                    first_passed_node_vehicle[(grid_row, grid_col)]['reward'] = self.grid_weight[grid_row, grid_col]
                else:
                    if first_passed_node_vehicle[(grid_row, grid_col)]['remaining_time'] < remaining_time:
                        first_passed_node_vehicle[(grid_row, grid_col)]['vehicle'] = i
                        first_passed_node_vehicle[(grid_row, grid_col)]['remaining_time'] = remaining_time
                self.grid_weight[grid_row, grid_col] = 0
                starting_code = ending_code
                ending_code = int(path[path_index])
                remaining_time -= self.experienced_travel_time[starting_code, ending_code]
                path_index += 1
            starting_row = starting_code // self.width
            starting_col = int(starting_code - starting_row * self.width)
            starting_node = (starting_row, starting_col)
            ending_row = ending_code // self.width
            ending_col = int(ending_code - ending_row * self.width)
            ending_node = (ending_row, ending_col)
            self.vehicle_states[i, 4] = -remaining_time
            self.vehicle_states[i, 2: 4] = ending_node
            if remaining_time == 0:
                grid_row = int(ending_node[0] / self.grid_height)
                grid_col = int(ending_node[1] / self.grid_width)
                if (grid_row, grid_col) not in first_passed_node_vehicle.keys():
                    first_passed_node_vehicle[(grid_row, grid_col)] = {}
                    first_passed_node_vehicle[(grid_row, grid_col)]['vehicle'] = i
                    first_passed_node_vehicle[(grid_row, grid_col)]['remaining_time'] = remaining_time
                    first_passed_node_vehicle[(grid_row, grid_col)]['reward'] = self.grid_weight[grid_row, grid_col]
                self.grid_weight[grid_row, grid_col] = 0
                self.vehicle_states[i, 0: 2] = self.vehicle_states[i, 2: 4].copy()
            else:
                self.vehicle_states[i, 0: 2] = starting_node
            #################################################################
            # calculate distance reward coefficient
            if reward_type == 'distance':
                all_distance_reward_coef[:, i] = (
                    (all_vehicles_starting_node[i, 0] - all_vehicles_starting_node[:, 0]) ** 2 + (
                     all_vehicles_starting_node[i, 1] - all_vehicles_starting_node[:, 1]) ** 2) ** 0.5
            #################################################################

        #################################################################
        # calculate distance reward coefficient
        if reward_type == 'distance':
            all_distance_reward_coef = all_distance_reward_coef / ((self.height ** 2 + self.width ** 2) ** 0.5)
            all_distance_reward_coef = np.tanh(all_distance_reward_coef) * cooperative_weight
            all_distance_reward_coef[(range(self.vehicle_num), range(self.vehicle_num))] = 1 - cooperative_weight
        #################################################################

        for key in first_passed_node_vehicle.keys():
            v = first_passed_node_vehicle[key]['vehicle']
            reward[v] += first_passed_node_vehicle[key]['reward']
            self.episode_got_scores[key] += first_passed_node_vehicle[key]['reward']

        #################################################################
        # calculate final reward
        if reward_type == 'distance':
            reward = np.dot(reward.reshape(1, -1), all_distance_reward_coef)
        elif reward_type == 'sum':
            reward[:] = np.sum(reward)
        elif reward_type == 'greedy_mean':
            reward += cooperative_weight * np.mean(reward)
        elif reward_type == 'team_spirit':
            reward = reward * (1 - cooperative_weight) + np.mean(reward) * cooperative_weight
        #################################################################

        if negative_constant_reward > 0:
            reward -= negative_constant_reward
        else:
            reward += negative_constant_reward
        self.vehicle_action_paths = [- 10000] * self.vehicle_num

        self.past_time += action_interval
        self.cal_node_weight(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.left_reward = self.grid_weight.sum()
        if self.left_reward <= self.left_reward_to_stop:
            done = True

        episode_time_cost += self.action_interval

        node_weight = copy.deepcopy(self.node_weight.flatten())
        vehicle_states_start = (self.vehicle_states[:, 0] * self.width + self.vehicle_states[:, 1]).reshape((-1, 1))
        vehicle_states_end = (self.vehicle_states[:, 2] * self.width + self.vehicle_states[:, 3]).reshape((-1, 1))
        vehicle_states = np.hstack((vehicle_states_start, vehicle_states_end))

        return [vehicle_states, node_weight], reward, done, episode_time_cost

    def step(self, ac_dict: dict, reward_type, cooperative_weight, negative_constant_reward, episode_time_cost):
        '''
        Env receives all agents' action and make one timestep forward
        :param ac_dict: dict, key allowed in list(range(self.vehicle_num)), key value allowed 0, 1, 2, 3
        :param reward_type:
        :param cooperative_weight:
        :param negative_constant_reward:
        :param episode_time_cost:
        :return:
        '''

        left_time = self.episode_duration - self.past_time
        assert left_time > 0 and self.left_reward > self.left_reward_to_stop, 'you need reset the environment first'

        # iterate the item in self.vehicle_action_paths where key value equal to - 10000
        for i in np.where(np.array(self.vehicle_action_paths) == - 10000)[0]:
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=ac_dict[i]
            )
            self.vehicle_action_paths[i] = path
        reselect_agent = np.where(np.array(self.vehicle_action_paths) == - 10000)[0]
        if len(reselect_agent) > 0:
            return reselect_agent

        return self.execute_action(
            left_time=left_time,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            episode_time_cost=episode_time_cost,
        )

    def step_by_action_probs(self, ac_probs_dict: dict, reward_type, cooperative_weight, negative_constant_reward,
                             episode_time_cost):
        '''
        Env receives all agents' action probability, sampling from it, and make one timestep forward
        :param ac_probs_dict: dict, key allowed in list(range(self.vehicle_num)), key value are torch.tensor like
        [0.2512, 0.2487, 0.2500, 0.2501]
        :param reward_type:
        :param cooperative_weight:
        :param negative_constant_reward:
        :param episode_time_cost:
        :return:
        '''

        left_time = self.episode_duration - self.past_time
        assert left_time > 0 and self.left_reward > self.left_reward_to_stop, 'you need reset the environment first'

        actions = np.array([np.nan] * self.vehicle_num)
        for i in range(self.vehicle_num):
            ac_allowed = list(self.return_allowed_action(ending_node=self.vehicle_states[i, 2: 4]))
            actions[i] = ac_allowed[torch.multinomial(ac_probs_dict[i][ac_allowed], num_samples=1).item()]
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=actions[i]
            )
            self.vehicle_action_paths[i] = path

        obs, reward, done, episode_time_cost = self.execute_action(
            left_time=left_time,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            episode_time_cost=episode_time_cost,
        )

        return actions, obs, reward, done, episode_time_cost


if __name__ == '__main__':

    episode_total_scores = []
    for seed in [4000, 8000, 12000, 16000, 20000]:
        vehicle_num = 20
        height = 30
        width = 30
        grid_height = 2
        grid_width = 2
        link_weight_distribution = 'UD'
        torch.manual_seed(seed)
        env = generate_rectangle_network_action_destination_env(
            height=height,
            width=width,
            low_second=30,
            high_second=300,
            grid_height=grid_height,
            grid_width=grid_width,
            action_interval=180,
            left_reward_to_stop=0.01,
            episode_duration=int(3600),
            vehicle_num=vehicle_num,
            seed=seed,
            link_weight_distribution=link_weight_distribution,
        )
        with open(project_path + '/ITSC_project/simulation_environment/experienced_travel_time/experienced_travel_time_height{}_width{}_seed{}.pickle'.format(height, width, seed), 'rb') as file:
            env.experienced_travel_time = pickle.load(file)
        ac_probs_dict = {}
        for i in range(vehicle_num):
            ac_probs_dict[i] = torch.tensor([0.25] * 4)
        for i in range(100):
            done = False
            while not done:
                _, _, _, done, _ = env.step_by_action_probs(
                    ac_probs_dict=ac_probs_dict,
                    reward_type='greedy',
                    cooperative_weight=0.5,
                    negative_constant_reward=0,
                    episode_time_cost=0,
                )
            episode_total_score = 1 - env.left_reward
            episode_total_scores.append(episode_total_score)
            env.reset(link_weight_distribution=link_weight_distribution)
    print(np.mean(episode_total_scores))