'''
reinforcement learning simulation environment
'''

import sys
import os
import math

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_env.network import *
from simulation_env.Shortest_path import *


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
            episode_duration: Union[int, float, None], vehicle_num: int,
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
        :param vehicle_num: the number of agent(vehicle) in this simulation environment
        :return:
        '''
        super(generate_rectangle_network_action_destination_env, self).__init__(height, width)
        self.generate_random_experienced_travel_time(
            low_second=low_second,
            high_second=high_second,
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_interval = action_interval
        self.past_time = 0
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
            self.vehicle_action_paths.append(False)
        self.vehicle_states = np.array(self.vehicle_states)

    def reset(self):
        '''
        reset the agents(vehicles) position and grid link_weight(rewards)
        '''
        self.past_time = 0
        self.vehicle_states[:, 0] = -1  # location1 row
        self.vehicle_states[:, 1] = -1  # location1 col
        coordinate_row = np.random.randint(low=0, high=self.height, size=self.vehicle_num)
        coordinate_col = np.random.randint(low=0, high=self.width, size=self.vehicle_num)
        self.vehicle_states[:, 2] = coordinate_row  # location2 row
        self.vehicle_states[:, 3] = coordinate_col  # location2 col
        self.vehicle_states[:, 4] = 0  # remaining_time
        self.vehicle_action_paths = [False] * self.vehicle_num
        self.generate_grid(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.cal_node_weight(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.left_reward = self.grid_weight.sum()

        return copy.deepcopy((self.vehicle_states[:, 0: 4], self.node_weight))

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
        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
        # C = math.degrees(math.acos((c * c - a * a - b * b)/(-2 * a * b)))

        return B

    def determine_path(self, vehicle_state: Union[tuple, list, np.ndarray], action: int):
        '''
        deciding the final node destination when agent makes an allowed action else return False to reselect action
        :param vehicle_state: agent's state
        :param action: 0-up, 1-left, 2-right, 3-down
        :return: False or list
        '''

        ac_allowed = {0, 1, 2, 3}
        vehicle_loc = vehicle_state[2: 4]
        remaining_time = vehicle_state[4]
        if vehicle_loc[0] == 0:
            ac_allowed -= {0}
        elif vehicle_loc[0] == self.height - 1:
            ac_allowed -= {3}
        if vehicle_loc[1] == 0:
            ac_allowed -= {1}
        elif vehicle_loc[1] == self.width - 1:
            ac_allowed -= {2}
        if action not in ac_allowed:
            return False  # reselect the action

        arrive_path = []
        start = int(vehicle_loc[0] * self.width + vehicle_loc[1])
        start_time = self.action_interval - remaining_time
        if start_time < 0:
            start_time = 0
        came_from, node_list = dijkstra_search(
            cost=self.experienced_travel_time,
            start=start,
            start_time=start_time,
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

    def step(self, ac_dict: dict):
        '''
        Env receives all agents' action and make one timestep forward
        :param ac_dict: dict, key allowed in list(range(self.vehicle_num)), key value allowed 0, 1, 2, 3
        :return:
        '''

        left_time = self.episode_duration - self.past_time
        assert left_time > 0 and self.left_reward > 0, 'you need reset the environment first'

        # iterate the item in self.vehicle_action_paths where key value equal to False
        for i in np.where(np.array(self.vehicle_action_paths, dtype=np.object) == False)[0]:
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=ac_dict[i]
            )
            self.vehicle_action_paths[i] = path
        reselect_agent = np.where(np.array(self.vehicle_action_paths, dtype=np.object) == False)[0]
        if len(reselect_agent) > 0:
            return reselect_agent

        # start execute the action
        # need calculate return reward
        done = False
        reward = 0
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
                reward += self.grid_weight[grid_row, grid_col]
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
                reward += self.grid_weight[grid_row, grid_col]
                self.grid_weight[grid_row, grid_col] = 0
                self.vehicle_states[i, 0: 2] = -1
            else:
                self.vehicle_states[i, 0: 2] = starting_node
        self.vehicle_action_paths = [False] * self.vehicle_num

        self.past_time += action_interval
        self.cal_node_weight(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.left_reward = self.grid_weight.sum()
        if self.left_reward <= 0:
            done = True

        return copy.deepcopy((self.vehicle_states[:, 0: 4], self.node_weight)), reward, done


if __name__ == '__main__':

    import time

    vehicle_num = 50
    env = generate_rectangle_network_action_destination_env(
        height=20,
        width=20,
        low_second=30,
        high_second=300,
        grid_height=2,
        grid_width=2,
        action_interval=180,
        episode_duration=3600,
        vehicle_num=vehicle_num,
    )
    env.reset()
    st = time.time()
    for _ in range(10000):
        actions = {}
        for i in range(vehicle_num):
            actions[i] = np.random.randint(low=0, high=3)

        output = env.step(ac_dict=actions)
        print(env.left_reward)
        if len(output) == 3:
            vehicle_states, reward, done = output
            print(vehicle_states)
            print(reward)
            print(done)
    et = time.time()
    print(et - st)