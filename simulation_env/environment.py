import sys
import os
import math

import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_env.network import *
from simulation_env.Shortest_path import *


class generate_rectangle_network_action_destination_env(generate_rectangle_network):

    def __init__(
            self, height, width, low_second, high_second, grid_height, grid_width, action_interval,
            vehicle_num
    ):
        super(generate_rectangle_network_action_destination_env).__init__(height, width)
        self.generate_random_experienced_travel_time(
            low_second=low_second,
            high_second=high_second,
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_interval = action_interval
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        self.vehicle_action_paths = []
        for i in range(self.vehicle_num):
            self.vehicle_states.append([
                np.nan, np.nan,  # location1
                np.nan, np.nan,  # location2
                0,  # remaining_time
            ])
            self.vehicle_action_paths.append(False)
        self.vehicle_states = np.array(self.vehicle_states)
        self.vehicle_action_paths = np.array(self.vehicle_action_paths)
        # self.vehicle_action_costs = copy.deepcopy(self.vehicle_action_paths)

    def reset(self):
        self.vehicle_states[:, 0] = np.nan  # location1 row
        self.vehicle_states[:, 1] = np.nan  # location1 col
        coordinate_row = np.random.randint(low=0, high=self.height, size=self.vehicle_num)
        coordinate_col = np.random.randint(low=0, high=self.width, size=self.vehicle_num)
        self.vehicle_states[:, 2] = coordinate_row  # location2 row
        self.vehicle_states[:, 3] = coordinate_col  # location2 col
        self.vehicle_states[:, 4] = 0  # remaining_time
        self.vehicle_action_paths[:] = False
        # self.vehicle_action_costs[:] = False
        self.generate_grid(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )

    def cal_angle(self, point_1, point_2, point_3):
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

    def determine_path(self, vehicle_state, action):
        '''

        :param vehicle_state:
        :param action: 0-up, 1-left, 2-right, 3-down
        :return:
        '''

        ac_allowed = set((0, 1, 2, 3))
        vehicle_loc = vehicle_state[2: 4]
        remaining_time = vehicle_state[4]
        if vehicle_loc[0] == 0:
            ac_allowed -= set((0))
        elif vehicle_loc[0] == self.height - 1:
            ac_allowed -= set((3))
        if vehicle_loc[1] == 0:
            ac_allowed -= set((1))
        elif vehicle_loc[1] == self.width - 1:
            ac_allowed -= set((2))
        if action not in ac_allowed:
            return False, False  # reselect the action

        arrive_path = []
        # arrive_cost = []
        came_from, node_list, cost_so_far = dijkstra_search(
            cost=self.experienced_travel_time,
            start=vehicle_loc[0] * self.width + vehicle_loc[1],
            start_time=self.action_interval - remaining_time,
            node_length=len(ac_allowed),
        )
        for node in node_list:
            path = reconstruct_path(
                came_from=came_from,
                start=0,
                goal=node,
                # start_time=self.action_interval - remaining_time,
                # cost_so_far=cost_so_far,
            )
            arrive_path.append(path)
            # arrive_cost.append(node_cost)

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

        return arrive_path[index]  # , arrive_cost[index]

    def step(self, ac_dict: dict):
        '''

        :param ac_dict: dict, key value allowed 0, 1, 2, 3
        :return:
        '''

        # iterate the item in self.vehicle_action_paths where key value equal to False
        for i in np.where(self.vehicle_action_paths == False)[0]:
            path = self.determine_path(
                vehicle_state=self.vehicle_states[i],
                action=ac_dict[i]
            )
            self.vehicle_action_paths[i] = path
            # self.vehicle_action_costs[i] = cost
        reselect_agent = np.where(self.vehicle_action_paths == False)[0]
        if len(reselect_agent) > 0:
            return reselect_agent

        # start execute the action
        # need calculate return reward
        for i, path in enumerate(self.vehicle_action_paths):
            path_index = 1
            remaining_time = self.action_interval
            remaining_time -= self.vehicle_states[i, 4]
            starting_node = self.vehicle_states[i, 0: 2]
            starting_code = starting_node[0] * self.width + starting_node[1]
            ending_node = self.vehicle_states[i, 2: 4]
            ending_code = ending_node[0] * self.width + ending_node[1]
            while remaining_time > 0:
                starting_code = ending_code
                ending_code = path[path_index]
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
                self.vehicle_states[i, 0: 2] = np.nan
            else:
                self.vehicle_states[i, 0: 2] = starting_node



        self.vehicle_action_paths[:] = False