import sys
import os

import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_env.network import *


class generate_rectangle_network_env(generate_rectangle_network):

    def __init__(self, height, width, low_second, high_second, grid_height, grid_width, action_interval, vehicle_num):
        super(generate_rectangle_network_env).__init__(height, width)
        self.generate_random_experienced_travel_time(
            low_second=low_second,
            high_second=high_second,
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_interval = action_interval
        self.vehicle_num = vehicle_num
        self.vehicle = {}
        for i in range(self.vehicle_num):
            self.vehicle[i] = {
                'location1': None,
                'location2': None,
                'remaining_time': None,
            }

    def reset(self):
        coordinate_row = np.random.randint(low=0, high=self.height, size=self.vehicle_num)
        coordinate_col = np.random.randint(low=0, high=self.width, size=self.vehicle_num)
        for i in range(self.vehicle_num):
            self.vehicle[i] = {
                'location1': (coordinate_row[i], coordinate_col[i]),
                'location2': None,
                'remaining_time': 0,
            }
        self.generate_grid(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )