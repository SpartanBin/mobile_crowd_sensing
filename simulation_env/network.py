import copy

import numpy as np


class generate_rectangle_network():
    '''

    '''

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.node = np.zeros((self.height, self.width))
        self.link_matrix = np.zeros((self.height * self.width, self.height * self.width))
        for row in range(self.node.shape[0]):
            for col in range(self.node.shape[1]):
                link_matrix_row = row * width + col
                try:
                    _ = self.link_matrix[link_matrix_row, link_matrix_row - 1]
                    left = link_matrix_row - 1
                    if col == 0:
                        left = None
                except:
                    left = None
                try:
                    _ = self.link_matrix[link_matrix_row, link_matrix_row + 1]
                    right = link_matrix_row + 1
                    if col == width - 1:
                        right = None
                except:
                    right = None
                try:
                    _ = self.link_matrix[link_matrix_row, link_matrix_row - width]
                    up = link_matrix_row - width
                except:
                    up = None
                try:
                    _ = self.link_matrix[link_matrix_row, link_matrix_row + width]
                    down = link_matrix_row + width
                except:
                    down = None
                for link_matrix_col in (left, right, up, down):
                    if link_matrix_col is not None:
                        self.link_matrix[link_matrix_row, link_matrix_col] = 1

    def generate_random_experienced_travel_time(self, low_second, high_second):
        self.experienced_travel_time = copy.deepcopy(np.triu(self.link_matrix))
        link_node = np.where(self.experienced_travel_time != 0)
        self.experienced_travel_time[link_node] = np.random.uniform(
            low=low_second, high=high_second, size=self.experienced_travel_time[link_node].shape
        )
        self.experienced_travel_time += self.experienced_travel_time.T - np.diag(
            self.experienced_travel_time.diagonal())

    def generate_grid(self, grid_height, grid_width):
        self.grid = np.zeros((int(self.height / grid_height + 1), int(self.width / grid_width + 1)))
        self.grid_weight = copy.deepcopy(self.grid)
        index = np.where(self.grid_weight == 0)
        self.grid_weight[index] = np.random.dirichlet(self.grid_weight[index], size=1)

    def generate_random_link_weight(self):
        self.link_weight = copy.deepcopy(np.triu(self.link_matrix))
        link_node = np.where(self.link_weight != 0)
        self.link_weight[link_node] = np.random.dirichlet(self.link_weight[link_node], size=1)
        self.link_weight += self.link_weight.T - np.diag(self.link_weight.diagonal())

    def move(self, cur_loc, action):
        '''

        :param cur_loc:
        :param action: 0-up, 1-left, 2-right, 3-down
        :return:
        '''

        if action == 0:
            next_loc = (cur_loc[0] - 1, cur_loc[1])
        elif action == 1:
            next_loc = (cur_loc[0], cur_loc[1] - 1)
        elif action == 2:
            next_loc = (cur_loc[0], cur_loc[1] + 1)
        elif action == 3:
            next_loc = (cur_loc[0] + 1, cur_loc[1])
        else:
            next_loc = None

        try:
            _ = self.node[next_loc]
        except:
            next_loc = cur_loc

        return next_loc