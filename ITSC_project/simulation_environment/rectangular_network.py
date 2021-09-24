'''
Base simulation road network, without method to receiving action and returning
next state and immediate reward, cannot be used for reinforcement learning
environment directly
'''

import copy
from typing import Union

import numpy as np
import scipy.special as sps


class generate_rectangle_network():
    '''
    road network with the same number of row node and column node in order
    '''

    def __init__(self, height: int, width: int):
        '''
        generate link matrix saving link relation between every network node, because of rectangle shape,
        per node link only around four nodes
        :param height: the number of row
        :param width: the number of column
        :return:
        '''
        self.height = height
        self.width = width
        self.node = np.zeros((self.height, self.width))
        self.link_matrix = np.zeros((self.height * self.width, self.height * self.width))
        self.stored_link_weight = {}
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
                    if row == 0:
                        up = None
                except:
                    up = None
                try:
                    _ = self.link_matrix[link_matrix_row, link_matrix_row + width]
                    down = link_matrix_row + width
                    if row == height - 1:
                        down = None
                except:
                    down = None
                for link_matrix_col in (left, right, up, down):
                    if link_matrix_col is not None:
                        self.link_matrix[link_matrix_row, link_matrix_col] = 1

    def generate_random_experienced_travel_time(self, low_second: Union[int, float], high_second: Union[int, float]):
        '''
        generate random experienced travel time between the range low_second to high_second on link nodes
        :param low_second: seconds, the lower bound of random experienced travel time
        :param high_second: seconds, the upper bound of random experienced travel time
        :return:
        '''
        self.experienced_travel_time = copy.deepcopy(np.triu(self.link_matrix))
        link_node = np.where(self.experienced_travel_time != 0)
        self.experienced_travel_time[link_node] = np.random.uniform(
            low=low_second, high=high_second, size=self.experienced_travel_time[link_node].shape
        )
        self.experienced_travel_time += self.experienced_travel_time.T - np.diag(
            self.experienced_travel_time.diagonal())
        self.experienced_travel_time[self.experienced_travel_time == 0] = float('inf')

    def discrete_gaussian_distribution(self, grid_height, grid_width):
        '''
        generate link_weight(rewards) with gaussian distribution
        :param grid_height: per grid height
        :param grid_width: per grid width
        :return:
        '''
        height = int(self.height / grid_height + 0.999999999999)
        width = int(self.width / grid_width + 0.999999999999)
        gd = np.zeros((height, width))
        for i in range(len(gd)):
            for j in range(len(gd[i])):
                gd[i, j] = sps.comb(height - 1, i, exact=True) + sps.comb(width - 1, j, exact=True)
        gd = np.float32(gd) / np.float32(gd).sum()
        return gd

    def generate_grid(self, grid_height: int, grid_width: int, link_weight_distribution: str):
        '''
        generate grid with random link_weight(rewards). Nodes in same grid share the same link_weight
        :param grid_height: per grid height
        :param grid_width: per grid width
        :param link_weight_distribution: now only support gaussian distribution and uniform distribution
        :return:
        '''

        if (grid_height, grid_width) not in self.stored_link_weight.keys():
            self.stored_link_weight[(grid_height, grid_width)] = self.discrete_gaussian_distribution(
                grid_height, grid_width)
        link_weight_GD = copy.deepcopy(self.stored_link_weight[(grid_height, grid_width)])

        self.grid = np.ones((int(self.height / grid_height + 0.999999999999), int(
            self.width / grid_width + 0.999999999999)))
        if link_weight_distribution == 'GD':
            self.grid_weight = link_weight_GD
        else:
            self.grid_weight = copy.deepcopy(self.grid)
            index = np.where(self.grid_weight == 1)
            self.grid_weight[index] = np.random.dirichlet(self.grid_weight[index], size=1)

    def cal_node_weight(self, grid_height: int, grid_width: int):
        '''
        map the weights of grid into node
        :param grid_height: per grid height
        :param grid_width: per grid width
        :return:
        '''
        self.node_weight = np.repeat(self.grid_weight, grid_height, axis=0)
        self.node_weight = np.repeat(self.node_weight, grid_width, axis=1)
        self.node_weight = self.node_weight[0: self.height, 0: self.width]

    def generate_random_link_weight(self):
        '''
        generate random link_weight(rewards) for every roads
        '''
        self.link_weight = copy.deepcopy(np.triu(self.link_matrix))
        link_node = np.where(self.link_weight != 0)
        self.link_weight[link_node] = np.random.dirichlet(self.link_weight[link_node], size=1)
        self.link_weight += self.link_weight.T - np.diag(self.link_weight.diagonal())