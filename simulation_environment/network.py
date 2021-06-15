'''
Use for generating networks
'''

import copy
import random
from typing import Union

import numpy as np
import pandas as pd


def generate_rectangle_network(
        height: int,
        width: int,
        low_second: Union[int, float],
        high_second: Union[int, float],
        per_grid_height: int,
        per_grid_width: int,
        seed: int,
):
    '''
    generate adjacency matrix about edge relation between every network node with rectangle shape.
    :param height: the number of row
    :param width: the number of column
    :param low_second: seconds, the lower bound of random experienced travel time
    :param high_second: seconds, the upper bound of random experienced travel time
    :param per_grid_height: per grid height
    :param per_grid_width: per grid width
    :param seed:
    :return:
    '''

    random.seed(seed)
    np.random.seed(seed)

    # generate adjacency matrix
    node = np.zeros((height, width))
    adjacency_matrix = np.zeros((height * width, height * width))
    for row in range(node.shape[0]):
        for col in range(node.shape[1]):
            adjacency_matrix_row = row * width + col
            try:
                _ = adjacency_matrix[adjacency_matrix_row, adjacency_matrix_row - 1]
                left = adjacency_matrix_row - 1
                if col == 0:
                    left = None
            except:
                left = None
            try:
                _ = adjacency_matrix[adjacency_matrix_row, adjacency_matrix_row + 1]
                right = adjacency_matrix_row + 1
                if col == width - 1:
                    right = None
            except:
                right = None
            try:
                _ = adjacency_matrix[adjacency_matrix_row, adjacency_matrix_row - width]
                up = adjacency_matrix_row - width
                if row == 0:
                    up = None
            except:
                up = None
            try:
                _ = adjacency_matrix[adjacency_matrix_row, adjacency_matrix_row + width]
                down = adjacency_matrix_row + width
                if row == height - 1:
                    down = None
            except:
                down = None
            for adjacency_matrix_col in (left, right, up, down):
                if adjacency_matrix_col is not None:
                    adjacency_matrix[adjacency_matrix_row, adjacency_matrix_col] = 1

    # generate random experienced travel time
    experienced_travel_time = copy.deepcopy(np.triu(adjacency_matrix))
    link_node = np.where(experienced_travel_time != 0)
    experienced_travel_time[link_node] = np.random.uniform(
        low=low_second, high=high_second, size=experienced_travel_time[link_node].shape
    )
    experienced_travel_time += experienced_travel_time.T - np.diag(
        experienced_travel_time.diagonal())
    experienced_travel_time[experienced_travel_time == 0] = float('inf')

    # generate grid
    grid = np.ones((int(height / per_grid_height + 0.999999999999), int(width / per_grid_width + 0.999999999999)))
    node_id_to_grid_id = pd.DataFrame({'node_id': list(range(height * width)), 'grid_id': [-1] * (height * width)})
    for row in range(height):
        for col in range(width):
            node_id = int(row * width + col)
            grid_row = int(row / per_grid_height)
            grid_col = int(col / per_grid_width)
            grid_id = int(grid_row * grid.shape[1] + grid_col)
            node_id_to_grid_id.loc[node_id, 'grid_id'] = grid_id

    return experienced_travel_time, node_id_to_grid_id


if __name__ == '__main__':

    experienced_travel_time, node_id_to_grid_id = generate_rectangle_network(
        height=10,
        width=15,
        low_second=30,
        high_second=300,
        per_grid_height=3,
        per_grid_width=4,
        seed=4000,
    )

    print(experienced_travel_time)
    print(node_id_to_grid_id)