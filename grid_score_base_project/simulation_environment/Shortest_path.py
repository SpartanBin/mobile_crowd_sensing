#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:32:42 2021

@author: zhexianli
"""

import heapq

import numpy as np


class PriorityQueue:
    '''
    queue for saving
    '''

    def __init__(self):
        self.elements = []
    
    def empty(self) -> bool:
        return len(self.elements) == 0
    
    def put(self, item : int, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]


def dijkstra_search(cost, start, start_time, end_time, node_length):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    node_list = []
    
    while not frontier.empty():
        current = frontier.get()
        
        if cost_so_far[current] > start_time:
            if current not in node_list and came_from[current] not in node_list:
                node_list.append(current)
        
        if len(node_list) >= node_length or cost_so_far[current] > end_time:
            break

        cost_ = cost[current]
        neighbor = np.where(cost_ < float('inf'))[0].tolist()
                
        for next in neighbor:
            new_cost = cost_so_far[current] + cost[current, next]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, node_list


def reconstruct_path(came_from, start, goal):

    current = goal

    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional

    return path