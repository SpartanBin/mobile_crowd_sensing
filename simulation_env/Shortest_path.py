#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:32:42 2021

@author: zhexianli
"""

import pickle
import numpy as np
import heapq
import datetime

class PriorityQueue: #栈存储
    def __init__(self):
        self.elements = []
    
    def empty(self) -> bool:
        return len(self.elements) == 0
    
    def put(self, item : int, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def dijkstra_search(cost, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        neighbor = []
        for i in range(cost.shape[0]):
            if cost[current, i] < float('inf'):
                neighbor.append(i)
                
        for next in neighbor:
            new_cost = cost_so_far[current] + cost[current, next]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

if __name__ == "__main__":
    
    with open('experienced_travel_time.pickle', 'rb') as file:
        a = pickle.load(file)
    
    a[a == 0] = float('inf')
    
    time_stamp1 = datetime.datetime.now()
    came_from, cost_so_far = dijkstra_search(a, 0, 5)
    path = reconstruct_path(came_from, 0, 5)
    time_stamp2 = datetime.datetime.now()
    print(time_stamp2 - time_stamp1)