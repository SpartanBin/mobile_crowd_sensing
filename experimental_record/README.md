# experimental record



## experiment 1



### Environment

a 20 * 20 rectangle road network with grids containing 4 intersections(nodes)

#### timestep

- Define timestep is unit time interval(3 min). When pass one unit time interval, environment is to the next timestep. In per timestep, every vehicle needs to make the next direction to go, and environment decides the vehicles' final path according to vehicles' location, vehicles' decision, unit time interval, shortest path.
- In every timestep, vehicle may be at an intersection, or it may be on the road.

#### reward

- Environment has 20 * 20 / 4 number of grids. Every grid has random weight(reward). Sum of all grids' rewards is equal to 1.
- When a vehicle passes a grid, this grid's reward has been consumed.
- Vehicles get the same reward when execute action to next timestep. For example, vehicle A arrives at grid A with 0.02 weight which has no vehicle arrived at, and vehicle B arrives at grid B with 0.003 weight which has no vehicle arrived at, environment returning 0.02 + 0.003 point as final reward for every vehicle.

#### state

Use these as policy input.

- every vehicle's location and it's own location, such as [[15, 17, 16, 17], [-1, -1, 12, 13], [15, 17, 16, 17], [5, 7, 5, 8]] (as the second vehicle's input, 3 vehicles in total, array[:, 0: 2] representing vehicle's starting node, array[:, 2: 4] representing vehicle's ending node, array[1] representing the first vehicle's location, array[3] representing the third vehicle's location and so on, array[0] representing the second vehicle's location same as the array[2] because of this array regard as the second vehicle's input)

- current node's weight from grid's weight, with the same shape as road network, 20 * 20

#### action

- Have 4 directions at most, 2 direction at least to select due to vehicle's location.
- Environment calculates the shortest path to the nodes a vehicle can reach per unit time interval. Choose the node from nodes with smallest angle with direction made by vehicle as final node and it's path way as vehicle's path.



### model

- Use PPO as policy without sharing actor parameters with critic.
- Every vehicle shares one policy to control them.



### result

- not converging for many vehicles
- not converging for one vehicle



### discussion

possible problems

- Reward design may not lead to optimize our objective. Find the implementation of environment(maybe Atari games) with the objective maximizing scores in fixed time interval.
- Location state as input may needs to be encoded. Without encoded can express geographical relation. Or input sparse matrix  with 0 and 1, and same shape as road network.
- There may be some bug on implementation of environment or policy. Check it.
- There is hidden state and transformation between making action decision and final executed action which policy cannot learn it's rule.



## TODO

- [ ] Check environment code logic.
- [ ] Check policy code logic.
- [ ] Fix node weight to train.
- [ ] Change all format of input to matrix.
- [ ] Modify reward.