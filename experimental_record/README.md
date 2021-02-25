# experimental record



## experiment 1



### Environment

- a 20 * 20 rectangle road network with grids containing 4 intersections(nodes)
- Episode over conditions: get more than 60% weight(reward) in network or episode duration time exceeds 3600 * 24 * 4 seconds.

#### timestep

- Define timestep is unit time interval(3 min). When pass one unit time interval, environment is to the next timestep. In per timestep, every vehicle needs to make the next direction to go, and environment decides the vehicles' final path according to vehicles' location, vehicles' decision, unit time interval, shortest path.
- In every timestep, vehicle may be at an intersection, or it may be on the road.

#### reward

- Environment has 20 * 20 / 4 number of grids. Every grid has random weight(reward). Sum of all grids' rewards is equal to 1. When reset environment, Every grid's weight is updated randomly and sum is also equal to 1.
- When a vehicle passes a grid, this grid's reward has been consumed.
- Vehicles get the same reward when execute action to next timestep. For example, vehicle A arrives at grid A with 0.02 weight which has no vehicle arrived at, and vehicle B arrives at grid B with 0.003 weight which has no vehicle arrived at, environment returning 0.02 + 0.003 point as final reward for every vehicle.
- Environment returning reward plus a constant negative value(- 0.05) as final reward.

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
- Vehicle's location features' input to DNN input layers and node's weight input to CNN input layers, concatenating two input layers results as final output DNN layers input.



### result

- Model cannot converge in one vehicle condition.
- Model cannot converge in many vehicles condition.



### discussion

possible problems

- Reward design may not lead to optimize our objective. Find the implementation of environment(maybe Atari games) with the objective maximizing scores in fixed time interval.
- Location state as input may needs to be encoded. Without encoded can express geographical relation. Or input sparse matrix  with 0 and 1, and same shape as road network.
- There may be some bug on implementation of environment or policy. Check it.
- There is hidden state and transformation between making action decision and final executed action which policy cannot learn it's rule.



## experiment 2



### Environment

same to experiment 1

#### timestep

same to experiment 1

#### reward

- same to experiment 1
- Fix node weight(in every reset, node weights are the same).

#### state

Use these as policy input.

- Node's weight is same to experiment 1.

- Change location state features to matrix with the same shape like Node's weight. Use two sparse matrix to respectively represent corresponding vehicle's location and all vehicles' location. If vehicle's location on node (13, 14), matrix row 14 column 15 is equal to 1, other no vehicle's location is equal to 0. Matrix one only marks one corresponding vehicle's location and Matrix two marks all vehicle's location.

#### action

same to experiment 1



### model

- RL model is the same to experiment 1.
- Use only CNN as input layers receiving all features, and DNN as output layers.



### result

- Model converged to average 10000 seconds cost until episode over in one vehicle condition(in random policy, finishing an episode needs about 90000 seconds cost).



### discussion

- Results proves experiment 1 has wrong feature engineering.
- Then try same model on the dynamic node weight.



## experiment 3



### Environment

same to experiment 2

#### timestep

same to experiment 2

#### reward

same to experiment 1

#### state

same to experiment 2

#### action

same to experiment 2



### model

same to experiment 2



### result

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 1                      | 14929            | 92967             | 10720                    |
| 2                      | 12825            | 45137             | 7468                     |
| 4                      | 9280             | 17481             | 4099                     |
| 10                     | 3163             | 4884              | 1512                     |
| 20                     | 1492             | 1671              | 642                      |

Last 100 episode: the last 100 episodes mean time cost(seconds), could represent the performance of trained policy.

First 100 episode: the first 100 episodes mean time cost(seconds), could represent the performance of random policy.

The shortest 100 episode: the shortest 100 episodes mean time cost(seconds), could represent the best performance of policy.



### discussion

- In one vehicle condition, model can find a well policy, but the more vehicles there are, the harder it is to find a better policy.

possible problems

- Reward design may lead to this result. When step to next timestep, all vehicle receive the same reward. It maybe causes vehicle to confuse the value of its own behavior with that of other vehicles. Try to give a greedy reward and a corporative reward to vehicle.



## experiment 4



### Environment

same to experiment 3

#### timestep

same to experiment 3

#### reward

- Return reward swatch to every vehicle's own rewards earned(greedy reward), not sum of all.
- Others are same to experiment 3.

#### state

same to experiment 3

#### action

same to experiment 3



### model

same to experiment 3



### result

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 2                      | 13657            | 43528             | 8046                     |
| 4                      | 7662             | 17626             | 3971                     |
| 10                     | 2955             | 4885              | 1492                     |
| 20                     | 1354             | 1672              | 614                      |



### discussion

- When number of vehicles is more than 2, optimal policy is better than that in returning same reward(experiment 3). This result maybe partially proved same reward causes vehicle to confuse the value of its own behavior with that of other vehicles. Try to give a greedy reward(help to distinguish the value of its own behavior from that of other vehicles) plus the mean of all greedy reward(encourage collaboration).



## experiment 5



### Environment

same to experiment 4

#### timestep

same to experiment 4

#### reward

- Return reward swatch to every vehicle's own rewards earned(greedy reward) plus mean of all.
- Others are same to experiment 4.

#### state

same to experiment 4

#### action

same to experiment 4



### model

same to experiment 4



### result

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 2                      | 11990            | 42730             | 7249                     |
| 4                      | 7460             | 16961             | 4167                     |
| 10                     | 3105             | 4885              | 1525                     |
| 20                     | 1315             | 1672              | 610                      |

There is another notable result below when change constant negative reward value to -0.005.

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 2                      | 12406            | 42195             | 7207                     |
| 4                      | 9173             | 18561             | 5040                     |
| 10                     | 3293             | 5088              | 1661                     |
| 20                     | 1419             | 1672              | 702                      |



### discussion

- It is obviously that own rewards plus mean can promote policy, but when the number of vehicles is 10, policy becomes weak. This phenomenon indicates that the result may be leaded by randomness, but it's worth trying again this reward type with more weight(in this experiment, mean weight is equal to 1).
- Another notable result indicates that constant negative reward value could affect the performance of the policy, which is a parameter that needs to be adjusted.



## experiment 6



### Environment

same to experiment 5

#### timestep

same to experiment 5

#### reward

- Return reward swatch to every vehicle's own rewards earned(greedy reward) plus the weighted mean of all. Try the case where the weights are equal to 0.5 and 1.5.
- Others are same to experiment 5.

#### state

same to experiment 5

#### action

same to experiment 5



### model

same to experiment 5



### result

result with 0.5 weight

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 2                      | 11670            | 45706             | 7115                     |
| 4                      | 6779             | 17585             | 3971                     |
| 10                     | 3165             | 4885              | 1463                     |
| 20                     | 1260             | 1672              | 538                      |

result with 1.5 weight

| the number of vehicles | last 100 episode | first 100 episode | the shortest 100 episode |
| ---------------------- | ---------------- | ----------------- | ------------------------ |
| 2                      |                  |                   |                          |
| 4                      |                  |                   |                          |
| 10                     |                  |                   |                          |
| 20                     |                  |                   |                          |



### discussion



## TODO

- [x] Check environment code logic.
- [x] Check policy code logic.
- [x] Fix node weight to train.
- [x] Change all format of input to matrix.
- [x] Try greedy reward.
- [x] Try greedy reward plus the mean of all greedy reward.
- [ ] Try greedy reward plus the weighted mean of all greedy reward.
- [ ] ~~Try genetic algorithm(if work, then try imitation learning to learn genetic policy).~~
- [ ] Contrast to Genetic Algorithm.
- [ ] Try more constant negative reward value.
- [ ] Try distance weighted reward.