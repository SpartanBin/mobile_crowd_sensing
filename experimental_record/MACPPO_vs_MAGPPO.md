# experimental record



## The same setting



### Environment 1: 20 * 20 rectangle network

#### timestep

- Define timestep is unit time interval(3 min). In per timestep, every vehicle needs to make the next direction to go, and environment decides the vehicles' final path according to vehicles' location, vehicles' decision, unit time interval, shortest path.
- In every timestep, vehicle may be at an intersection, or it may be on the road.

#### reward

- Environment has 20 * 20 / 4 number of grids. Every grid has random weight(reward) with uniform distribution. Sum of all grids' rewards is equal to 1. When reset environment, Every grid's weight is updated randomly and sum is also equal to 1.
- When a vehicle passes a grid, this grid's reward has been consumed.
- Vehicles get the same reward when execute action to next timestep. For example, vehicle A arrives at grid A with 0.02 weight which has no vehicle arrived at, and vehicle B arrives at grid B with 0.003 weight which has no vehicle arrives at, environment returning 0.02 + 0.003 point as final reward for every vehicle.
- Environment returning reward plus a constant negative value(- 0.05) as final reward.

#### action

- Have 4 directions at most, 2 direction at least to select due to vehicle's location.
- Environment calculates the shortest path to the nodes a vehicle can reach per unit time interval. Choose the node from nodes with smallest angle with direction made by vehicle as final node and it's path way as vehicle's path.



### Environment 2: Longquan real road network

See [lq](../grid_score_base_project/lq_network_data/node_base_features/) for the information of grids, nodes and edges.

#### timestep

same to environment 1

#### reward

- The rewards returned are the rewards that each vehicle gets on its own (greedy reward), not sum of all.
- See [grid_score_han](./grid_score_han.pdf) for details.

#### action

- There are 8 directions for vehicles to choose.
- Others are same to environment 1.



### Environment 3: Chengdu real road network

See [cd](../grid_score_base_project/cd_network_data/) for the information of grids, nodes and edges.

#### timestep

same to environment 1

#### reward

same to environment 2

#### action

same to environment 2



## Experiment 1 GNN pooling + node embedding layers test



### state features

- The number of GNN nodes is equal to the network. 
- Features include a binary vector representing the position of an independent agent, a vector representing the position of all agents, a vector representing the grid score,  a vector representing coordinate x, and a vector representing coordinate y, five in total. 



### model

- Use PPO as optimization method and policy does not share parameters with critic.
- Every vehicle shares one policy to control them.
- Use GNN as policy and value model, GNN structures:
- - model 1: GCNConv + Global Attention
  - model 2: GCNConv + Set2Set
  - model 3: GraphSAGE + Global Attention
  - model 4: GraphSAGE + Set2Set



### result

Environment 2 (two vehicles): random policy 1.810. 

| model type | best train session | best performance | final performance | remark |
| ---------- | ------------------ | ---------------- | ----------------- | ------ |
| model 1    | 1894               | 3.121            | 3.115             |        |
| model 2    | 1920               | 3.162            | 3.138             |        |
| model 3    | 1925               | 3.212            | 3.189             |        |
| model 4    | 1905               | 2.891            | 2.887             |        |



### discussion



## Experiment 2 other network structure and feature engineering



### state features

Basically the same as experiment 1, there are some changes depending on the model, and explain as follow.



### model

The content in [] is the input feature. Environment 1 use DQN as optimization method. Environment 2 use PPO as optimization method. 

- model 1: MADQN
- model 2: CNN concatenating CNN to CNN to DNN [binary individual agent location vector, binary all loc vector + grid score vector]
- model 3: (CNN to DNN) concatenating (CNN to DNN) to DNN [same to model 2]
- model 4: DNN concatenating  (CNN to DNN) to DNN [same to model 2]
- model 5: same to model 4 [ID individual agent location, binary all loc vector + grid score vector]
- model 6: same to model 4 [ID individual agent location + ID all location, grid score vector]
- model 7: DNN concatenating DNN to DNN [same to model 6]
- model 8: DNN [ID individual agent location + ID all location + grid score vector]



### result

Environment 1 (two vehicles): random policy 0.154. Use DQN. 

| model type                                                   | best train session | best performance | final performance | remark      |
| ------------------------------------------------------------ | ------------------ | ---------------- | ----------------- | ----------- |
| model 1                                                      |                    | 0.180            |                   |             |
| model 2                                                      | 8524               | 0.183            | 0.169             |             |
| model 3                                                      | 4463               | 0.188            | 0.170             |             |
| model 4                                                      | 2724               | 0.173            | 0.162             |             |
| model 5                                                      | 36733              | 0.189            | 0.171             |             |
| model 6                                                      | 101                | 0.189            | 0.166             | instability |
| model 7                                                      | 459                | 0.185            | 0.167             | instability |
| model 8 ([ID_allID_DNN_structure1](../grid_score_base_rectangular_network_project/multi_agent_dispatching/common_utils/model.py)) | 79694              | 0.210            | 0.180             |             |

Environment 2 (two vehicles): random policy 1.810. Use PPO. 

| model type                                                   | best train session | best performance | final performance | remark |
| ------------------------------------------------------------ | ------------------ | ---------------- | ----------------- | ------ |
| model 1                                                      | 1784               | 3.233            |                   |        |
| model 8 ([ID_allID_DNN_structure1](../grid_score_base_project/multi_agent_dispatching/MACPPO/model.py)) | 1938               | 2.952            |                   |        |



## TODO

- [ ] Try MAGPPO model with structure GraphSAGE + Global Attention and feature engineering representing vehicles' direction.
