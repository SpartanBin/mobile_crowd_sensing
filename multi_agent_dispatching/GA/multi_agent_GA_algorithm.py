import sys
import os
from typing import Optional
import copy

import numpy as np
import torch

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from multi_agent_dispatching.common_utils import policies, multi_agent_control


def params_to_chrom(multi_agent_params):
    """ 模型参数转码为染色体 """
    chrom = np.empty(0)
    for v in multi_agent_params.keys():
        for key in multi_agent_params[v].keys():
            if 'num_batches_tracked' not in key:
                chrom = np.append(chrom, multi_agent_params[v][key].cpu().numpy().flatten(), axis=-1)
    return chrom


def chrom_to_params(chrom, multi_agent_params):
    """ 染色体转码为模型参数（需参数模版） """
    multi_agent_params_c = copy.deepcopy(multi_agent_params)
    idx = 0
    for v in multi_agent_params_c.keys():
        for key in multi_agent_params_c[v].keys():
            if 'num_batches_tracked' not in key:
                param_length = int(np.prod(multi_agent_params[v][key].shape))
                param = torch.from_numpy(chrom[idx: idx+param_length].reshape(multi_agent_params[v][key].shape))
                multi_agent_params_c[v][key] = param
                idx += param_length
    return multi_agent_params_c


class Genome():
    """ 种群基因组，用于管理整个种群的染色体，数据分布为 U(0, 1)

    Args:
        pop_size: 种群规模
        chrom_len: 染色体长度

    Attributes:
        pop_size: 种群规模
        chrom_len: 染色体长度
        data: 基因组数据
        best: 最佳染色体
    """

    def __init__(self, pop_size, chrom_len):
        """ 初始化种群 """
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.data = None
        self.best = None

    def select(self, fitness_array):
        """ 选择 """
        raise NotImplementedError()

    def cross(self, cross_prob):
        """ 交叉 """
        raise NotImplementedError()

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        raise NotImplementedError()

    def __getitem__(self, index):
        """ 直接获取内部数据 """
        return self.data[index].copy()

    def __setitem__(self, index, value):
        """ 直接修改内部数据 """
        self.data[index] = value.copy()

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        raise NotImplementedError()

    def view(self, index, bound):
        """ 获取某一项编码数据的真实分布数据 """
        chrom = self._to_view(self.data[index])
        return (bound[1] - bound[0]) * chrom + bound[0]

    def view_best(self, bound):
        """ 获取最佳编码数据的真实分布数据 """
        chrom = self._to_view(self.best)
        return (bound[1] - bound[0]) * chrom + bound[0]


class GenomeReal(Genome):
    """ 实值编码基因组 """

    def __init__(self, pop_size, chrom_len):
        super().__init__(pop_size, chrom_len)
        self.data = np.random.uniform(0, 1, size=(pop_size, chrom_len))

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(
                    np.delete(np.arange(self.pop_size), idx), size=1)
                cross_points = np.random.random(
                    self.chrom_len) < np.random.rand()
                cross_rate = np.random.rand()
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    (1-cross_rate) * self.data[idx, cross_points] + cross_rate * self.data[idx_other, cross_points], \
                    (1-cross_rate) * self.data[idx_other, cross_points] + \
                    cross_rate * self.data[idx, cross_points]

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len), size=1)
                self.data[idx][mutate_position] += np.random.uniform(0, 1) * (
                    np.random.randint(2)-self.data[idx][mutate_position]) * (1-progress)**2

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        return chrom


class GenomeBinary(Genome):
    """ 二进制编码基因组 """

    def __init__(self, pop_size, chrom_len, code_len=16):
        super().__init__(pop_size, chrom_len)
        self.code_len = code_len
        self.data = np.random.random((pop_size, chrom_len*code_len)) < 0.5
        self.binary_template = np.zeros(code_len)
        for i in range(code_len):
            self.binary_template[i] = (2**i) / 2**code_len

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(
                    np.delete(np.arange(self.pop_size), idx), size=1)
                cross_points = np.random.random(
                    self.chrom_len*self.code_len) < np.random.rand()
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    self.data[idx_other,
                              cross_points], self.data[idx, cross_points]

    def mutate(self, mutate_prob, progress):
        """ 突变 """
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len*self.code_len), size=1)
                self.data[idx][mutate_position] = ~self.data[idx][mutate_position]

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式，分布仍为 U(0, 1) """
        return np.sum(chrom.reshape(self.chrom_len, self.code_len) * self.binary_template, axis=-1)


class multi_agent_GA(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        vehicle_num,
        weight_shape,
        share_policy,
        conv_params,
        output_dim,
        action_dim,
        num_episodes_to_cal,

        pop_size,
        bound: Optional[np.ndarray] = None,
        GenomeClass=GenomeBinary,
        cross_prob=0.8,
        mutate_prob=0.03,

        seed: int = 4000,
    ):
        """

        :param vehicle_num:
        :param weight_shape:
        :param share_policy:
        :param conv_params:
        :param output_dim:
        :param action_dim:
        :param pop_size: population size
        :param bound: model parameters' lower bound and upper bound
        :param GenomeClass: 基因组所使用的编码（实值编码：GenomeReal，二进制编码：GenomeBinary）
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :return:
        """

        super(multi_agent_GA, self).__init__(
            env=env,
            reward_type='greedy',
            cooperative_weight=0,
            negative_constant_reward=0,
            vehicle_num=vehicle_num,
            seed=seed,
            device="cpu",
        )

        self.weight_shape = weight_shape
        self.share_policy = share_policy
        self.weight_feature_params = conv_params
        self.output_dim = output_dim
        self.action_dim = action_dim

        self.policy = policies.multi_agent_ACP(
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            share_policy=self.share_policy,
            ortho_init=True,
            conv_params=self.weight_feature_params,
            output_dim=self.output_dim,
            share_params=False,
            action_dim=self.action_dim,
            learning_rate=0.0001,
        ).eval()

        self.multi_agent_params = copy.deepcopy(self.policy.state_dict())
        self.num_episodes_to_cal = num_episodes_to_cal

        chrom_len = len(params_to_chrom(self.multi_agent_params))
        self.pop_size = pop_size
        self.bound = bound
        if self.bound is None:
            self.bound = np.zeros((2, chrom_len))
            self.bound[0] = -1
            self.bound[1] = 1
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob

        self.fitness_array = np.zeros(pop_size)
        self.genome = GenomeClass(pop_size=pop_size, chrom_len=chrom_len)
        self.learn()
        self.update_fitness()
        self.best_fitness = 0
        self.update_records()

    def update_records(self):
        """ 更新最佳记录 """
        best_index = np.argmax(self.fitness_array)
        self.genome.best = self.genome[best_index]
        self.best_fitness = self.fitness_array[best_index]

    def replace(self):
        """ 使用前代最佳替换本代最差 """
        worst_index = np.argmin(self.fitness_array)
        self.genome[worst_index] = self.genome.best
        self.fitness_array[worst_index] = self.best_fitness

    def calculate_fitness(self, chrom, multi_agent_params, env, new_obs, num_episodes_to_cal):

        multi_agent_params = chrom_to_params(chrom, multi_agent_params)
        self.policy.load_state_dict(multi_agent_params)

        fitness = []
        for _ in range(num_episodes_to_cal):
            done = False
            episode_time_cost = 0
            while not done:
                new_obs = list(new_obs)
                new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
                new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)
                with torch.no_grad():
                    loc_features = torch.as_tensor(new_obs[0])
                    weight_features = torch.as_tensor(new_obs[1])
                    distributions, _, _ = self.policy.forward(
                        loc_features=loc_features,
                        weight_features=weight_features,
                    )
                _, _, new_obs, _, done, episode_time_cost = self.make_one_step_forward_for_env(
                    env=env,
                    distributions=distributions,
                    episode_time_cost=episode_time_cost,
                )
            fitness.append(1 / episode_time_cost)
            new_obs = env.reset()

        mean_episode_time_cost = np.mean(1 / np.array(fitness))
        fitness = np.mean(fitness)

        return fitness, mean_episode_time_cost

    def update_fitness(self):
        """ 重新计算适应度 """
        new_obs = self.env.reset()
        episodes_time_cost = []
        for idx in range(self.pop_size):
            self.fitness_array[idx], episode_time_cost = self.calculate_fitness(
                chrom=self.genome.view(idx, self.bound),
                multi_agent_params=self.multi_agent_params,
                env=copy.deepcopy(self.env),
                new_obs=copy.deepcopy(new_obs),
                num_episodes_to_cal=self.num_episodes_to_cal,
            )
            episodes_time_cost.append(episode_time_cost)
        return np.min(episodes_time_cost)

    def genetic(self, num_gen):
        """ 开始运行遗传算法 """

        for e in range(num_gen):
            self.genome.select(self.fitness_array)
            self.genome.cross(self.cross_prob)
            self.genome.mutate(self.mutate_prob, progress=e/num_gen)
            episode_time_cost = self.update_fitness()
            cur_best_fitness = self.fitness_array.max()

            self.the_last_100_episodes_time_cost.append(episode_time_cost)
            self.the_shortest_100_episodes_time_cost.append(episode_time_cost)
            last_100_episodes_mean_time_cost = np.mean(self.the_last_100_episodes_time_cost)
            if len(self.the_last_100_episodes_time_cost) > 100 / self.num_episodes_to_cal:
                if last_100_episodes_mean_time_cost < self.the_best_last_100_episodes_mean_time_cost:
                    self.the_best_last_100_episodes_mean_time_cost = last_100_episodes_mean_time_cost
                if self.the_first_100_episodes_mean_time_cost is None:
                    self.the_first_100_episodes_mean_time_cost = last_100_episodes_mean_time_cost
                self.the_last_100_episodes_time_cost.pop(0)
                self.the_shortest_100_episodes_time_cost.sort()
                self.the_shortest_100_episodes_time_cost.pop()
            print('''
            ******************************************************************************************************
            in these {} episodes, the number of vehicle is {}, 
            time_mean_{}_episodes_time_cost = {}, 
            the_shortest_100_episodes_mean_time_cost = {}, 
            the_first_100_episodes_mean_time_cost = {}, 
            the_last_100_episodes_mean_time_cost = {}, 
            the_best_last_100_episodes_mean_time_cost = {}
            ******************************************************************************************************
            '''.format(
                self.num_episodes_to_cal, self.vehicle_num,
                self.num_episodes_to_cal, episode_time_cost,
                np.mean(self.the_shortest_100_episodes_time_cost),
                self.the_first_100_episodes_mean_time_cost,
                last_100_episodes_mean_time_cost,
                self.the_best_last_100_episodes_mean_time_cost
            ))

            if cur_best_fitness > self.best_fitness:
                self.replace()
                self.update_records()

    def result(self):
        """ 输出最佳染色体 """
        return self.genome.view_best(self.bound)