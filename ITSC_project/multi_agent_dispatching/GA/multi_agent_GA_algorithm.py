import sys
import os
from typing import Optional
import copy
from multiprocessing.dummy import Pool
import pickle

import numpy as np
import torch

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from ITSC_project.multi_agent_dispatching.common_utils import policies, multi_agent_control


def params_to_chrom(multi_agent_params):
    chrom = np.empty(0)
    for v in multi_agent_params.keys():
        for key in multi_agent_params[v].keys():
            if 'num_batches_tracked' not in key:
                chrom = np.append(chrom, multi_agent_params[v][key].cpu().numpy().flatten(), axis=-1)
    return chrom


def chrom_to_params(chrom, multi_agent_params):
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

    def __init__(self, pop_size, chrom_len):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.data = None
        self.best = None

    def select(self, fitness_array):
        raise NotImplementedError()

    def cross(self, cross_prob):
        raise NotImplementedError()

    def mutate(self, mutate_prob, progress):
        raise NotImplementedError()

    def __getitem__(self, index):
        return self.data[index].copy()

    def __setitem__(self, index, value):
        self.data[index] = value.copy()

    def _to_view(self, chrom):
        raise NotImplementedError()

    def view(self, index, bound):
        chrom = self._to_view(self.data[index])
        return (bound[1] - bound[0]) * chrom + bound[0]

    def view_best(self, bound):
        chrom = self._to_view(self.best)
        return (bound[1] - bound[0]) * chrom + bound[0]


class GenomeReal(Genome):

    def __init__(self, pop_size, chrom_len):
        super().__init__(pop_size, chrom_len)
        self.data = np.random.uniform(0, 1, size=(pop_size, chrom_len))

    def select(self, fitness_array):
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
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
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len), size=1)
                self.data[idx][mutate_position] += np.random.uniform(0, 1) * (
                    np.random.randint(2)-self.data[idx][mutate_position]) * (1-progress)**2

    def _to_view(self, chrom):
        return chrom


class GenomeBinary(Genome):

    def __init__(self, pop_size, chrom_len, code_len=16):
        super().__init__(pop_size, chrom_len)
        self.code_len = code_len
        self.data = np.random.random((pop_size, chrom_len*code_len)) < 0.5
        self.binary_template = np.zeros(code_len)
        for i in range(code_len):
            self.binary_template[i] = (2**i) / 2**code_len

    def select(self, fitness_array):
        indices = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
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
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(
                    np.arange(self.chrom_len*self.code_len), size=1)
                self.data[idx][mutate_position] = ~self.data[idx][mutate_position]

    def _to_view(self, chrom):
        return np.sum(chrom.reshape(self.chrom_len, self.code_len) * self.binary_template, axis=-1)


class multi_agent_GA(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        vehicle_num,
        link_weight_distribution,
        weight_shape,
        share_policy,
        conv_params,
        add_BN,
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
        :param GenomeClass: the code used by the genome
        :param cross_prob: crossover probability
        :param mutate_prob: mutation probability
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

        self.link_weight_distribution = link_weight_distribution

        self.weight_shape = weight_shape
        self.share_policy = share_policy
        self.conv_params = conv_params
        self.add_BN = add_BN
        self.output_dim = output_dim
        self.action_dim = action_dim

        self.policy = policies.multi_agent_ACP(
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            share_policy=self.share_policy,
            ortho_init=True,
            conv_params=self.conv_params,
            add_BN=self.add_BN,
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
        self.init_learn(train_link_weight_distribution=link_weight_distribution)
        self.update_fitness()
        self.best_fitness = 0
        self.update_records()

    def update_records(self):
        best_index = np.argmax(self.fitness_array)
        self.genome.best = self.genome[best_index]
        self.best_fitness = self.fitness_array[best_index]

    def replace(self):
        worst_index = np.argmin(self.fitness_array)
        self.genome[worst_index] = self.genome.best
        self.fitness_array[worst_index] = self.best_fitness

    def calculate_fitness(self, policy, env, new_obs, num_episodes_to_cal):

        fitness = []
        for _ in range(num_episodes_to_cal):
            done = False
            episode_time_cost = 0
            while not done:
                new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
                new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)
                with torch.no_grad():
                    loc_features = new_obs[0]
                    weight_features = new_obs[1]
                    distributions, _, _ = policy.forward(
                        loc_features=loc_features,
                        weight_features=weight_features,
                    )
                _, _, new_obs, _, done, episode_time_cost = self.make_one_step_forward_for_env(
                    env=env,
                    distributions=distributions,
                    episode_time_cost=episode_time_cost,
                )
            fitness.append(1 / episode_time_cost)
            new_obs = env.reset(link_weight_distribution=self.link_weight_distribution)

        mean_episode_time_cost = np.mean(1 / np.array(fitness))
        fitness = np.mean(fitness)

        return fitness, mean_episode_time_cost

    def update_fitness(self):

        new_obs = self.env.reset(link_weight_distribution=self.link_weight_distribution)
        episodes_time_cost = np.zeros(self.pop_size)
        episodes_time_cost[:] = np.inf
        pop_ids = list(range(self.pop_size))

        def calculate_fitness(idx):
            policy = copy.deepcopy(self.policy)
            multi_agent_params = chrom_to_params(
                chrom=self.genome.view(idx, self.bound),
                multi_agent_params=self.multi_agent_params,
            )
            policy.load_state_dict(multi_agent_params)
            self.fitness_array[idx], episode_time_cost = self.calculate_fitness(
                policy=policy,
                env=copy.deepcopy(self.env),
                new_obs=copy.deepcopy(new_obs),
                num_episodes_to_cal=self.num_episodes_to_cal,
            )
            episodes_time_cost[idx] = episode_time_cost

        pool_ = Pool()
        pool_.map(calculate_fitness, pop_ids)
        pool_.close()
        pool_.join()

        return episodes_time_cost.min()

    def result(self):
        return self.genome.view_best(self.bound)

    def genetic(self, num_gen):

        for e in range(num_gen):
            self.genome.select(self.fitness_array)
            self.genome.cross(self.cross_prob)
            self.genome.mutate(self.mutate_prob, progress=e/num_gen)
            episode_time_cost = self.update_fitness()
            cur_best_fitness = self.fitness_array.max()

            if episode_time_cost < self.the_best_last_100_episodes_mean_time_cost:
                self.the_best_last_100_episodes_mean_time_cost = episode_time_cost
            print('''
            ******************************************************************************************************
            in these {} episodes, the number of vehicle is {}, 
            time_mean_{}_episodes_time_cost = {}, 
            the_best_{}_episodes_mean_time_cost = {}
            ******************************************************************************************************
            '''.format(
                self.num_episodes_to_cal, self.vehicle_num,
                self.num_episodes_to_cal, episode_time_cost,
                self.num_episodes_to_cal, self.the_best_last_100_episodes_mean_time_cost
            ))

            if cur_best_fitness > self.best_fitness:
                self.replace()
                self.update_records()

            if e % 10 == 0:
                with open('genome_v{}.pickle'.format(self.vehicle_num), 'wb') as file:
                    pickle.dump(self.genome, file)