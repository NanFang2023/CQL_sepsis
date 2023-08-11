import torch
import numpy as np
import pickle
import pandas as pd
from copy import deepcopy
import absl.app
import absl.flags
import sys
from conservative_sac import ConservativeSAC
from replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from sampler import StepSampler, TrajSampler
from utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from utils import WandBLogger
from viskit.logging_ import logger, setup_logger
import pickle
import pandas as pd

import random
def get_sepsis_dataset_val(eval_n_trajs,deterministic=False, replay_buffer=None):
    dataset_path = f'/home/fn/Mynew_Spesis/sepsisrl-master/Decision_trans/decision-transformer-master/gym/my_cql_new_val_df.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    trajectories = random.sample(trajectories,eval_n_trajs)

    return trajectories

def sample(valpath,policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for i in range(n_trajs):
            observations = []
            actions_cql = []
            actions_py = []
            rewards = []
            next_observations = []
            dones = []

            observation = valpath[i]['observations']
            action = valpath[i]['actions']
            reward = valpath[i]['rewards']
            done = valpath[i]['terminals']
            length = len(observation)
            # print(observation[0]) # observation数据类型有问题
            for j in range(length):
                action_cql = policy(
                    np.expand_dims(np.array(observation[j]), 0), deterministic=deterministic
                )[0, :]
                if j < length-1:
                    next_observation = observation[j+1]
                    observations.append(observation[j])
                    actions_cql.append(action_cql.tolist())
                    actions_py.append(action[j])
                    rewards.append(reward[j])
                    dones.append(done[j])
                    next_observations.append(next_observation)

                if done[j]:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions_py=np.array(actions_py, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                actions_cql=np.array(actions_cql, dtype=np.float32),
            ))

        return trajs

observation_dim = 48
action_dim = 2
policy_arch = '256-256'
policy_log_std_multiplier = 1.0
policy_log_std_offset = -1.0
orthogonal_init = False
qf_arch = '256-256'
cql=ConservativeSAC.get_default_config()
policy = TanhGaussianPolicy(
    observation_dim,
    action_dim,
    arch=policy_arch,
    log_std_multiplier=policy_log_std_multiplier,
    log_std_offset=policy_log_std_offset,
    orthogonal_init=orthogonal_init,
)

qf1 = FullyConnectedQFunction(
    observation_dim,
    action_dim,
    arch=qf_arch,
    orthogonal_init=orthogonal_init,
)
target_qf1 = deepcopy(qf1)

qf2 = FullyConnectedQFunction(
    observation_dim,
    action_dim,
    arch=qf_arch,
    orthogonal_init=orthogonal_init,
)
target_qf2 = deepcopy(qf2) 
sac = ConservativeSAC(cql, policy, qf1, qf2, target_qf1, target_qf2)
sac.torch_to_device('cuda')
sac = torch.load("/home/fn/Mynew_Spesis/sepsisrl-master/CQL/CQL-master/d4rl/examples/My_model/cql_model.pt")

eval_n_trajs = 5
max_traj_length=1000
trajs  = get_sepsis_dataset_val(eval_n_trajs)  # 获得数据
sampler_policy = SamplerPolicy(policy, 'cuda')
trajs = sample(trajs,sampler_policy, eval_n_trajs, deterministic=True)