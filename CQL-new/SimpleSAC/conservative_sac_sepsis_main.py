import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint
import random
import gym
import torch
import d4rl

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

FLAGS_DEF = define_flags_with_default(
    env='EnvSepsis-v1',
    max_traj_length=1000, #1000
    seed=42,
    device='cuda',
    save_model=True,
    batch_size=100,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch= '64-64',     #'256-256',
    qf_arch='64-64',   #'256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000, #1000
    bc_epochs=0,
    n_train_step_per_epoch=1000, #1000
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

def get_normalized_score(score):
    ref_min_score = -15
    ref_max_score = 15
    return (score - ref_min_score) / (ref_max_score - ref_min_score)

def get_sepsis_dataset_train():
    dataset_path = f'/home/fn/Mynew_Spesis/sepsisrl-master/Decision_trans/decision-transformer-master/gym/my_cql_new_df.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    trajectories['observations'] = trajectories['observations'].astype(np.float32)
    trajectories['rewards'] = (trajectories['rewards'].astype(np.float32)+15)/30
    trajectories['next_observations'] = trajectories['next_observations'].astype(np.float32)
    trajectories['actions'] = trajectories['actions'].astype(np.float32)
    trajectories['dones'] = trajectories['dones'].astype(np.float32)
    return trajectories

def get_sepsis_dataset_val(eval_n_trajs,deterministic=False, replay_buffer=None):
    dataset_path = f'/home/fn/Mynew_Spesis/sepsisrl-master/Decision_trans/decision-transformer-master/gym/my_cql_new_val_df.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    trajectories = random.sample(trajectories,eval_n_trajs)

    return trajectories

def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    # load_data_df(eval_sampler.env) # 加载数据

    dataset = get_sepsis_dataset_train()
    #dataset_val = get_sepsis_dataset_val(FLAGS.eval_n_trajs)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    
    observation_dim = 48
    action_dim = 2

    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        observation_dim,
        action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        observation_dim,
        action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(action_dim).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}

    batch = subsample_batch(dataset, FLAGS.batch_size) # 100
    batch = batch_to_torch(batch, FLAGS.device)
    
    for epoch in range(FLAGS.n_epochs): # 1000
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            
            for batch_idx in range(FLAGS.n_train_step_per_epoch): # 1000
                # batch = subsample_batch(dataset, FLAGS.batch_size) # 100
                # batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

        # with Timer() as eval_timer:
        #     if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
        #         trajs = get_sepsis_dataset_val(FLAGS.eval_n_trajs)  # 获得数据
        #         trajs = eval_sampler.sample(
        #             sampler_policy, FLAGS.eval_n_trajs, deterministic=True
        #         )
        #         metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
        #         metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
        #         metrics['average_normalizd_return'] = np.mean(
        #             [get_normalized_score(np.sum(t['rewards'])) for t in trajs]
        #         )
        #         if FLAGS.save_model:
        #             save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        #             wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        # metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() #+ eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model_test.pkl')
        torch.save(sac,'/home/fn/Mynew_Spesis/sepsisrl-master/CQL/CQL-master/d4rl/examples/My_model/cql_model.pt')

if __name__ == '__main__':
    absl.app.run(main)
