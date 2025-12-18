import numpy as np

import gym

import pyrallis
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import random, os, tqdm, copy, rich

import wandb
import uuid
from dataclasses import asdict, dataclass

import reward_utils
from reward_utils import collect_feedback, collect_human_feedback, consist_test_dataset
from reward_model import RewardModel

import sys

sys.path.append("./algorithms")
import utils_env


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_box-close-v2"  # environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    checkpoints_path: Optional[str] = None  # checkpoints path
    load_model: str = ""  # Model load file name, "" doesn't load
    # preference learning
    feedback_num: int = 1000
    data_quality: float = 5.0  # Replay buffer size (data_quality * 100000)
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.0
    data_aug: str = "none"
    q_budget: int = 10000
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    lambda_bw: float = 1.0  # Weight for worst constraints in BW feedback
    method_tag: str = "RLT"  # Method identifier for wandb logging: "RLT", "BW", or "BW_PL"
    human: bool = False
    # MLP
    epochs: int = int(1e3)
    batch_size: int = 256
    activation: str = "tanh"  # Final Activation function
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    # Wandb logging
    project: str = "Reward Learning"
    group: str = "Reward learning"
    name: str = "Reward"

    def __post_init__(self):
        # Clear naming with method tag for easy sorting and identification
        self.group = f"{self.method_tag}_{self.model_type}_{self.env}_data_{self.data_quality}_fn_{self.feedback_num}_qb_{self.q_budget}_n_{self.noise}"
        checkpoints_name = f"{self.name}/{self.env}/data_{self.data_quality}/fn_{self.feedback_num}/qb_{self.q_budget}/ft_{self.feedback_type}/m_{self.model_type}/n_{self.noise}/e_{self.epochs}/s_{self.seed}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, checkpoints_name
            )
            if not os.path.exists(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
        # Name includes seed and model type for clarity
        self.name = f"seed_{self.seed}_{self.model_type}"


def wandb_init(config: dict) -> None:
    wandb.init(
        mode="online",
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@pyrallis.wrap()
def train(config: TrainConfig):
    rich.print(config)
    reward_utils.set_seed(config.seed)

    if "metaworld" in config.env:
        env_name = config.env.replace("metaworld-", "")
        env = utils_env.make_metaworld_env(env_name, config.seed)
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        env_name = config.env.replace("dmc-", "")
        print("env_name ", env_name)
        env = utils_env.make_dmc_env(env_name, config.seed)
        dataset = utils_env.DMC_dataset(config)
        config.threshold *= 0.1  # because reward scaling is different from metaworld

    N = dataset["observations"].shape[0]
    traj_total = N // 500  # each trajectory has 500 steps

    if config.normalize:
        state_mean, state_std = reward_utils.compute_mean_std(
            dataset["observations"], eps=1e-3
        )
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = reward_utils.normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = reward_utils.normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    assert config.q_budget >= 1
    if config.human == False:
        multiple_ranked_list = collect_feedback(dataset, traj_total, config)
    elif config.human == True:
        multiple_ranked_list = collect_human_feedback(dataset, config)

    # ---- NEW: BW + PL (Plackett-Luce Algorithm 1) ----
    # If using PL models, train directly on blocks without pairwise expansion
    if config.feedback_type == "BW" and config.model_type in ["PL", "linear_PL"]:
        blocks = multiple_ranked_list  # list of (starts, best_pos, worst_pos, returns)
        K = config.q_budget
        T = config.segment_size

        block_obs_act = []
        best_pos_list = []
        worst_pos_list = []

        for starts, bpos, wpos, _ in blocks:
            # Build indices for all K segments in this block: [K, T]
            idx = [[j for j in range(k, k + T)] for k in starts]
            # Concatenate observations and actions: [K, T, obs_dim+act_dim]
            obs_act = np.concatenate(
                (dataset["observations"][idx], dataset["actions"][idx]),
                axis=-1
            )
            block_obs_act.append(obs_act)
            best_pos_list.append(bpos)
            worst_pos_list.append(wpos)

        block_obs_act = np.asarray(block_obs_act, dtype=np.float32)  # [M, K, T, D]
        best_pos = np.asarray(best_pos_list, dtype=np.int64)         # [M]
        worst_pos = np.asarray(worst_pos_list, dtype=np.int64)       # [M]

        # Create test set (pairwise, for evaluation)
        test_feedback_num = 5000
        test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels = (
            consist_test_dataset(
                dataset,
                test_feedback_num,
                traj_total,
                segment_size=config.segment_size,
                threshold=config.threshold,
            )
        )

        wandb_init(asdict(config))

        # Log dataset statistics for BW+PL
        num_blocks = len(blocks)
        block_size_K = config.q_budget
        wandb.log({
            'dataset/num_blocks': num_blocks,
            'dataset/block_size_K': block_size_K,
            'dataset/segments_per_query': block_size_K,  # K segments rated per query
            'dataset/feedback_type': config.feedback_type,
            'dataset/model_type': config.model_type,
        }, step=0)

        dimension = block_obs_act.shape[-1]

        # Create RewardModel with blocks (not pairs)
        reward_model = RewardModel(
            config,
            obs_act_1=None, obs_act_2=None, labels=None,
            dimension=dimension,
            weights=None,
            blocks=block_obs_act, best_pos=best_pos, worst_pos=worst_pos
        )
        reward_model.save_test_dataset(
            test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
        )
        reward_model.train_model()
        reward_model.save_model(config.checkpoints_path)
        return
    # ---- END NEW: BW + PL ----

    idx_st_1 = []
    idx_st_2 = []
    labels = []
    weights = []

    # Construct the preference pairs
    if config.feedback_type == "BW":
        # Best-Worst block feedback → expand to pairwise constraints (Algorithm 2)
        blocks = multiple_ranked_list  # Actually contains blocks, not ranked lists
        lam = config.lambda_bw  # Weight for worst constraints

        for starts, best_pos, worst_pos, returns in blocks:
            best_k = starts[best_pos]
            worst_k = starts[worst_pos]

            # Constraint 1: best > all others
            for k in starts:
                if k == best_k:
                    continue
                idx_st_1.append(k)         # worse segment
                idx_st_2.append(best_k)    # better segment
                labels.append([0, 1])
                weights.append(1.0)

            # Constraint 2: all others > worst
            for k in starts:
                if k == worst_k:
                    continue
                idx_st_1.append(worst_k)   # worse segment
                idx_st_2.append(k)         # better segment
                labels.append([0, 1])
                weights.append(lam)

        labels = np.asarray(labels, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)

    else:
        # Original RLT/SeqRank handling
        for single_ranked_list in multiple_ranked_list:
            sub_index_set = []
            for i, group in enumerate(single_ranked_list):
                for tup in group:
                    sub_index_set.append((tup[0], i, tup[1]))
            for i in range(len(sub_index_set)):
                for j in range(i + 1, len(sub_index_set)):
                    idx_st_1.append(sub_index_set[i][0])
                    idx_st_2.append(sub_index_set[j][0])
                    if sub_index_set[i][1] < sub_index_set[j][1]:
                        labels.append([0, 1])
                    else:
                        labels.append([0.5, 0.5])

        labels = np.array(labels)
        weights = np.ones(len(labels), dtype=np.float32)  # Uniform weights for non-BW
    idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
    idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
    obs_act_1 = np.concatenate(
        (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
    )
    obs_act_2 = np.concatenate(
        (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
    )
    # test query set (for debug the training, not used for training)
    test_feedback_num = 5000
    test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels = (
        consist_test_dataset(
            dataset,
            test_feedback_num,
            traj_total,
            segment_size=config.segment_size,
            threshold=config.threshold,
        )
    )

    wandb_init(asdict(config))

    # Log dataset statistics
    num_pairwise_samples = len(labels)
    if config.feedback_type == "BW":
        # For BW: log blocks and expansion ratio
        num_blocks = len(multiple_ranked_list)
        block_size_K = config.q_budget
        # Each block generates ~(2K-2) pairs: (K-1) for best + (K-1) for worst
        expected_pairs = num_blocks * (2 * block_size_K - 2)
        expansion_ratio = num_pairwise_samples / num_blocks if num_blocks > 0 else 0
        pairwise_per_query = num_pairwise_samples / config.feedback_num if config.feedback_num > 0 else 0
        wandb.log({
            'dataset/num_blocks': num_blocks,
            'dataset/block_size_K': block_size_K,
            'dataset/num_pairwise_samples': num_pairwise_samples,
            'dataset/expansion_ratio': expansion_ratio,
            'dataset/expected_pairs': expected_pairs,
            'dataset/pairwise_per_query': pairwise_per_query,  # Sample efficiency metric
            'dataset/feedback_type': config.feedback_type,
            'dataset/model_type': config.model_type,
        }, step=0)
    else:
        # For RLT/SeqRank: just log pairwise samples
        num_ranked_lists = len(multiple_ranked_list)
        pairwise_per_query = num_pairwise_samples / config.feedback_num if config.feedback_num > 0 else 0
        wandb.log({
            'dataset/num_ranked_lists': num_ranked_lists,
            'dataset/num_pairwise_samples': num_pairwise_samples,
            'dataset/pairwise_per_query': pairwise_per_query,  # Sample efficiency metric
            'dataset/feedback_type': config.feedback_type,
            'dataset/model_type': config.model_type,
        }, step=0)

    dimension = obs_act_1.shape[-1]
    reward_model = RewardModel(config, obs_act_1, obs_act_2, labels, dimension, weights=weights)

    reward_model.save_test_dataset(
        test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
    )

    reward_model.train_model()
    reward_model.save_model(config.checkpoints_path)


if __name__ == "__main__":
    train()
