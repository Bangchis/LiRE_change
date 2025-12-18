#!/bin/bash
# MetaWorld Best-Worst (BW) + Plackett-Luce (PL) Experiment Script
# Uses BW blocks but trains reward model directly with Plackett-Luce loss (no pairwise expansion)

env=${ENV_NAME:-metaworld_button-press-topdown-v2}   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
data_quality=1.0    # data quality.
                    # The lower the quality, the more random policy data, and the higher the quality, the more expert policy data. (maximum is 10.0)
feedback_num=500    # M = number of block queries (total feedback number)
q_budget=20         # K = block size (number of segments per block)
                    # For BW: recommended values are 10-30. Must be >= 2.
                    # This is different from RLT where q_budget means max segments per ranked list.
feedback_type=BW    # Best-Worst block feedback

# --- IMPORTANT: PL model types (no pairwise expansion) ---
model_type=${MODEL_TYPE:-PL}       # ["PL", "linear_PL"]
                    # "PL": Exponential Plackett-Luce (f = exp(s))
                    # "linear_PL": Linear Plackett-Luce (f = s + const)
                    # Both train directly on blocks using Algorithm 1 (no pairwise expansion)
method_tag=${METHOD_NAME:-BW_PL}   # Method identifier for wandb logging

lambda_bw=1.0       # λ in Algorithm 1: weight for worst term
                    # lambda_bw=1.0 means equal weight for best and worst terms
                    # Try values: 0.5, 1.0, 2.0 for experiments
epochs=300          # we use 300 epochs in the paper, but more epochs (e.g., 5000) can be used for better performance
activation=tanh     # final activation function of the reward model (use tanh for bounded reward)
seed=10             # random seed
threshold=0.5       # Thresholds for determining tie labels when selecting best/worst
                    # Used only for tie-handling when choosing best/worst inside a block
segment_size=25     # segment size
data_aug=none       # ["none", "temporal"]: if you want to use data augmentation (TDA), set data_aug=temporal
batch_size=512      # batch size (number of blocks per batch for PL training)
ensemble_num=3      # number of reward models to ensemble
ensemble_method=mean    # we average the reward values of the ensemble models
noise=0.0           # probability of preference labels (0.0 is noiseless label and 0.1 is 10% noise label)
human=False         # [True, False]: use human feedback or not


# Reward model learning with BW + Plackett-Luce
echo "================================================"
echo "Step 1: Training Reward Model with BW + Plackett-Luce"
echo "================================================"
echo "Feedback: $feedback_type"
echo "Model: $model_type (Plackett-Luce)"
echo "Blocks (M): $feedback_num"
echo "Block size (K): $q_budget"
echo "Lambda_bw (λ): $lambda_bw"
echo ""
echo "NOTE: PL trains directly on blocks → no pairwise expansion."
echo "      This is more sample-efficient than Bradley-Terry with pairwise expansion."
echo ""

CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method --batch_size=$batch_size \
--lambda_bw=$lambda_bw --method_tag=$method_tag


# Offline IQL with learned reward model
echo ""
echo "================================================"
echo "Step 2: Training IQL with Learned Reward Model"
echo "================================================"
echo ""

CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--method_tag=$method_tag

echo ""
echo "================================================"
echo "Experiment Completed!"
echo "================================================"
