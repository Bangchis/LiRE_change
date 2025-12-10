#!/bin/bash
# MetaWorld Best-Worst (BW) Feedback Experiment Script
# Based on MetaWorld.sh but optimized for BW feedback method

env=metaworld_button-press-topdown-v2   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
data_quality=1.0    # data quality.
                    # The lower the quality, the more random policy data, and the higher the quality, the more expert policy data. (maximum is 10.0)
feedback_num=500    # M = number of block queries (total feedback number)
                    # Note: BW generates ~M*(2K-2) pairwise comparisons from M blocks
q_budget=20         # K = block size (number of segments per block)
                    # For BW: recommended values are 10-30. Must be >= 2.
                    # This is different from RLT where q_budget means max segments per ranked list.
feedback_type=BW    # Best-Worst block feedback (new method)
model_type=BT       # ["BT", "linear_BT"]: BT means exponential bradley-terry model, and linear_BT use linear score function
lambda_bw=1.0       # Weight for worst constraints in BW feedback
                    # lambda_bw=1.0 means equal weight for "best > others" and "others > worst"
                    # Try values: 0.5, 1.0, 2.0 for experiments
epochs=300          # we use 300 epochs in the paper, but more epochs (e.g., 5000) can be used for better performance
activation=tanh     # final activation function of the reward model (use tanh for bounded reward)
seed=10             # random seed
threshold=0.5       # Thresholds for determining tie labels (equally preferred pairs)
                    # Larger thresholds result in more tie labels
                    # In BW: used when selecting best/worst from candidates with similar returns
segment_size=25     # segment size
data_aug=none       # ["none", "temporal"]: if you want to use data augmentation (TDA), set data_aug=temporal
batch_size=512
ensemble_num=3      # number of reward models to ensemble
ensemble_method=mean    # we average the reward values of the ensemble models
noise=0.0           # probability of preference labels (0.0 is noiseless label and 0.1 is 10% noise label)
human=False         # [True, False]: use human feedback or not


# Reward model learning with BW feedback
echo "================================================"
echo "Step 1: Training Reward Model with BW Feedback"
echo "================================================"
echo "Feedback: $feedback_type"
echo "Blocks (M): $feedback_num"
echo "Block size (K): $q_budget"
echo "Lambda_bw: $lambda_bw"
echo "Expected pairwise comparisons: ~$((feedback_num * (2 * q_budget - 2)))"
echo ""

CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method --batch_size=$batch_size \
--lambda_bw=$lambda_bw


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
--lambda_bw=$lambda_bw

echo ""
echo "================================================"
echo "Experiment Completed!"
echo "================================================"
