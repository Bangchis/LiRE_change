#!/bin/bash
# =============================================================================
# Heap (Tournament Max-Heap Forest) Experiment Script
# Chay tat ca to hop K, model_type, seed tren nhieu moi truong
# Su dung file YAML config rieng thay vi truyen CLI args dai
# =============================================================================

for env in metaworld_button-press-topdown-v2 metaworld_dial-turn-v2 metaworld_sweep-v2; do
  for K in 5 10; do
    for seed in 10 20 30; do

      # --- Heap + BT (pairwise expansion) ---
      echo "================================================"
      echo "Heap+BT | Env: $env | K=$K | seed=$seed"
      echo "================================================"

      CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py \
        --config=configs/reward_heap_BT.yaml \
        --env=$env --seed=$seed --q_budget=$K --checkpoints_path=logs/

      CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py \
        --use_reward_model=True --config=configs/iql.yaml \
        --env=$env --seed=$seed --q_budget=$K \
        --feedback_type=heap --model_type=BT --feedback_num=500 \
        --max_timesteps=250_000 --eval_freq=5_000

      # --- Heap + linear_BT (pairwise expansion, linear score) ---
      echo "================================================"
      echo "Heap+linear_BT | Env: $env | K=$K | seed=$seed"
      echo "================================================"

      CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py \
        --config=configs/reward_heap_BT.yaml \
        --env=$env --seed=$seed --q_budget=$K --model_type=linear_BT --checkpoints_path=logs/

      CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py \
        --use_reward_model=True --config=configs/iql.yaml \
        --env=$env --seed=$seed --q_budget=$K \
        --feedback_type=heap --model_type=linear_BT --feedback_num=500 \
        --max_timesteps=250_000 --eval_freq=5_000

      # --- Heap + PL (listwise, best-only) ---
      echo "================================================"
      echo "Heap+PL | Env: $env | K=$K | seed=$seed"
      echo "================================================"

      CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py \
        --config=configs/reward_heap_PL.yaml \
        --env=$env --seed=$seed --q_budget=$K --checkpoints_path=logs/

      CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py \
        --use_reward_model=True --config=configs/iql.yaml \
        --env=$env --seed=$seed --q_budget=$K \
        --feedback_type=heap --model_type=PL --feedback_num=500 \
        --max_timesteps=250_000 --eval_freq=5_000

      # --- Heap + linear_PL (listwise, linear score) ---
      echo "================================================"
      echo "Heap+linear_PL | Env: $env | K=$K | seed=$seed"
      echo "================================================"

      CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py \
        --config=configs/reward_heap_PL.yaml \
        --env=$env --seed=$seed --q_budget=$K --model_type=linear_PL --checkpoints_path=logs/

      CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py \
        --use_reward_model=True --config=configs/iql.yaml \
        --env=$env --seed=$seed --q_budget=$K \
        --feedback_type=heap --model_type=linear_PL --feedback_num=500 \
        --max_timesteps=250_000 --eval_freq=5_000

    done
  done
done

echo ""
echo "All Heap experiments completed!"
