#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_parallel.sh [options]

Options:
  --num-gpus <int>            Number of GPUs to use (default: 4)
  --seed <int>                Seed for all jobs (default: 10)
  --iql-steps <int>           IQL max timesteps (default: 65000)
  --iql-eval-freq <int>       IQL evaluation frequency (default: 10000)
  --iql-eval-episodes <int>   IQL evaluation episodes (default: 10)
  --reward-epochs <int>       Reward model training epochs (default: 220)
  --wandb-project <string>    W&B project name (required unless --dry-run)
  --wandb-entity <string>     W&B entity/team (optional)
  --dry-run                   Print job matrix and exit
  --help                      Show this help

Outputs:
  results/job_status.csv
  results/logs/*.log
  results/report_accuracy.csv
  results/report.md
EOF
}

NUM_GPUS=4
SEED=10
IQL_STEPS=65000
IQL_EVAL_FREQ=10000
IQL_EVAL_EPISODES=10
REWARD_EPOCHS=220
WANDB_PROJECT=""
WANDB_ENTITY=""
DRY_RUN=false
THREADS_PER_WORKER=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --iql-steps)
      IQL_STEPS="$2"
      shift 2
      ;;
    --iql-eval-freq)
      IQL_EVAL_FREQ="$2"
      shift 2
      ;;
    --iql-eval-episodes)
      IQL_EVAL_EPISODES="$2"
      shift 2
      ;;
    --reward-epochs)
      REWARD_EPOCHS="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$WANDB_PROJECT" && "$DRY_RUN" == "false" ]]; then
  echo "Error: --wandb-project is required unless --dry-run is used." >&2
  exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "Error: --num-gpus must be a positive integer." >&2
  exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Error: --seed must be a non-negative integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/logs"
STATUS_CSV="${RESULTS_DIR}/job_status.csv"
STATUS_LOCK="${RESULTS_DIR}/.job_status.lock"
GPU_LOCK_PREFIX="/tmp/lire_gpu_lock_"
SESSION_ID="$(date +%Y%m%d_%H%M%S)"
STATUS_HEADER="timestamp,job_name,gpu,method,feedback_type,model_type,q_budget,env,seed,reward_epochs,iql_steps,iql_eval_freq,iql_eval_episodes,wandb_project,status,stage,message,session_id"

mkdir -p "$LOG_DIR"
touch "$STATUS_LOCK"

if [[ ! -f "$STATUS_CSV" ]]; then
  echo "$STATUS_HEADER" > "$STATUS_CSV"
else
  current_header="$(head -n 1 "$STATUS_CSV")"
  if [[ "$current_header" != "$STATUS_HEADER" ]]; then
    backup_path="${STATUS_CSV}.bak_${SESSION_ID}"
    mv "$STATUS_CSV" "$backup_path"
    echo "$STATUS_HEADER" > "$STATUS_CSV"
    echo "Detected old results/job_status.csv format. Backed up to: $backup_path"
  fi
fi

append_status() {
  local line="$1"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 200
      echo "$line" >> "$STATUS_CSV"
    ) 200>>"$STATUS_LOCK"
  else
    echo "$line" >> "$STATUS_CSV"
  fi
}

job_has_success() {
  local job_name="$1"
  awk -F',' -v name="$job_name" -v seed="$SEED" -v reward_epochs="$REWARD_EPOCHS" -v iql_steps="$IQL_STEPS" -v iql_eval_freq="$IQL_EVAL_FREQ" -v iql_eval_episodes="$IQL_EVAL_EPISODES" -v wandb_project="$WANDB_PROJECT" '
    $2 == name &&
    $9 == seed &&
    $10 == reward_epochs &&
    $11 == iql_steps &&
    $12 == iql_eval_freq &&
    $13 == iql_eval_episodes &&
    $14 == wandb_project &&
    $15 == "SUCCESS" { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$STATUS_CSV"
}

ENVS=(
  "box-close-v2"
  "button-press-topdown-v2"
  "button-press-topdown-wall-v2"
  "sweep-into-v2"
  "drawer-open-v2"
  "peg-insert-side-v2"
)

declare -a JOBS
for env in "${ENVS[@]}"; do
  full_env="metaworld_${env}"
  JOBS+=("Heap+BT|heap|BT|5|configs/reward_heap_BT.yaml|heap|${full_env}")
  JOBS+=("Heap+linear_BT|heap|linear_BT|5|configs/reward_heap_BT.yaml|heap|${full_env}")
  JOBS+=("Heap+PL|heap|PL|5|configs/reward_heap_PL.yaml|heap|${full_env}")
  JOBS+=("Heap+linear_PL|heap|linear_PL|5|configs/reward_heap_PL.yaml|heap|${full_env}")
  JOBS+=("LiRE+linear_BT|RLT|linear_BT|100|configs/reward_RLT.yaml|RLT|${full_env}")
done

print_job_matrix() {
  local idx=1
  echo "Session ID: ${SESSION_ID}"
  echo "Total jobs: ${#JOBS[@]}"
  echo "seed=${SEED}, reward_epochs=${REWARD_EPOCHS}, iql_steps=${IQL_STEPS}, eval_freq=${IQL_EVAL_FREQ}, n_episodes=${IQL_EVAL_EPISODES}"
  echo "--------------------------------------------------------------------------------"
  printf "%-4s %-18s %-8s %-12s %-8s %-36s\n" "ID" "METHOD" "K" "MODEL" "FT" "ENV"
  echo "--------------------------------------------------------------------------------"
  for job in "${JOBS[@]}"; do
    IFS='|' read -r method feedback_type model_type q_budget _cfg _tag env_full <<< "$job"
    printf "%-4s %-18s %-8s %-12s %-8s %-36s\n" "$idx" "$method" "$q_budget" "$model_type" "$feedback_type" "$env_full"
    idx=$((idx + 1))
  done
  echo "--------------------------------------------------------------------------------"
}

run_single_job() {
  local gpu="$1"
  local method="$2"
  local feedback_type="$3"
  local model_type="$4"
  local q_budget="$5"
  local reward_cfg="$6"
  local method_tag="$7"
  local env_full="$8"
  local env_short="${env_full#metaworld_}"
  local job_name="${feedback_type}_${model_type}_K${q_budget}_${env_short}_s${SEED}"
  local reward_log="${LOG_DIR}/${job_name}_reward.log"
  local iql_log="${LOG_DIR}/${job_name}_iql.log"
  local start_ts
  local end_ts
  local duration

  release_gpu_lock() {
    rmdir "${GPU_LOCK_PREFIX}${gpu}" 2>/dev/null || true
  }
  trap release_gpu_lock EXIT

  start_ts="$(date +%s)"
  append_status "$(date -Iseconds),${job_name},${gpu},${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},STARTED,reward,NA,${SESSION_ID}"
  echo "[GPU ${gpu}] START ${job_name}"

  (
    cd "$REPO_ROOT" || exit 1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
    if [[ -n "$WANDB_ENTITY" ]]; then
      export WANDB_ENTITY="$WANDB_ENTITY"
    fi
    python3 Reward_learning/learn_reward.py \
      --config="${reward_cfg}" \
      --env="${env_full}" \
      --seed="${SEED}" \
      --feedback_num=500 \
      --q_budget="${q_budget}" \
      --feedback_type="${feedback_type}" \
      --model_type="${model_type}" \
      --epochs="${REWARD_EPOCHS}" \
      --method_tag="${method_tag}" \
      --project="${WANDB_PROJECT}" \
      --checkpoints_path=logs/ \
      2>&1 | tee "${reward_log}"
  )
  local reward_rc=$?
  if [[ "$reward_rc" -ne 0 ]]; then
    end_ts="$(date +%s)"
    duration=$((end_ts - start_ts))
    append_status "$(date -Iseconds),${job_name},${gpu},${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},FAILED_REWARD,reward,exit_${reward_rc}_dur_${duration},${SESSION_ID}"
    echo "[GPU ${gpu}] FAIL(reward) ${job_name}"
    return 1
  fi

  append_status "$(date -Iseconds),${job_name},${gpu},${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},STARTED,iql,NA,${SESSION_ID}"

  (
    cd "$REPO_ROOT" || exit 1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
    if [[ -n "$WANDB_ENTITY" ]]; then
      export WANDB_ENTITY="$WANDB_ENTITY"
    fi
    python3 algorithms/iql.py \
      --config=configs/iql.yaml \
      --use_reward_model=True \
      --env="${env_full}" \
      --seed="${SEED}" \
      --feedback_num=500 \
      --q_budget="${q_budget}" \
      --feedback_type="${feedback_type}" \
      --model_type="${model_type}" \
      --epochs="${REWARD_EPOCHS}" \
      --method_tag="${method_tag}" \
      --project="${WANDB_PROJECT}" \
      --max_timesteps="${IQL_STEPS}" \
      --eval_freq="${IQL_EVAL_FREQ}" \
      --n_episodes="${IQL_EVAL_EPISODES}" \
      2>&1 | tee "${iql_log}"
  )
  local iql_rc=$?
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))
  if [[ "$iql_rc" -ne 0 ]]; then
    append_status "$(date -Iseconds),${job_name},${gpu},${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},FAILED_IQL,iql,exit_${iql_rc}_dur_${duration},${SESSION_ID}"
    echo "[GPU ${gpu}] FAIL(iql) ${job_name}"
    return 1
  fi

  append_status "$(date -Iseconds),${job_name},${gpu},${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},SUCCESS,all,dur_${duration},${SESSION_ID}"
  echo "[GPU ${gpu}] DONE ${job_name} (${duration}s)"
  return 0
}

print_job_matrix

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry-run enabled. No jobs were launched."
  exit 0
fi

for ((g=0; g<NUM_GPUS; g++)); do
  rmdir "${GPU_LOCK_PREFIX}${g}" 2>/dev/null || true
done

launched=0
skipped=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r method feedback_type model_type q_budget reward_cfg method_tag env_full <<< "$job"
  env_short="${env_full#metaworld_}"
  job_name="${feedback_type}_${model_type}_K${q_budget}_${env_short}_s${SEED}"

  if job_has_success "$job_name"; then
    echo "[SKIP] ${job_name} already marked SUCCESS."
    skipped=$((skipped + 1))
    append_status "$(date -Iseconds),${job_name},NA,${method},${feedback_type},${model_type},${q_budget},${env_full},${SEED},${REWARD_EPOCHS},${IQL_STEPS},${IQL_EVAL_FREQ},${IQL_EVAL_EPISODES},${WANDB_PROJECT},SKIPPED,resume,already_success,${SESSION_ID}"
    continue
  fi

  assigned_gpu=-1
  while [[ "$assigned_gpu" -lt 0 ]]; do
    for ((g=0; g<NUM_GPUS; g++)); do
      if mkdir "${GPU_LOCK_PREFIX}${g}" 2>/dev/null; then
        assigned_gpu="$g"
        break
      fi
    done
    if [[ "$assigned_gpu" -lt 0 ]]; then
      sleep 2
    fi
  done

  run_single_job "$assigned_gpu" "$method" "$feedback_type" "$model_type" "$q_budget" "$reward_cfg" "$method_tag" "$env_full" &
  launched=$((launched + 1))
done

wait

success_count=$(awk -F',' -v sid="$SESSION_ID" '$18==sid && $15=="SUCCESS" {c++} END {print c+0}' "$STATUS_CSV")
failed_count=$(awk -F',' -v sid="$SESSION_ID" '$18==sid && ($15=="FAILED_REWARD" || $15=="FAILED_IQL") {c++} END {print c+0}' "$STATUS_CSV")

echo "Session ${SESSION_ID} summary: launched=${launched}, skipped=${skipped}, success=${success_count}, failed=${failed_count}"

if [[ "$failed_count" -gt 0 ]]; then
  echo "Some jobs failed. Inspect results/job_status.csv and results/logs/*.log"
  exit 1
fi

report_cmd=(python3 "${SCRIPT_DIR}/generate_report.py" \
  --project "${WANDB_PROJECT}" \
  --output-dir "${RESULTS_DIR}" \
  --seed "${SEED}" \
  --reward-epochs "${REWARD_EPOCHS}" \
  --iql-steps "${IQL_STEPS}" \
  --iql-eval-freq "${IQL_EVAL_FREQ}" \
  --iql-eval-episodes "${IQL_EVAL_EPISODES}")

if [[ -n "$WANDB_ENTITY" ]]; then
  report_cmd+=(--entity "${WANDB_ENTITY}")
fi

echo "Generating W&B report..."
"${report_cmd[@]}"
echo "All done."
