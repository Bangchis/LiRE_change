#!/bin/bash
# Master script to run all RLT experiments across environments and model types
# Total: 3 envs × 2 model types = 6 experiments

ENVIRONMENTS=("box-close-v2" "button-press-topdown-v2" "drawer-open-v2")
MODEL_TYPES=("BT" "linear_BT")
METHOD="RLT"

total_runs=$((${#ENVIRONMENTS[@]} * ${#MODEL_TYPES[@]}))
current_run=0

echo "========================================"
echo "RLT EXPERIMENTS BATCH"
echo "Total runs: $total_runs"
echo "Environments: ${ENVIRONMENTS[@]}"
echo "Model types: ${MODEL_TYPES[@]}"
echo "========================================"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    for model in "${MODEL_TYPES[@]}"; do
        current_run=$((current_run + 1))

        echo ""
        echo "========================================"
        echo "RUN $current_run/$total_runs"
        echo "Method: $METHOD"
        echo "Model: $model"
        echo "Environment: $env"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
        echo ""

        # Export environment variables for the script
        export ENV_NAME="metaworld_${env}"
        export MODEL_TYPE="$model"
        export METHOD_NAME="$METHOD"

        # Run the experiment
        bash scripts/MetaWorld.sh

        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo ""
            echo "ERROR: Run $current_run failed with exit code $exit_code"
            echo "Continuing to next experiment..."
        else
            echo ""
            echo "SUCCESS: Run $current_run completed"
        fi

        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        # Small delay between runs
        sleep 5
    done
done

echo ""
echo "========================================"
echo "ALL RLT EXPERIMENTS COMPLETED"
echo "Total runs: $current_run"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
