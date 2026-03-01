#!/usr/bin/env bash
set -euo pipefail

download_tasks=(
    box-close-v2
    button-press-topdown-v2
    button-press-topdown-wall-v2
    drawer-open-v2
    peg-insert-side-v2
    sweep-into-v2
)

declare -A file_ids=(
    ["box-close-v2"]="1yva0VXvnnyMOCLfWstj5q0TK-oi3Rt65"
    ["button-press-topdown-v2"]="1McVLA6KWi6KWJOI0lpIQ3dm60dxF_SpL"
    ["button-press-topdown-wall-v2"]="1dxdzUom2NsKFKkv0nrD2590UojoMYFmf"
    # ["dial-turn-v2"]="11lC_Ihn55Lruv-GDPe7lrkLNd0pxSCy4"
    ["drawer-open-v2"]="1ixXsiscRFFypnQBLRkb2WGYSTOwzU7XS"
    # ["hammer-v2"]="1QLQUTxlt9kFig6kzcAA6tm8oByLAe07v"
    # ["handle-pull-side-v2"]="16wrAL6708u8aODyuqAHgEjVmrHDuIUzY"
    # ["lever-pull-v2"]="17kLkqfAX3OPefb1bsuVXSMdglDmtYwuJ"
    ["peg-insert-side-v2"]="1Edy5_RPsoKW3gIKH4D7tNjMgDCWN-iqI"
    # ["peg-unplug-side-v2"]="1Elc7IU-J8D2IxTc8GnussLjFLIUW85SC"
    ["sweep-into-v2"]="1G3VghYKH5Mm2XHM69-oMl6uDNx7fdCxC"
    # ["sweep-v2"]="1u7f5WZYQlqXSxyJGI56kWlafYluFrgJb"
)

if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown not found. Install it with: pip install gdown" >&2
    exit 1
fi

mkdir -p dataset/MetaWorld

for task in "${download_tasks[@]}"; do
    file_id="${file_ids[$task]:-}"
    if [[ -z "$file_id" ]]; then
        echo "Skip ${task}: missing file_id"
        continue
    fi

    echo "Downloading ${task}"
    gdown --continue --output "dataset/MetaWorld/${task}.zip" "$file_id"
    unzip -q -o "dataset/MetaWorld/${task}.zip" -d dataset/MetaWorld/
    rm -f "dataset/MetaWorld/${task}.zip"
done

echo "Done. Dataset extracted under dataset/MetaWorld/"
