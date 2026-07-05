#!/usr/bin/env bash
# Run one experiment's full seed grid locally, pushing every completed seed.
# Bootstraps everything on a fresh machine: venv, cpu torch, data, artifact.
#
# Usage (from anywhere, after git pull):
#     bash scripts/run_local_grid.sh E1_dqn_n2 "Milan Sazdov" milansazdov@gmail.com
# Identity args are optional if git config user.name and user.email are set.
# Progress: tail -f logs_<EXP>/seed_42.log
# Safe to rerun anytime: completed seeds are skipped, interrupted ones resume.
cd "$(dirname "$0")/.." || exit 1

CONFIG_NAME=${1:?usage: run_local_grid.sh <config name, e.g. E1_dqn_n2> [git name] [git email]}
GIT_NAME=${2:-$(git config user.name)}
GIT_EMAIL=${3:-$(git config user.email)}
CFG=configs/experiments/$CONFIG_NAME.yaml
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 1; }
[ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ] || { echo "git identity unknown, pass name and email as arguments"; exit 1; }

EXP_ID=$(grep '^experiment:' "$CFG" | awk '{print $2}')
EXP=results/experiments/$EXP_ID
LOGDIR=logs_$EXP_ID
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=1

# workers: physical cores plus two (the training math is core bound, extra
# hyperthread workers mostly contend), capped so each keeps about 2GB of ram
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$PHYS" -lt 1 ] && PHYS=$(( $(nproc) / 2 ))
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
PARALLEL=$((PHYS + 2))
[ "$PARALLEL" -gt $((AVAIL_GB / 2)) ] && PARALLEL=$((AVAIL_GB / 2))
[ "$PARALLEL" -lt 2 ] && PARALLEL=2

echo "== bootstrap =="
if ! venv/bin/python3 -c "import torch, pettingzoo" 2>/dev/null; then
    echo "setting up venv and dependencies (one time, a few minutes)"
    python3 -m venv venv || { echo "hint: sudo apt install python3-venv python3-pip"; exit 1; }
    venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || exit 1
    venv/bin/pip install -q -r requirements.txt || exit 1
    venv/bin/pip install -q -e . || exit 1
fi
if [ ! -f data/processed/demand_dataset.parquet ]; then
    echo "fetching data and preprocessing (one time, about 15 minutes)"
    venv/bin/python3 scripts/download_data.py || exit 1
    venv/bin/python3 scripts/preprocess.py || exit 1
fi
venv/bin/python3 -c "from airbnb_marl.demand.interface import DemandModel; DemandModel.load('results/demand')" \
    || { echo "demand artifact missing or stale, git pull first"; exit 1; }

push_seed() {
    seed=$1
    git add "$EXP/config.json" "$EXP/seed_$seed" 2>/dev/null
    git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
        -c commit.gpgsign=false commit -q -m "feat: add $CONFIG_NAME run results for seed $seed" \
        || { echo "nothing to commit for seed $seed"; return 0; }
    git -c rebase.autoStash=true pull --rebase -q origin main
    if git push -q origin main; then
        echo "pushed seed $seed"
    else
        echo "PUSH FAILED for seed $seed (kept locally, retried with next seed)"
    fi
}

declare -A running=()
active=0
seeds=$(seq 42 61)
queue=""
for s in $seeds; do
    if [ -f "$EXP/seed_$s/summary.json" ]; then
        echo "seed $s already complete, skipping"
    else
        queue="$queue $s"
    fi
done

echo "== $CONFIG_NAME: $PARALLEL workers ($CORES cores, ${AVAIL_GB}GB free), queue:$queue =="
while [ -n "$queue" ] || [ "$active" -gt 0 ]; do
    while [ -n "$queue" ] && [ "$active" -lt "$PARALLEL" ]; do
        s=$(echo $queue | cut -d' ' -f1)
        queue=$(echo $queue | cut -s -d' ' -f2-)
        venv/bin/python3 scripts/train_rl.py --config "$CFG" --seed "$s" \
            > "$LOGDIR/seed_$s.log" 2>&1 &
        running[$s]=$!
        active=$((active + 1))
        echo "started seed $s (pid ${running[$s]})"
    done
    sleep 30
    for s in "${!running[@]}"; do
        if ! kill -0 "${running[$s]}" 2>/dev/null; then
            unset "running[$s]"
            active=$((active - 1))
            if [ -f "$EXP/seed_$s/summary.json" ]; then
                echo "seed $s complete"
                push_seed "$s"
            else
                echo "seed $s FAILED, last log lines:"
                tail -3 "$LOGDIR/seed_$s.log"
            fi
        fi
    done
done

done_count=0
for s in $seeds; do
    [ -f "$EXP/seed_$s/summary.json" ] && done_count=$((done_count + 1))
done
echo "finished: $done_count/20 seeds of $EXP_ID have complete results"
[ "$done_count" -eq 20 ] && echo "ALL DONE" || echo "rerun this script to retry the missing seeds"
