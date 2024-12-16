#!/bin/sh
export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> /scratch/xvfb.log & 
for exp in rl_velocity; do
    for seed in 0; do
        python scripts/evaluate_policy.py outdir=/workspaces/eval_results/${exp}/${seed} policy_path=/workspaces/policies/${exp}/${seed}/snapshot.pt || continue
    done
done