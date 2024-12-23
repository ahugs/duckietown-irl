#!/bin/sh
export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> /scratch/xvfb.log & 

# for exp in partial_irl full_irl rl_human rl_velocity; do
#     for seed in 0 1 2; do
#         python scripts/evaluate_policy.py outdir=/workspaces/eval_results/mixed_small_loop/${exp}/${seed} policy_path=/workspaces/policies/mixed/${exp}/${seed}/snapshot.pt || continue
#     done
# done

for exp in partial_irl full_irl rl_human rl_velocity; do
    for seed in 0 1 2; do
        python scripts/evaluate_policy.py env.map_name=ETH_large_loop outdir=/workspaces/eval_results/mixed_large_loop/${exp}/${seed} policy_path=/workspaces/policies/mixed/${exp}/${seed}/snapshot.pt || continue
    done
done