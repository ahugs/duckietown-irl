

# DrQ-v2 on Duckietown

Implementation of DrQ-v2 taken from original [repo](https://github.com/facebookresearch/drqv2).

[[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning]](https://arxiv.org/abs/2107.09645) by

## Method
DrQ-v2 is a model-free off-policy algorithm for image-based continuous control. DrQ-v2 builds on [DrQ](https://github.com/denisyarats/drq), an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements including:
- Switch the base RL learner from SAC to DDPG.
- Incorporate n-step returns to estimate TD error.
- Introduce a decaying schedule for exploration noise.
- Make implementation 3.5 times faster.
- Find better hyper-parameters.

## Instructions

```
docker run -e WANDB_API_KEY={WANDB_API_KEY}  -v $(pwd):/workspaces/ --gpus all -it duckietown-irl:latest
python train.py
```

```
 docker build -f Dockerfile -t duckietown-irl:latest .
 docker run --gpus all -v ~/PGM/duckietown-irl:/workspace -it duckietown-irl:latest
 git clone https://github.com/florenceCloutier/gym-duckietown.git
 pip install --no-cache-dir -e gym-duckietown 
```