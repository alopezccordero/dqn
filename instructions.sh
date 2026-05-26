#!/bin/bash
#SBATCH --job-name=spaceinvaders-dqn
#SBATCH --output=spaceinvaders-dqn.out.%j
#SBATCH --partition=gpua30q
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate atari-project

echo "Python"
which python
python --version


echo "Starting spaceinvaders training"

python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml

echo "training finished"

echo "Evaluation"

python -m rl_zoo3.enjoy --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder logs/

echo "evaluation finished"
