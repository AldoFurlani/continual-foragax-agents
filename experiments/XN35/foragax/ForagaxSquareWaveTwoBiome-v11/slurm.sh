for fov in 9;
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1-l2-init.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1_crelu.json
done
