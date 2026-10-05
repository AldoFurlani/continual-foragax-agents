for fov in 9;
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_HINT_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_RT_128.json

    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_BALANCED.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_HINT-RTU.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_HT.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_HINT_128.json

    python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN.json
    python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN_RT.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN_HINT.json
done
