# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/DQN.json
# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/DQN_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/DQN_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/DQN.json
# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/DQN_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/DQN_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512_frozen_1M.json    
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO_L2_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO_L2_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512_frozen_5M.json    
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32_frozen_1M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_512_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_256_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_128_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_64_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/15/PPO-RTU_32_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/5/PPO-RTU_L2_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/9/PPO-RTU_L2_frozen_5M.json
