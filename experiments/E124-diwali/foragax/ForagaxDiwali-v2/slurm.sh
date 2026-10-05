python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/DQN_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/DQN_frozen_5M.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_frozen_5M.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO_L2_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO_L2_frozen_5M.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_frozen_5M.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_frozen_5M.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/5/PPO-RTU_L2_frozen_5M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_L2_frozen_1M.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax/ForagaxDiwali-v2/9/PPO-RTU_L2_frozen_5M.json
