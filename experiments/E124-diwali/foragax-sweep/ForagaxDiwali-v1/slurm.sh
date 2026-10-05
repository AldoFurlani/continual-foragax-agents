python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN_L2.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN_SWR.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --tasks 128 --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN_SWR.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/PPO.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/PPO.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/PPO_L2.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/PPO-RTU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/PPO-RTU.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/PPO-RTU_L2.json
