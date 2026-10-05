python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 02:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_512.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 02:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_512.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 02:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO_512.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 512 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO_L2.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_512.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_512.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO-RTU_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/15/PPO-RTU_512.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/5/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v6/9/PPO-RTU_L2.json
