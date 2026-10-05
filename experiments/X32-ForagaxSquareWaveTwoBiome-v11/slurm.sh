# foragax-sweep (10 runs)
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time 03:00:00 --tasks 32 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-16G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json


# foragax (30 runs)
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_Shrink_and_Perturb.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-16G.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json
