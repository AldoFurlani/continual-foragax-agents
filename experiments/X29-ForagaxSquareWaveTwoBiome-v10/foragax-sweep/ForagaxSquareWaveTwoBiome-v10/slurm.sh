for fov in {9}
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_PConv.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_Color.json
done
