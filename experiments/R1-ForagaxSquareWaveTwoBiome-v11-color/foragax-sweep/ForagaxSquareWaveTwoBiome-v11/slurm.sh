for fov in 9;
do
    # RTU-PPO (conv) 2048 ver
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO-RTU_LN_2048.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO-RTU_LN_2048_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO-RTU_LN_2048-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO-RTU_LN_2048-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 3 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO-RTU_LN_2048_crelu.json

    # PPO w/ crelu
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/PPO_LN_2048_crelu.json

    # ActorCriticMLP's
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-l2.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-l2_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-reset.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP-reset_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/${fov}/ActorCriticMLP_tanh.json

    # RTU-PPO (MLP) 2048 ver
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 3 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP_crelu.json

    # ActorCriticMLP's
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2-init_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2-init_relu_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2_relu_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-reset_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-reset_relu_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_relu_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_tanh_2.json


done
