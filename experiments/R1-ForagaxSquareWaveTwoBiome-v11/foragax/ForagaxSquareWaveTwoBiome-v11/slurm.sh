for fov in 9;
do
    # ActorCrticMLP
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-reset.json

    # 128 ver
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 2 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1_crelu.json

    # 2048 ver
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_2048.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_2048-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 2 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_2048_crelu.json

    # PPO CRELU
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO_LN_2048_crelu.json

    # PPO and RTU-PPO with no Layernorm
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO_2048.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_2048.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 50 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_noLN.json

    # relu PPO agents
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_128_1-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_2048_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_LN_2048-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO_2048_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/PPO-RTU_2048_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2-init_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-l2_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_noLN_relu.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP-reset_relu.json

    # test 
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP_test.json

done
