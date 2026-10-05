for fov in 9;
do
    # 128 ver
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_1.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_1-l2-init.json
    #python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_128_1_crelu.json

    # 2048 ver
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048-l2-init.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --entry src/rtu_ppo.py --force -e experiments/XN35/foragax-sweep/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048_crelu.json
done
