for fov in 9;
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128G.json --tasks 3 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN37/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048_1.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128G.json --tasks 3 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN37/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048-l2-init.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128G.json --tasks 2 --time 06:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/XN37/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_2048-crelu.json
done
