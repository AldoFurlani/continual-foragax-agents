python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name PPO-bar-horizontal --filter-alg-apertures Search-Oracle Search-9 PPO_LN_2048:9 PPO_LN_2048_crelu:9 PPO_LN_2048-l2-init:9 PPO_LN_128:9 PPO_LN_128-l2-init:9 PPO_LN_128_1:9 PPO_LN_128_1_crelu:9 PPO_LN_128_1-l2-init:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40
python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name PPO-RT-bar-horizontal --filter-alg-apertures Search-Oracle Search-9 PPO_LN_RT_2048:9 PPO_LN_RT_2048_crelu:9 PPO_LN_RT_2048-l2-init:9 PPO_LN_RT_128:9 PPO_LN_RT_128-l2-init:9 PPO_LN_RT_128_1:9 PPO_LN_RT_128_1_crelu:9 PPO_LN_RT_128_1-l2-init:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40
python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name RTU-bar-horizontal --filter-alg-apertures Search-Oracle Search-9 PPO-RTU_LN_2048:9 PPO-RTU_LN_2048_crelu:9 PPO-RTU_LN_2048-l2-init:9 PPO-RTU_LN_128:9 PPO-RTU_LN_128-l2-init:9 PPO-RTU_LN_128_1:9 PPO-RTU_LN_128_1_crelu:9 PPO-RTU_LN_128_1-l2-init:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40

python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name horizontal --filter-alg-apertures Search-Oracle Search-9 PPO_LN_128_1_crelu:9 PPO_LN_RT_128_1_crelu:9 PPO-RTU_LN_128_1_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40

python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name horizontal-base --filter-alg-apertures Search-Oracle Search-9 PPO_LN_128_1:9 PPO_LN_RT_128_1:9 PPO-RTU_LN_128_1:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40

python src/grouped_biome_bar.py experiments/XN35/foragax/ForagaxBig-v5 --filter-alg-apertures Search-Oracle Search-9 PPO_LN_128_1_crelu:9 PPO_LN_RT_128_1_crelu:9 PPO-RTU_LN_128_1_crelu:9 --plot-name biome-bar-horizontal --font-size 10

python src/grouped_biome_bar.py experiments/XN35/foragax/ForagaxBig-v5 --filter-alg-apertures Search-Oracle Search-9 PPO_LN_128_1:9 PPO_LN_RT_128_1:9 PPO-RTU_LN_128_1:9 --plot-name biome-bar-horizontal-base --font-size 10

python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name Search --filter-alg-apertures Search-Oracle Search-15 Search-9 Search-5 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40

python src/learning_curve.py experiments/XN35/foragax/ForagaxBig-v5 --plot-name Search-smooth --filter-alg-apertures Search-Oracle Search-15 Search-9 Search-5 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 40 --metric ewm_reward_5
