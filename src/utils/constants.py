import tol_colors as tc

LABEL_MAP = {
    "DQN": "DQN",
    "DQN_CReLU": "DQN (CReLU)",
    "DQN_L2": "DQN (L2)",
    "DQN_L2_Init": "DQN (L2 Init)",
    "DQN_LN": "DQN (LayerNorm)",
    "DQN_Reset_Head": "DQN (Head Reset)",
    "DQN_Hare_and_Tortoise": "DQN (Hare & Tortoise)",
    "DQN_ReDo": "DQN (ReDo, PostLN, ScoreReLU)",
    "DQN_ReDo_PostLNScore": "DQN (ReDo, PostLN, ScoreLN)",
    "DQN_ReDo_PreActLN": "DQN (ReDo, PreLN, ScoreReLU)",
    "DQN_Shrink_and_Perturb": "DQN (Shrink & Perturb)",
    "DQN_privileged": "DQN (Privileged)",
    "DQN_world": "DQN (World)",
    "Search-Brown": "Search (Brown)",
    "Search-Brown-Avoid-Green": "Search (Brown, Avoid Green)",
    "Search-Morel": "Search (Morel)",
    "Search-Morel-Avoid-Green": "Search (Morel, Avoid Green)",
    "Search-Nearest": "Search (Nearest)",
    "Search-Oracle": "Search (Oracle)",
    "Search-Oyster": "Search (Oyster)",
}

frozen_label_map = {}
for key in list(LABEL_MAP.keys()):
    frozen_label_map[f"{key}_greedy_frozen_5M"] = (
        f"{LABEL_MAP[key]} (Greedy Frozen @ 5M)"
    )
    frozen_label_map[f"{key}_greedy_frozen_1M"] = (
        f"{LABEL_MAP[key]} (Greedy Frozen @ 1M)"
    )
    frozen_label_map[f"{key}_frozen_1M"] = f"{LABEL_MAP[key]} (Frozen @ 1M)"
    frozen_label_map[f"{key}_frozen_5M"] = f"{LABEL_MAP[key]} (Frozen @ 5M)"
LABEL_MAP.update(frozen_label_map)

# Biome definitions for different environments
BIOME_DEFINITIONS = {
    "ForagaxTwoBiomeSmall-v2": {
        "Morel": ((3, 3), (6, 6)),
        "Oyster": ((11, 3), (14, 6)),
    },
    "ForagaxTwoBiome-v1": {
        "Morel": ((3, 0), (5, 15)),
        "Oyster": ((10, 0), (12, 15)),
    },
}

# Environment mapping
ENV_MAP = {"ForagaxTwoBiomeSmall": "ForagaxTwoBiomeSmall-v2"}

# Color scheme for plotting
colorset = tc.colorsets["high_contrast"]
sunset_colormap = tc.sunset

# Biome colors for plotting
TWO_BIOME_COLORS = {
    "Morel": colorset.red,
    "Oyster": colorset.yellow,
    "Neither": colorset.blue,
}

WEATHER_BIOME_COLORS = {
    "Cold": sunset_colormap(0.0),
    "Neither": sunset_colormap(0.5),
    "Hot": sunset_colormap(1.0),
}
