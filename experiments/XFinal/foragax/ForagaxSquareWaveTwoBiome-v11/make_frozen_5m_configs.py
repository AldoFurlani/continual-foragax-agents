import json
from pathlib import Path


SUFFIX = "_frozen_5M"

# Process the folder this script is in.
ROOT = Path(__file__).resolve().parent

FROZEN_LIST = [
    "DQN",
    "DRQN",
    "DQN_reward_trace",
    "RealTimeActorCriticMLP",
    "ActorCriticMLP",
    "ActorCriticMLP-reward-trace"
]

def main():
    for src in ROOT.rglob("*.json"):

        if src.stem not in FROZEN_LIST:
            continue
        
        if src.name.endswith(f"{SUFFIX}.json"):
            continue

        dst = src.with_name(src.stem + SUFFIX + src.suffix)

        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        data["agent"] = data["agent"] + SUFFIX
        data["metaParameters"]["freeze_steps"] = 5000000
        with dst.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.write("\n")

        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
