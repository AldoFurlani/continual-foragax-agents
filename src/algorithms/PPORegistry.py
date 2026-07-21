from algorithms.nn.ACConv import ActorCriticConv
from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.ESMAC import ESMAC
from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvHint import RealTimeActorCriticConvHint
from algorithms.nn.RealTimeACConvHintRTU import RealTimeActorCriticConvHintRTU
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.nn.RealTimeACMLPStacked import RealTimeActorCriticMLPStacked


def getAgent(name):
    if name.startswith("RealTimeActorCriticConvPooling"):
        return RealTimeActorCriticConvPooling

    if name.startswith("RealTimeActorCriticConvHintRTU"):
        return RealTimeActorCriticConvHintRTU

    if name.startswith("RealTimeActorCriticConvHint"):
        return RealTimeActorCriticConvHint

    if name.startswith("RealTimeActorCriticConv"):
        return RealTimeActorCriticConv

    if name.startswith("RealTimeActorCriticMLPMulti"):
        return RealTimeActorCriticMLPMulti

    # Must precede the generic RealTimeActorCriticMLP prefix check.
    if name.startswith("RealTimeActorCriticMLPStacked"):
        return RealTimeActorCriticMLPStacked

    if name.startswith("ActorCriticConv"):
        return ActorCriticConv

    # tanh and ReLU share one class; the activation comes from the explicit
    # `representation.activation` config field (read in rtu_ppo.py). Variants are
    # kept as separate config files so results/plots stay separable.
    if name.startswith("RealTimeActorCriticMLP"):
        return RealTimeActorCriticMLP

    if name.startswith("ActorCriticMLP"):
        return ActorCriticMLP

    if name.startswith("ESMAC"):
        return ESMAC

    # Hint-aware RTU variants must be checked before the generic PPO-RTU fallback.
    if "HINT-RTU" in name:
        return RealTimeActorCriticConvHintRTU

    if "_HT" in name and name.startswith("PPO-RTU"):
        # Trace is applied externally; use base conv arch
        return RealTimeActorCriticConv

    if "BALANCED" in name and name.startswith("PPO-RTU"):
        return RealTimeActorCriticConvHint

    if name.startswith("PPO-RTU"):
        return RealTimeActorCriticConv

    if name.startswith("PPO"):
        return ActorCriticConv

    raise Exception("Unknown algorithm")
