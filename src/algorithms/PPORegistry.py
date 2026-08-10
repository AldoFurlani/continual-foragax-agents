from algorithms.nn.ACConv import ActorCriticConv
from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.BPTTACConv import BPTTActorCriticConv
from algorithms.nn.BPTTACConvStacked import BPTTActorCriticConvStacked
from algorithms.nn.BPTTACMLP import BPTTActorCriticMLP
from algorithms.nn.BPTTACMLPStacked import BPTTActorCriticMLPStacked
from algorithms.nn.ESMAC import ESMAC
from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvMulti import RealTimeActorCriticConvMulti
from algorithms.nn.RealTimeACConvHint import RealTimeActorCriticConvHint
from algorithms.nn.RealTimeACConvHintRTU import RealTimeActorCriticConvHintRTU
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.nn.RealTimeACMLPStacked import RealTimeActorCriticMLPStacked


def getAgent(name):
    # T-BPTT variants: same architectures as the RealTime* classes, but the RTU
    # is unrolled over a `seq_len` time axis instead of stepped with the
    # real-time RTRL cell.  Checked first -- their names share no prefix with
    # the RealTime* family, so ordering here is for readability only.
    # Must precede the generic BPTTActorCriticConv prefix check.
    if name.startswith("BPTTActorCriticConvStacked"):
        return BPTTActorCriticConvStacked

    if name.startswith("BPTTActorCriticConv"):
        return BPTTActorCriticConv

    # Must precede the generic BPTTActorCriticMLP prefix check.
    if name.startswith("BPTTActorCriticMLPStacked"):
        return BPTTActorCriticMLPStacked

    if name.startswith("BPTTActorCriticMLP"):
        return BPTTActorCriticMLP

    if name.startswith("RealTimeActorCriticConvPooling"):
        return RealTimeActorCriticConvPooling

    if name.startswith("RealTimeActorCriticConvHintRTU"):
        return RealTimeActorCriticConvHintRTU

    if name.startswith("RealTimeActorCriticConvHint"):
        return RealTimeActorCriticConvHint

    # Must precede the generic RealTimeActorCriticConv prefix check.
    if name.startswith("RealTimeActorCriticConvMulti"):
        return RealTimeActorCriticConvMulti

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
