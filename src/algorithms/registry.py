from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.DebugAgent import DebugAgent
from algorithms.MCTSAgent import MCTSAgent
from algorithms.nn.AADRQN import AADRQN
from algorithms.nn.ATAADRQN import ATAADRQN
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN_Hare_and_Tortoise import DQN_Hare_and_Tortoise
from algorithms.nn.DQN_L2 import DQN_L2
from algorithms.nn.DQN_L2_Init import DQN_L2_Init
from algorithms.nn.DQN_ReDo import DQN_ReDo
from algorithms.nn.DQN_Reset import DQN_Reset
from algorithms.nn.DQN_Shrink_and_Perturb import DQN_Shrink_and_Perturb
from algorithms.nn.DQN_Spectral_Reg import DQN_Spectral_Reg
from algorithms.nn.DRQN import DRQN
from algorithms.nn.PT_DQN import PT_DQN
from algorithms.nn.EQRC import EQRC
from algorithms.nn.MADRQN import MADRQN
from algorithms.RandomAgent import RandomAgent
from algorithms.SearchAgent import SearchAgent
from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC


def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN_L2_Init"):
        return DQN_L2_Init

    if name.startswith("DQN_L2"):
        return DQN_L2

    if name.startswith("DQN_Spectral_Reg"):
        return DQN_Spectral_Reg

    if name == "DQN":
        return DQN

    if name.startswith("DRQN"):
        return DRQN

    if name.startswith("AADRQN"):
        return AADRQN

    if name.startswith("ATAADRQN"):
        return ATAADRQN

    if name.startswith("MADRQN"):
        return MADRQN

    if name.startswith("DQN_ReDo"):
        return DQN_ReDo

    if name.startswith("DQN_Reset"):
        return DQN_Reset

    if name.startswith("DQN_Shrink_and_Perturb"):
        return DQN_Shrink_and_Perturb

    if name.startswith("DQN_Hare_and_Tortoise"):
        return DQN_Hare_and_Tortoise

    if name.startswith("PT_DQN"):
        return PT_DQN

    if name.startswith("DQN"):
        return DQN

    if name == "EQRC":
        return EQRC

    if name == "ESARSA":
        return ESARSA

    if name == "SoftmaxAC":
        return SoftmaxAC

    if name == "Debug":
        return DebugAgent

    if name == "Random":
        return RandomAgent

    if name.startswith("Search"):
        return SearchAgent

    if name.startswith("MCTS"):
        return MCTSAgent

    raise Exception("Unknown algorithm")
