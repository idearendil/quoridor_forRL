"""
Othello Environment Speed Test
"""

import re
import sys

sys.path.append("../")

import colorama

import numpy as np
import time
from colorama import Fore, Style

from fights.base import BaseAgent
import faster_quoridor
import quoridor

class TestAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions_original(self, state: quoridor.QuoridorState):
        actions = []
        for action_type in [0, 1, 2]:
            for coordinate_x in range(quoridor.QuoridorEnv.board_size):
                for coordinate_y in range(quoridor.QuoridorEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        quoridor.QuoridorEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions
    
    def _get_all_actions_faster(self, state: faster_quoridor.QuoridorState):
        actions = []
        for action_type in [0, 1, 2]:
            for coordinate_x in range(faster_quoridor.QuoridorEnv.board_size):
                for coordinate_y in range(faster_quoridor.QuoridorEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        faster_quoridor.QuoridorEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, original_state: quoridor.QuoridorState, faster_state: faster_quoridor.QuoridorState) -> quoridor.QuoridorAction:
        original_actions = self._get_all_actions_original(original_state)
        faster_actions = self._get_all_actions_faster(faster_state)
        if not original_actions == faster_actions:
            raise ValueError(f"error! error!")
        return self._rng.choice(original_actions)

def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s

def colorize_walls(s: str) -> str:
    return s.replace("━", Fore.BLUE + "━" + Style.RESET_ALL).replace(
        "┃", Fore.RED + "┃" + Style.RESET_ALL
    )


def run():
    assert quoridor.QuoridorEnv.env_id == TestAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        original_state = quoridor.QuoridorEnv().initialize_state()
        faster_state = faster_quoridor.QuoridorEnv().initialize_state()
        agents = [TestAgent(0, game), TestAgent(1, game)]

        while not original_state.done:

            for agent in agents:

                action = agent(original_state, faster_state)
                original_state = quoridor.QuoridorEnv().step(original_state, agent.agent_id, action)
                faster_state = faster_quoridor.QuoridorEnv().step(faster_state, agent.agent_id, action)

                if original_state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run()