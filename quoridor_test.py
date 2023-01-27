"""
Quoridor game example.
Prints board state to stdout with random agents by default.
"""

import re
import sys

sys.path.append("../")

import colorama
import numpy as np
from colorama import Fore, Style

from fights.base import BaseAgent
import faster_quoridor

class RandomAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: faster_quoridor.QuoridorState):
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

    def __call__(self, state: faster_quoridor.QuoridorState) -> faster_quoridor.QuoridorAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

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
    assert faster_quoridor.QuoridorEnv.env_id == RandomAgent.env_id
    colorama.init()

    state = faster_quoridor.QuoridorEnv().initialize_state()
    agents = [RandomAgent(0, 0), RandomAgent(1, 0)]

    print("\x1b[2J")

    it = 0
    while not state.done:

        print("\x1b[1;1H")
        print(fallback_to_ascii(colorize_walls(str(state))))

        for agent in agents:

            action = agent(state)
            state = faster_quoridor.QuoridorEnv().step(state, agent.agent_id, action)

            print("\x1b[1;1H")
            print(fallback_to_ascii(colorize_walls(str(state))))
            print(state.memory_cells[1,:,:,1])

            a = input()

            if state.done:
                print(f"agent {agent.agent_id} won in {it} iters")
                break

        it += 1

if __name__ == "__main__":
    run()