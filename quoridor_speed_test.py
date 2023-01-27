"""
Quoridor Environment Speed Test
"""

import numpy as np
import time

from fights.base import BaseAgent
import faster_quoridor
import quoridor

class RandomAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: quoridor.QuoridorState):
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

    def __call__(self, state: quoridor.QuoridorState) -> quoridor.QuoridorAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

class FasterAgent(BaseAgent):
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

def run_original():
    assert quoridor.QuoridorEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = quoridor.QuoridorEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = quoridor.QuoridorEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

def run_faster():
    assert faster_quoridor.QuoridorEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = faster_quoridor.QuoridorEnv().initialize_state()
        agents = [FasterAgent(0, game), FasterAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = faster_quoridor.QuoridorEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run_original()
    run_faster()