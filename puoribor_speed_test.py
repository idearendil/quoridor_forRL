"""
Othello Environment Speed Test
"""

import numpy as np
import time

from fights.base import BaseAgent
import faster_puoribor
import puoribor

class RandomAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: puoribor.PuoriborState):
        actions = []
        for action_type in [0, 1, 2, 3]:
            for coordinate_x in range(puoribor.PuoriborEnv.board_size):
                for coordinate_y in range(puoribor.PuoriborEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        puoribor.PuoriborEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, state: puoribor.PuoriborState) -> puoribor.PuoriborAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

class FasterAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: faster_puoribor.PuoriborState):
        actions = []
        for action_type in [0, 1, 2, 3]:
            for coordinate_x in range(faster_puoribor.PuoriborEnv.board_size):
                for coordinate_y in range(faster_puoribor.PuoriborEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        faster_puoribor.PuoriborEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, state: faster_puoribor.PuoriborState) -> faster_puoribor.PuoriborAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

def run_original():
    assert puoribor.PuoriborEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = puoribor.PuoriborEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = puoribor.PuoriborEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

def run_faster():
    assert faster_puoribor.PuoriborEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = faster_puoribor.PuoriborEnv().initialize_state()
        agents = [FasterAgent(0, game), FasterAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = faster_puoribor.PuoriborEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run_original()
    run_faster()