import numpy as np
import cv2
from gym import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class State(object):
    def __init__(self):
        self.t = None
        self.done = None
        self.agent_pos = [0, 0]
        self.goal_pos = [0, 0]


class Environment(object):
    def __init__(self, seed):

        self.height = 11
        self.width = 11
        self.t_max = 100

        self.agent_color = np.asarray([255, 255, 255])
        self.goal_color = np.asarray([153, 255, 153])
        self.lava_color = np.asarray([153, 153, 255])

        self.obs_shape = (self.height, self.width, 3)
        self.obs_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[-1, :] = 1
        self.grid[:, -1] = 1
        self.grid[5, :] = 1
        self.grid[:, 5] = 1
        self.grid[5, 2] = 0
        self.grid[5, 8] = 0
        self.grid[8, 5] = 0

        self.agent_pos = [1, 4]
        self.goal_pos = [1, 6]

        self.lava_positions = np.where(self.grid == 1)
        self.passage_positions = np.where(self.grid == 0)

        self.state_id_dict = {}
        self.transition_id_dict = {}
        state_id = 0
        transition_id = 0
        for n in range(len(self.passage_positions[0])):
            if not (self.passage_positions[0][n] == 1 and self.passage_positions[1][n] == 6):
                self.state_id_dict[(self.passage_positions[0][n], self.passage_positions[1][n])] = state_id
                for i_action in range(4):
                    self.transition_id_dict[(self.passage_positions[0][n], self.passage_positions[1][n], i_action)] = \
                        transition_id
                    transition_id += 1
                state_id += 1

        self.n_states = len(self.state_id_dict)
        self.n_transitions = len(self.transition_id_dict)

        self.base_obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.base_obs[self.goal_pos[0], self.goal_pos[1], 1] = 1
        for n in range(len(self.lava_positions[0])):
            self.base_obs[self.lava_positions[0][n], self.lava_positions[1][n], 2] = 1

        self.base_obs_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.base_obs_image[self.goal_pos[0], self.goal_pos[1], :] = self.goal_color
        for n in range(len(self.lava_positions[0])):
            self.base_obs_image[self.lava_positions[0][n], self.lava_positions[1][n], :] = self.lava_color

        self.random = np.random.RandomState(seed)
        self.state = None

    # ------------------------------------------------------------------------------------------------------------------

    def reset(self):
        self.state = State()
        self.state.t = 0
        self.state.done = False
        self.state.agent_pos = list(self.agent_pos)
        self.state.goal_pos = list(self.goal_pos)
        return self.state_to_obs(self.state)

    # ------------------------------------------------------------------------------------------------------------------

    def transitive(self, state, action):

        if self.random.random_sample() < 0.05:
            p = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            p[action] = 0.0
            action = self.random.choice(4, 1, p=p)[0]

        if action == 0:  # Down
            state.agent_pos[0] += 1
        elif action == 1:  # Up
            state.agent_pos[0] -= 1
        elif action == 2:  # Right
            state.agent_pos[1] += 1
        elif action == 3:  # Left
            state.agent_pos[1] -= 1

        reward = 0.0

        if self.grid[state.agent_pos[0], state.agent_pos[1]] == 1:
            state.done = True
        elif state.agent_pos[0] == state.goal_pos[0] and state.agent_pos[1] == state.goal_pos[1]:
            state.done = True
            reward = 1.0

        state.t += 1
        state.done = state.done or state.t >= self.t_max

        return reward

    # ------------------------------------------------------------------------------------------------------------------

    def state_to_obs(self, state):
        if state.done:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs = self.base_obs.copy()
        obs[state.agent_pos[0], state.agent_pos[1], 0] = 1
        return obs

    # ------------------------------------------------------------------------------------------------------------------

    def generate_obs(self, agent_pos):
        obs = self.base_obs.copy()
        obs[agent_pos[0], agent_pos[1], 0] = 1
        return obs

    # ------------------------------------------------------------------------------------------------------------------

    def optimal_action(self, state=None):
        if state is None:
            state = self.state

        # TODO: Save actions

        grid = Grid(matrix=(1 - self.grid))
        start = grid.node(state.agent_pos[1], state.agent_pos[0])
        end = grid.node(state.goal_pos[1], state.goal_pos[0])

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        dif_h = path[1][1] - path[0][1]
        dif_w = path[1][0] - path[0][0]

        # Down, Up, Right, Left
        if dif_w != 0:
            if dif_w > 0:
                return 2
            elif dif_w < 0:
                return 3
        elif dif_h != 0:
            if dif_h > 0:
                return 0
            elif dif_h < 0:
                return 1

    # ------------------------------------------------------------------------------------------------------------------

    def render(self, state=None):
        if state is None:
            state = self.state
        state = self.state
        obs_image = self.base_obs_image.copy()
        obs_image[state.agent_pos[0], state.agent_pos[1], :] = self.agent_color
        obs_image = cv2.resize(obs_image, (int(self.height * 30), int(self.width * 30)),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return obs_image

    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action):
        reward = self.transitive(self.state, action)
        return self.state_to_obs(self.state), reward, self.state.done

    # ------------------------------------------------------------------------------------------------------------------

    def get_state(self):
        return self.state.agent_pos

    # ------------------------------------------------------------------------------------------------------------------

    # (Re)sets the random state of the environment, useful to reproduce the same level sequences for evaluation purposes
    def set_random_state(self, seed):
        self.random = np.random.RandomState(seed)
