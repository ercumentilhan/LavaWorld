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
        self.agent_pos = None
        self.goal_pos = None
        self.grid = None


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

        self.random = np.random.RandomState(seed)

        self.state = None

    # ------------------------------------------------------------------------------------------------------------------

    def reset(self):
        self.state = State()

        # Initialize level structure:
        self.state.grid = np.zeros((self.height, self.width), dtype=int)
        self.state.grid[0, :] = 1
        self.state.grid[:, 0] = 1
        self.state.grid[-1, :] = 1
        self.state.grid[:, -1] = 1

        self.state.grid[5, :] = 1
        self.state.grid[:, 5] = 1
        self.state.grid[5, 2] = 0
        self.state.grid[5, 8] = 0
        self.state.grid[8, 5] = 0

        # self.state.grid[6, :] = 1
        # self.state.grid[:, 6] = 1
        # self.state.grid[6, 3] = 0
        # self.state.grid[6, 9] = 0
        # self.state.grid[9, 6] = 0

        # print(self.state.grid)

        self.state.t = 0
        self.state.done = False

        self.state.agent_pos = [1, 4]
        self.state.goal_pos = [1, 6]

        return self.state_to_obs(self.state)

    # ------------------------------------------------------------------------------------------------------------------

    def transitive(self, state, action):

        if self.random.random_sample() < 0.05:
            p = [1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0]
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

        if state.grid[state.agent_pos[0], state.agent_pos[1]] == 1:  # Lava
            state.done = True
        elif state.agent_pos[0] == state.goal_pos[0] and state.agent_pos[1] == state.goal_pos[1]:
            state.done = True
            reward = 1.0

        state.t += 1
        state.done = state.done or state.t >= self.t_max

        return reward

    # ------------------------------------------------------------------------------------------------------------------

    def state_to_obs(self, state):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not state.done:
            obs[state.agent_pos[0], state.agent_pos[1], 0] = 1
            obs[state.goal_pos[0], state.goal_pos[1], 1] = 1
            lavas = np.where(state.grid == 1)  # TODO: Remove
            for n in range(len(lavas[0])):
                obs[lavas[0][n], lavas[1][n], 2] = 1
        return obs

    # ------------------------------------------------------------------------------------------------------------------

    def optimal_action(self, state=None):
        if state is None:
            state = self.state

        # TODO: Save actions

        grid = Grid(matrix=(1 - state.grid))
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
        obs_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        obs_image[state.agent_pos[0], state.agent_pos[1], :] = self.agent_color
        obs_image[state.goal_pos[0], state.goal_pos[1], :] = self.goal_color

        lavas = np.where(state.grid == 1)  # TODO: Remove
        for n in range(len(lavas[0])):
            obs_image[lavas[0][n], lavas[1][n], :] = self.lava_color

        obs_image = cv2.resize(obs_image, (330, 330), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
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


e = Environment(0)
e.reset()