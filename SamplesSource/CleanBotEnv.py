"""
CleanBotEnv
===========

Provides
    - The Clean Bot Environment, in which a bot moves in a square grid and cleans dirty tiles. See CleanBotEnv class for
      details.
"""

from gym import Env, spaces
import enum
import numpy as np


class CleanBotEnv(Env):
    """
    Square grid environment in which some of the tiles are dirty. A bot can move along the coordinate
    axes and get rewards for cleaning the dirty tiles
    """

    class BotActions(enum.Enum):
        """Actions the bot can perform"""
        NORTH = 0
        """Move north"""
        EAST = 1
        """Move east"""
        SOUTH = 2
        """Move south"""
        WEST = 3
        """Move west"""
        CLEAN = 4
        """Move clean the tile the bot is currently on"""

    class TileState(enum.Enum):
        """State each tile can be in"""
        CLEAN = 0
        """The tile is clean"""
        DIRTY = 1
        """The bot is currently on the tile"""
        BOT = 2

    def __init__(self, width, dirty_rate=0.5):
        self.width = width
        """The width of the grid"""

        self.max_steps = width * width * 2
        """The maximum number of step before an episode terminates"""

        self.dirty_rate = dirty_rate
        """Upper bound of dirty cells as percentage of total the number of tiles"""

        self.step_count = 0
        """The number of steps taken in the current episode"""

        self.bot_x = 0
        """The horizontal location of the bot in the grid"""

        self.bot_y = 0
        """The vertical location of the bot in the grid"""

        self.state = None
        """2D numpy array holding the state of each cell of the grid. Does NOT contain the bot state."""

        self.dirty_count = None
        """The number of dirty cells currently in the grid"""

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.width, self.width), dtype=np.int)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.bot_x = 0
        self.bot_y = 0
        self.state = np.zeros(shape=(self.width, self.width), dtype=np.int)

        # Maximum number of cells to mark dirty as a percentage of the total number of cells
        dirty_upper_bound = max(1, int(self.width * self.width * self.dirty_rate))
        self.dirty_count = 0
        # Mark cells dirty. The same cell might be randomly chosen more than once, which reduced the number of cells
        # That are actually marked dirty
        for i in range(dirty_upper_bound):
            dirty_cell = tuple(np.random.randint(self.width, size=2))
            if self.state[dirty_cell] != self.TileState.DIRTY.value:
                self.state[dirty_cell] = self.TileState.DIRTY.value
                self.dirty_count += 1
        return self._get_obs()

    def render(self, mode='human'):
        def state_to_char(observation):
            """Convert state value to character representation"""
            if observation == self.TileState.CLEAN.value:
                return "-"
            if observation == self.TileState.DIRTY.value:
                return "d"
            if observation == self.TileState.BOT.value:
                return "b"

        rendering = [[state_to_char(self.state[y, x]) for x in range(self.width)] for y in range(self.width)]
        # Bot is only visible if the cell it is on is not dirty
        if self.state[self.bot_y, self.bot_x] != self.TileState.DIRTY.value:
            rendering[self.bot_y][self.bot_x] = "b"
        for col in rendering:
            for cell in col:
                print(cell, end="")
            print()
        print()

    def step(self, action):
        if self.step_count == self.max_steps or self.dirty_count == 0:
            raise Exception(f"Episode has finished ({self.step_count, self.dirty_count})")
        self.step_count += 1
        reward = 0
        if action == self.BotActions.SOUTH.value and self.bot_y < self.width-1:
            self.bot_y += 1
        if action == self.BotActions.NORTH.value and self.bot_y > 0:
            self.bot_y -= 1
        if action == self.BotActions.EAST.value and self.bot_x < self.width-1:
            self.bot_x += 1
        if action == self.BotActions.WEST.value and self.bot_x > 0:
            self.bot_x -= 1
        if action == self.BotActions.CLEAN.value:
            if self.state[self.bot_y, self.bot_x] == self.TileState.DIRTY.value:
                self.state[self.bot_y, self.bot_x] = self.TileState.CLEAN.value
                self.dirty_count -= 1
                reward = self.max_steps - self.step_count

        episode_done = self.dirty_count == 0

        return self._get_obs(), reward, episode_done or self.step_count == self.max_steps, None

    def _get_obs(self):
        """Convert internal state to observation setting the state of the tile the bot it on."""
        full_state = np.copy(self.state)
        if full_state[self.bot_y, self.bot_x] != self.TileState.DIRTY.value:
            full_state[self.bot_y, self.bot_x] = self.TileState.BOT.value
        return full_state
