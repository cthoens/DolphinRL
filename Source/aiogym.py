from gym import Env


class AsyncEnv(object):
    """
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    async def step(self, action):
        """
        """
        raise NotImplementedError

    async def reset(self):
        """
        """
        raise NotImplementedError

    async def render(self, mode='human'):
        """
        """
        raise NotImplementedError

    async def close(self):
        """
        """
        pass

    async def seed(self, seed=None):
        """
        """
        return

    @property
    def unwrapped(self):
        """
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False


class AsyncEnvWrapper(AsyncEnv):
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

    async def step(self, action):
        return self.env.step(action)

    async def reset(self):
        return self.env.reset()

    async def render(self, mode='human'):
        return self.env.render(mode)

    async def close(self):
        return self.env.close()

    async def seed(self, seed=None):
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env

    def __str__(self):
        return self.env.__str__()

    def __enter__(self):
        return self.env.__enter__()

    def __exit__(self, *args):
        return self.env.__exit__()
