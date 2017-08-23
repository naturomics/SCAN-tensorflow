import deepmind_lab


class DeepMindLabAgent(object):
    def __init__(self, level, obsevations, config={}):
        # Construct and start the environment.
        env = deepmind_lab.Lab(level, obsevations, config)
        env.reset()

    def step(self, reward, unused_frame):
        pass


if __name__ == '__main__':
    pass
