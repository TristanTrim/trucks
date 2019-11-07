
from gym import Env, spaces


simpleTestEdges = (
        (1,3,2),
        (2,3,2),
        (2,4,3),
        (3,5,1),
        (4,5,2),
        )

class Truckenv(Env):
    """
    Trucks! Gotta have em, gotta send em on jobs!
    """
    def __init__(slef,trucks=1,jobs=3,edges=simpleTestEdges):

if (__name__=="__main__"):
    #env = gym.make("trucks-v0")
    env = Truckenv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            #env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

