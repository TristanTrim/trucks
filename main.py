
from gym import Env, spaces

# (nodeIdA,nodeIdB,timeDistanceBetween)
# A < B .`. there is no duplication
simpleTestGraph = {
        "N":5,
        "E":(
            (1,3,2),
            (2,3,2),
            (2,4,3),
            (3,5,1),
            (4,5,2),
            ),
        "maxE":3,
        }

class Truckenv(Env):
    """
    Trucks! Gotta have em, gotta send em on jobs!

    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    """
    def __init__(self,trucks=1,jobs=3,graph=simpleTestGraph):

        self.nTrucks=trucks
        self.nJobs=jobs
        self.graph=graph

        self.action_space = spaces.MultiDiscrete((self.graph["maxE"]+3,)*self.nTrucks)
        self.observation_space = spaces.MultiDiscrete((self.graph["N"],)*self.nTrucks + (self.graph["N"],self.graph["N"])*self.nJobs)

        self.trucks=nuply array of a good size, not too big, don't want to be wasteful 'eh?

    def step(self,action):


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

