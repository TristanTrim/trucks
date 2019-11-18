
import random
import numpy as np
from gym import Env, spaces


# Structure of a graph for use in Truckenv:
# "N": number of nodes, eg: 3 -> Graph consists of Nodes 1, 2, and 3.
# "E": List of edge connections and the timecost between them:
#     (nodeIdA,nodeIdB,timeDistanceBetween)
#       as a convention: A < B
# "maxE": The most edges that will be found attached to any one node.
#       Used for figuring out how large the action-space needs to be.
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
    Trucks! Gotta have em, gotta send em on jobs! Pick up and drop off!

    The main API methods that users of this class need to know are:
        step
        reset
        render - not yet implemented
        close - not yet implemented
        seed - not yet implemented

    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    """

    def __init__(self,trucks=1,jobs=2,graph=simpleTestGraph):

        self.nTrucks=trucks
        self.nJobs=jobs
        self.graph=graph

        # transitions: {fromNodeId:[(toNodeId,timecost),],}
        self.transitions = {nodeId:[(0,0),] for nodeId in range(1,self.graph["N"]+1)}
        for edge in self.graph["E"]:
            #add go from node a to node b transition
            self.transitions[edge[0]].append((edge[1],edge[2]))
            #add go from node b to node a transition
            self.transitions[edge[1]].append((edge[0],edge[2]))
        # fill the rest of the options with the null action "hold"
        for key,val in self.transitions.items():
            for i in range(self.graph["maxE"]+1-len(val)):
                self.transitions[key].append((0,0))

        self.action_space = spaces.MultiDiscrete(((self.graph["maxE"]+4,)*self.nTrucks))
        self.observation_space = spaces.MultiDiscrete((self.graph["N"],)*self.nTrucks + (self.graph["N"],self.graph["N"])*self.nJobs)
        print(self.action_space.shape)
        print(self.action_space.dtype)
        print(self.observation_space.shape)
        print(self.observation_space.dtype)
        
        # Truck: (0:location,    1:driving to id,    2:driven dist,    3:job carried)
        #       location is id of node counting from 1 to # of nodes
        #       driving to id is the id of the adjacent node the truck is driving to
        #       driven dist is how far along the truck has driven from the previous node to the current node
        #       job carried is the id of the job counting from 1 to # of jobs
        self.trucks = np.array(((1,0,0,-1),)*self.nTrucks)
        # Job: (0:origin,    1:destination,    2:status)
        # status is 0:new job waiting, >0:truck carried by
        self.jobs = np.array(((-2,-2,-2),)*self.nJobs)


    def getJobIndexAt(self,locationId):
        """ Helper function, returns the index of the job found at a given location, or None """
        for i,job in enumerate(self.jobs):
            if job[0] == locationId:
                return(i)
        return(None)

    def newJob(self):
        """ Helper function, generates and returns a new, valid, job for insertion into job index """
        job=[0,0,-1]
        job[0]=random.randint(1,self.graph["N"])
        while(True):
            job[1]=random.randint(1,self.graph["N"])
            if(job[0]!=job[1]):
                break
        return(job)


    def step(self,actions):
        reward=0
        for trucki, truck in enumerate(self.trucks):
            # Truck: (0:location,    1:driving to id,    2:driven dist,    3:job carried)
            # Job: (0:origin,    1:destination,    2:status)
            # status is 0:new job waiting, >0:truck carried by
            action = actions[trucki]
            #return to last node
            if(action==0):
                if(truck[2]>0):
                    truck[2] -= 1
            #pickup
            elif(action==1):
                jobi = self.getJobIndexAt(truck[0])
                job = self.jobs[jobi]
                if(jobi!=None and job[2]==-1 and truck[3]==-1):
                    print("======================")
                    print("PICKUP!")
                    print("actions:")
                    print(actions)
                    print("prior state:")
                    print(self.trucks)
                    print(self.jobs)
                    job[2] = trucki
                    truck[3] = jobi
                    print("new state:")
                    print(self.trucks)
                    print(self.jobs)
                    print("======================")
            #dropoff
            elif(action==2):
                    # is carrying a job and that jobs destination is where the truck is.
                if(truck[3]!=-1 and self.jobs[truck[3]][1]==truck[0]):
                    print("======================")
                    print("WIN!")
                    print("actions:")
                    print(actions)
                    print("prior state:")
                    print(self.trucks)
                    print(self.jobs)
                    ###success!!!!!!!!!!!!!
                    reward += 1
                    self.jobs[truck[3]] = self.newJob()
                    self.trucks[trucki][3]=-1
                    print("new state:")
                    print(self.trucks)
                    print(self.jobs)
                    print("======================")
            #move
            else:
                destinationId, timecost = self.transitions[truck[0]][action-3]
                if(destinationId == 0):
                    # we don't actualy move to position 0, it just means to hold where we are
                    continue
                elif(destinationId == truck[1]):
                    if(truck[2]==timecost):
                        truck[0]=destinationId
                        truck[2]=0
                    else:
                        truck[2] += 1
                else:
                    if(truck[2]==0):
                        truck[1]=destinationId
                    else:
                        truck[2] -= 1
        return((self.trucks,self.jobs),reward,False,{})


    def reset(self):
        # set trucks back to id 1
        self.trucks = np.array(((1,0,0,0),)*self.nTrucks)
        # make random jobs
        for i in range(self.nJobs):
            self.jobs[i] = self.newJob()
        return((self.trucks,self.jobs))

def doRandomPolicy(env,max_steps=1000):
    reward = 0
    rewardSum = 0
    observation = env.reset()
    for t in range(max_steps):
        #env.render()
        rewardSum+=reward
        if(reward):
            print("@ timestep {}".format(t))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print("After {} timesteps: random agent exited with {} reward".format(t+1,rewardSum))

class valueIterationAgent():
    """yup, hard coded cause I hooked up the observation and action space wrong in my env..."""
    def __init__(self, environment):

        # Truck: (0:location,    1:driving to id,    2:driven dist,    3:job carried)
        # Job: (0:origin,    1:destination,    2:status)
        # status is 0:new job waiting, >0:truck carried by

        self.env = environment

      # nodes = self.env.graph["N"]
      # nActions = self.env.graph["maxE"]+4 #plus 4 '.' return, pick, drop, hold
      # nTrucks = self.env.nTrucks
      # nTruckPos = nodes
      # nTruckConfigs = nTruckPos**nTrucks #better than nTruck**nTruckPos amIright?
      # nJobs = self.env.nJobs
      # nJobPos = nodes*(nodes-1)
      # nJobConfigs = nJobPos**nJobs 
      # nStates = nTruckConfigs * nJobConfigs
      # nStateActions = nActions * nStates # I've coded myself into a dumb corner again! Whatever Don't optimize the system for use with a tabular system. We won't use a tab system for long!
        #alternatively:
        nodes = self.env.graph["N"]
        nActions = self.env.graph["maxE"]+4 #plus 4 '.' return, pick, drop, hold
        nTrucks = self.env.nTrucks
        nTruckPos = nodes
        nTruckConfigs = nTruckPos**nTrucks #better than nTruck**nTruckPos amIright?
        nOrigins = nodes + nTrucks
        nDest = nodes
        nJobs = self.env.nJobs
        nJobPos = nodes*(nodes-1)
        nJobConfigs = nJobPos**nJobs 
        nStates = nTruckConfigs * nJobConfigs
        print("There are {} distinct observable states... ish".format(nStates))
        self.stateValues = np.ndarray((nTruckPos,)*nTrucks+(nOrigins,nDest)*nJobs)
        self.stateValues.fill(0)
        print(self.stateValues.shape)
        print(self.stateValues[tuple(n-1 for n in self.stateValues.shape)])
        #self.policy = np.array((0,)*nStates)
        #(self.graph["N"],)*self.nTrucks + (self.graph["N"],self.graph["N"])*self.nJobs)

    def obsFromState(self,state):
        pass

    def train(self,**kwargs):
        for i in range(1000):
            pass


if (__name__=="__main__"):
    #env = gym.make("gym-trucks:trucks-v0") ## <-- this is how it would look if integrated nicely into gym.
    env = Truckenv()
    for i_episode in range(1):
        #doRandomPolicy(env)
        agt = valueIterationAgent(env)
        agt.load_stateValues("values.json")
        agt.train(value_dump_method = "overwrite", value_dump_file="values.json", statistics_file="stats.txt")
    env.close()

