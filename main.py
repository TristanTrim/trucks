
from time import sleep
import pickle
from copy import deepcopy
import random
import numpy as np
from gym import Env, spaces


# Structure of a graph for use in Truckenv:
# "N": number of nodes, eg: 3 -> Graph consists of Nodes 1, 2, and 3.
# "E": List of edge connections and the timecost between them:
#     (nodeIdA,nodeIdB,timeDistanceBetween)
#           timeDistanceBetween MUST be 1 or greater!
#       as a convention: A < B
# "maxE": The most edges that will be found attached to any one node.
#       Used for figuring out how large the action-space needs to be.
simpleTestGraph = {
        "N":5,
        "E":(
            (1,3,1),
            (2,3,1),
            (2,4,1),
            (3,5,1),
            (4,5,1),
            ),
        "maxE":3,
        }
#optimal for this graph should be 0.2857 wins / step
twoNodeGraph = {
        "N":2,
        "E":(
            (1,2,1),
            ),
        "maxE":1,
        "render":
            [[["T^---__", "T_---_^"],
              ["t^---_#", "t#---_^"]],

             [["_^---T_", "__---T^"],
              ["_^---t#", "_#---t^"]]]
        }


threeNTemplate="{}{}---{}{}---{}{}"
threeNRender=[]
for truckLoc in range(3):
    threeNRender+=[[]]
    for origin in range(3):
        threeNRender[truckLoc]+=[[]]
        for dest in range(3):
            data=["_","_","_","_","_","_"]
            if(origin == 0):
                truckGlyph = "T"
            else:
                truckGlyph = "t"
                if(origin<=dest):
                    data[(origin-1)*2+1] = "#"
                else:
                    data[(origin)*2+1] = "#"
            data[dest*2+1] = "^"
            data[truckLoc*2] = truckGlyph

            threeNRender[truckLoc][origin]+=[threeNTemplate.format(*data)]

threeNodeGraph = {
        "N":3,
        "E":(
            (1,2,1),
            (2,3,1),
            ),
        "maxE":2,
        "render":threeNRender,
        }

triangleTemplate="\n  {}{}\n /  \\\n{}{}--{}{}"
triangleRender=[]
for truckLoc in range(3):
    triangleRender+=[[]]
    for origin in range(3):
        triangleRender[truckLoc]+=[[]]
        for dest in range(3):
            data=["_","_","_","_","_","_"]
            if(origin == 0):
                truckGlyph = "T"
            else:
                truckGlyph = "t"
                if(origin<=dest):
                    data[(origin-1)*2+1] = "#"
                else:
                    data[(origin)*2+1] = "#"
            data[dest*2+1] = "^"
            data[truckLoc*2] = truckGlyph

            triangleRender[truckLoc][origin]+=[triangleTemplate.format(*data)]

triangleGraph = {
        "N":3,
        "E":(
            (1,2,1),
            (1,3,1),
            (2,3,1),
            ),
        "maxE":2,
        "render":triangleRender,
        }

def ndindex(nd,index):
 if(len(index)==1):
  return(nd[index[0]])
 else:
  return(ndindex(nd[index[0]],index[1:]))
def readableAction(action):
    if(action<4):
        return(["back","pick","drop","hold"][action])
    return("go:{}".format(action-4))

#######
# Env #
#######

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

    def simpleObs(self):
        state = (self.trucks,self.jobs)
        obs = [[],[]]
        # trucks
        for i in range(self.nTrucks):
            obs[0]+=[state[0][i][0]]
        # jobs
        for i in range(self.nJobs):
            origin = None
            if(state[1][i][2]!=-1):#status not on truck
                origin=state[1][i][2]
            elif (state[1][i][0]<state[1][i][1]):
                origin=state[1][i][0]-1 +self.nTrucks # -1 because the id is offset 0 if there is one truck
            else:
                origin=state[1][i][0]-2 +self.nTrucks # -2 because truck thing and origin never = destination
            obs[1]+=[[origin,state[1][i][1]]]
        return (tuple(obs))

    def simpleObsFlat(self):
        """When indexing a ndarray you need all the truck and job values in one flat list"""
        obs = self.simpleObs()
        obsFlat = [x-1 for x in obs[0]]
        for x in obs[1]:
            obsFlat+=[x[0],x[1]-1]
        return(obsFlat)

    def render(self,end="\n"):
        """
        This will throw an exception if your graph doesn't have a render doodad in it.
        """
        nd = self.graph["render"]
        index = self.simpleObsFlat()
        print(ndindex(nd,index),end=end)
        sleep(1)

    def __init__(self,trucks=1,jobs=1,graph=simpleTestGraph):

        self.nTrucks=trucks
        self.nJobs=jobs
        self.graph=graph
        self.nActions = self.graph["maxE"]+4

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

        self.action_space = spaces.MultiDiscrete(((self.nActions,)*self.nTrucks))
        self.observation_space = spaces.MultiDiscrete((self.graph["N"],)*self.nTrucks + (self.graph["N"],self.graph["N"])*self.nJobs)
        print(self.action_space.shape)
        print(self.action_space.dtype)
        self.generateActionsList()
        print("actions: {}".format(self.actions))
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
                    if(printPickups):
                        print("======================")
                        print("PICKUP!")
                        print("actions:")
                        print(actions)
                        print("prior state:")
                        print(self.trucks)
                        print(self.jobs)
                    job[2] = trucki
                    truck[3] = jobi
                    if(printPickups):
                        print("new state:")
                        print(self.trucks)
                        print(self.jobs)
                        print("======================")
            #dropoff
            elif(action==2):
                    # is carrying a job and that jobs destination is where the truck is.
                if(truck[3]!=-1 and self.jobs[truck[3]][1]==truck[0]):
                    if(printWins):
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
                    if(printWins):
                        print("new state:")
                        print(self.trucks)
                        print(self.jobs)
                        print("======================")
            #move
            else:
                destinationId, timecost = self.transitions[truck[0]][action-3]
                if(truck[2]==0):
                    truck[1]=destinationId
                if(destinationId == 0):
                    # we don't actualy move to position 0, it just means to hold where we are
                    continue
                elif(destinationId == truck[1]):
                    truck[2] += 1
                    if(truck[2]==timecost):
                        truck[2]=0
                        truck[0]=destinationId
                else:
                    truck[2] -= 1
        return((self.trucks,self.jobs),reward,False,{})


    def reset(self):
        # set trucks back to id 1
        self.trucks = np.array(((1,0,0,-1),)*self.nTrucks)
        # make random jobs
        for i in range(self.nJobs):
            self.jobs[i] = self.newJob()
        return((self.trucks,self.jobs))

    def generateActionsList(self):
        actions =[]
        for a in range(self.nActions):
            actions+=[[a]]
        for t in range(self.nTrucks-1):
            actions = [ x+[a] for a in range(self.nActions) for x in actions ]
        self.actions = tuple(tuple(x) for x in actions)

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


###########
## AGENT ##
###########

class Agent():
    """yup, hard coded cause I hooked up the observation and action space wrong in my env..."""
    def __init__(self, environment):

        # Truck: (0:location,    1:driving to id,    2:driven dist,    3:job carried)
        # Job: (0:origin,    1:destination,    2:status)
        # status is 0:new job waiting, >0:truck carried by

        self.env = environment

        self.gamma = 0.9
        self.alpha = 0.1
        self.eplislon = .1

        nodes = self.env.graph["N"]
        nActions = self.env.graph["maxE"]+4 #move along one of the possible edges, or chose from the 4 options: return, pick, drop, hold
        nTrucks = self.env.nTrucks
        nTruckPos = nodes
        nTruckConfigs = nTruckPos**nTrucks #better than nTruck**nTruckPos amIright?
        nJobs = self.env.nJobs
        nOrigins = nodes + nTrucks - 1 ##minus one because jobs never go from where they are to where they are.
        nDest = nodes
        nJobPos = nOrigins*(nDest)
        nJobConfigs = nJobPos**nJobs 
        nStates = nTruckConfigs * nJobConfigs
        print("There are {} distinct observable states.".format(nStates))

        self.stateValues = np.ndarray((nTruckPos,)*nTrucks+(nOrigins,nDest)*nJobs)
        self.stateValues.fill(30)
        print("%d bytes" % (self.stateValues.size * self.stateValues.itemsize))
        print("%d kilobytes" % (self.stateValues.size * self.stateValues.itemsize / 1024))
        print("%d megabytes" % (self.stateValues.size * self.stateValues.itemsize /1024 /1024))
        print(self.stateValues.shape)
        print(self.stateValues[tuple(n-1 for n in self.stateValues.shape)])
        self.policy = np.ndarray((nTruckPos,)*nTrucks+(nOrigins,nDest)*nJobs)
        self.policy.fill(0)
        #self.policy = np.array((0,)*nStates)
        #(self.graph["N"],)*self.nTrucks + (self.graph["N"],self.graph["N"])*self.nJobs)


    def bestAction(self,maxLookahead=5):
        # current observation
        currAgtObs = self.env.simpleObsFlat()#self.stateToObsFlat((self.env.trucks,self.env.jobs))
        # setup and find action leading to best new observation
        best = 0
        bA = self.env.actions[0]
        reward = 0
        for action in self.env.actions:
            testEnv = deepcopy(self.env)
            ## loop until the environment looks different to the agent
            ## TODO: THIS ALGORITHM COULD BE BETTER STREAMLINED
            if(all(x<4 for x in action)):
                observation, reward, done, info = testEnv.step(action)
                newAgtObs = testEnv.simpleObsFlat()#self.stateToObsFlat(observation)
            else:
                for i in range(maxLookahead):
                    observation, reward, done, info = testEnv.step(action)
                    newAgtObs = testEnv.simpleObsFlat()#self.stateToObsFlat(observation)
                    if(currAgtObs!=newAgtObs or reward!=0):#why? because it won't work right if it doesn't see the environment change based on its actions
                        break
            ## compare this hypothetical to the other actions
            try:
                stateActionValue = reward + self.gamma*self.stateValues[tuple(newAgtObs)]
            except:
                raise Exception("Observation outside state space.\nDid you remember to initialize environment with reset function?")
            if (stateActionValue > best):
                best = stateActionValue
                bA = action
        # update current state value based on best action
        self.stateValues[tuple(currAgtObs)] = self.stateValues[tuple(currAgtObs)] \
                                        +self.alpha*(best - self.stateValues[tuple(currAgtObs)])


        return(bA)

    def egreedy(self):
        if (random.random() > self.eplislon):
            action = agt.bestAction()
        else:
            action = env.action_space.sample()

        return(action)

    def saveStateValues(self,filename):
        fl = open(filename,'wb')
        fl.write(self.stateValues.dumps())
        fl.close()

    def loadStateValues(self,filename):
        fl = open(filename,'rb')
        self.stateValues = pickle.load(fl)

    def train(self,**kwargs):
        for i in range(1000):
            pass



## Training demonstration:


def trial(agt,env,demo_len):
    for i in range(demo_len):
        env.render(end="")
        action = agt.bestAction()
        print("    ",end="")
        print(readableAction(action[0]),end="")
        print("    ",end="")
        observation, reward, done, info = env.step(action)
        if(reward):
            print("Win!")
        else:
            print("meh.")

def training_demo(agt,env, demo_len = 13, train_len = 10, train_epoch = 3):
    for j in range(train_epoch):
        observation = env.reset()
        print("### value function")
        print(agt.stateValues)
        trial(agt,env,demo_len)
        print("        "+"_"*train_len)
        print("training",end="",flush=True)
        for k in range(train_len):
            print(".",end="",flush=True)
            for i in range(200):
                action = agt.bestAction()
                observation, reward, done, info = env.step(action)
        print()
    observation = env.reset()
    print("### value function")
    print(agt.stateValues)
    trial(agt,env,demo_len)

########
# Main #
########

printPickups = False
printWins = False
if (__name__=="__main__"):

    #env = gym.make("gym-trucks:trucks-v0") ## <-- this is how it would look if integrated nicely into gym.
    #env = Truckenv(trucks=1,jobs=1,graph=twoNodeGraph)
    env = Truckenv(trucks=1,jobs=1,graph=threeNodeGraph)
    #env = Truckenv(trucks=1,jobs=2,graph=simpleTestGraph)
    #env = Truckenv(trucks=2,jobs=3,graph=simpleTestGraph)

### 2 node graph
    env = Truckenv(trucks=1,jobs=1,graph=twoNodeGraph)
    agt = Agent(env)
    training_demo(agt,env, demo_len = 13, train_len = 5, train_epoch = 1)
    env.close()

### 3 node graph
    env = Truckenv(trucks=1,jobs=1,graph=threeNodeGraph)
    agt = Agent(env)
    training_demo(agt,env, demo_len = 13, train_len = 15, train_epoch = 2)
    env.close()

### triangle graph
    env = Truckenv(trucks=1,jobs=1,graph=triangleGraph)
    agt = Agent(env)
    training_demo(agt,env, demo_len = 13, train_len = 20, train_epoch = 2)
    env.close()
