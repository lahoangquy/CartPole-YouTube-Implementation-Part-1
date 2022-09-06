# The CartPole system is a classic problem of reinforcement learning. The system consist of a pole
# which acts like an inverted pendulum attached to a cart using a join.
# The system is controlled by applying a force of +1 and -1 to the cart.
# The force applied to the cart can be controlled and the purpose is to swing the pole upward and stabailize it.
# This must be done without the cart falling to the ground.
# At every step, the agent can choose to move the cart left or right
# and it receives a reward of 1 for every time step that the pole is balanced.
# If the pole ever deviates by more than 15 degrees fro upright, the procedure ends.


import pygame as pygame
import gym
import numpy as np
env = gym.make('CartPole-v0')
np.random.seed(1)
env.seed(1)
HighReward = 0
BestWeights = None
for i in range(200):
    observation = env.reset()
    Weights = np.random.uniform(-1,1,4)
    SumReward = 0
    for j in range(1000):
        env.render()
        action = 0 if np.matmul(Weights,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        SumReward += reward
        print(i, j, Weights, observation, action, SumReward, BestWeights)
    if SumReward > HighReward:
        HighReward = SumReward
        BestWeights = Weights

# Calling the render() method will visually display the current state
# while subsequent calls to env.step() will allow us to interact with the environment returning the new states
# in response to the actions with which we call it.
# In this way we have adopted random actions at each step. At this point it is certainly useful to know
# What actions we are doing on the environment to decide future actions. The step() method returns exactly this.
# In effect , this method will return the 4 values. The first value is the observation.
# Observation is an environment specific object representing your observation of the environment.
# The second value is the reward and it is the amount of reward achieve by the previous action.
# The scale varies between environment but the goal is ways to increase the total reward.
# The third value is the done and this determines whether it is time to reset the environment again.
# Most (but not all) tasks are divided into well-defined episodes and done being True indicates that the episode is terminated.
# The final value is the info. This shows you dianostic information that is useful for debugging and learning.
#  A window will be displayed that contains our system and this is not stable and will soon move out of the screen.
# This is because the cart is pushed randomly without taking into account the position of the pole.
# To solve this problem that is to balance the pole it is important to set the push in the opposite direction to the inclination of the pole.
# So we have to set only 2 actions -1 and 1, pushing the cart to the left and right.
# But to do so, we do need know the data deriving from the observation of the environment at all times.
# As mentioned before, these pieces of data are returned by the step() method.
# In particular they are contained in the observation object.
# This object contains cart position and velocity, pole angle and velocity at tip. And these values becone the input of the problem
# As we have also anticipated , the system is balanced by applying a push to the car. There are 2 possible options
# The first one is push the cart to the left and the second one is push the cart to the right.
# It is clear that this is a binary classification problem: 4 inputs and a single binary output.
# First let's consider how we can extract the values to be sued as input.
# As can be seen below, we can see that the values contained in the observation objects are printed in the console.
# All of this will be very useful. Using the values that are returned from the environment observation,
# The agent has to decide on one of 2 possible actions: to move the cart left or right.
# Now we will face the most demanding phase: training of our system.
# The agent experience will be divided into a series of episodes.
# The initial state of the agent is randonly sampled by a distribution and the interaction preceeds until the environment reaches a terminal state
# This procedure is repeated for each episode with the aim of maximizing the total reward expection per episode and achieving a high level of performance in the fewest possible episode.
# In learning phase, we must estimate the evaluation function.
# This function must be able to evaluate through the sum of the rewards, the convenience or otherwise of a particular policy.
# In other words, we must approxiate the evaluation function. how can we do this?
# One solution is to use the artificial neural entwrok as a function approximator. Recall that the training of a neural network aims to identify the weights of the connections between neurons
# In this case, we will choose random values with weights for each episode.
# At the end, we will choose the combination of weights that has collected the max reward.
# The state of the system at a given moment is returned ot us by the observation object .
# To choose an action from the actual state, we cna use a linear combination of the weights and the observation.
# This is one of the most important special cases of function approximation in which the approximate function is a linear function of the weight vector w.
# For every state, s there is a real-valued vector x(s), with the smae number of components as w.
# Linear method approxiamte the state-value function by the inner product between w and x(s)
# In this way, we have specified the methodology that we intend to adopt for th esolution of the problem
