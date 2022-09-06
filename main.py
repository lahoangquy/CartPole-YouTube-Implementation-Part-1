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
# The HighReward variable will ontain the maximum reward that is obtained up to current episode, this value will be used as a comparison value
# The BestWeight variable will contain the sequence of weights that has registerd the maximum reward.
# We can now implement the best weight sequence search through an iterative procedure for expisodes
for i in range(200):
    observation = env.reset()
    # To fix the weights we have used the np.random.uniform() function. This function draws samples from a uniform distribution
    # The samples are uniformly distributed over the haft-open interval (low and high). It includes low but excludes high
    # In order words, any value within the given interval is equally likely to be drawn by a uniform distribution
    # 3 parameters have been passed; the lower boundary of the output interval. its upper boundary and the output shape.
    # In our case, we requested 4 random values in the interval (-1,1). After doing this, we need to initialize the sum of the rewards
    Weights = np.random.uniform(-1,1,4)
    SumReward = 0
    # We implement another iterative cycle to determine the maximum reward we can get with these weights.
    # With this instruction, the training phase ends which will give us the sequence of weights that best approximates the evaluation function.
    # We can test the system
    # When the training phase is achieved, in practice it means that we have found the sequenc of weights that best approximates this function that is the one that has returned the best reward.
    # Now we ahve to test the system with these values to check whether the pole is able to stand for at least 100 time steps
    # Since we have done with the training phase to make the whole testing easily understandable, we will report the whole code block and then comment on it in detail on a line-by-line basis.
    for j in range(100):
        env.render() # display the current state
# We have to decide the action. To decide the action we have used a linear combination of 2 vectors: weights and observation
# To perform a linear combination we have used the np.matmul() function and it implements a matrix product of 2 ways
# So if this product is 0 (move left), otherwise the action is 1 (move right)
# The -ve product means that the pole is titled to the left. To balance this trend, it is important to push the cart to the left.
# A positive product means that the pole is title to the right. To balance this trend, it is important to push the cart to the right.
        action = 0 if np.matmul(Weights,observation) < 0 else 1
# We use the step() method to return the new states in response to the action with which we call it.
# Obviously, the action we pass to the method is the one we have just decided
# We print the step numbers and that action that has been deicded on for visual control of the flow.
# After running the below code, we can verify that after the training phase the system is able to keep the pole in equilibrium 1000 steps
        observation, reward, done, info = env.step(action)
        SumReward += reward
        print(i, j, Weights, observation, action, SumReward, BestWeights)
# At the end of the current iteration, we can make a comparison to check whether the total reward obtained is the highest one obtained so far
    if SumReward > HighReward:
# If it is the highest reward obtained so far, update the high reward parameter.
        HighReward = SumReward
# Once this is done, fix the sequence of Weights of the current step as the best one
        BestWeights = Weights
