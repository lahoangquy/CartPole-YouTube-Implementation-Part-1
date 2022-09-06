# We did import numpy library which will be used to set the seed value
# We do need gym library to help us to define the environment
# We also import some functions of the keras library to build a neural network model
import numpy as np
import gym
# First the Sequential model is imported it is a linear stack of layers.
# Then some keras Layers are imported: Dense, Activation and Flatten
# A dense is a fully connected neural network layer.
# The activation layer applies an activation function to an output
# The Flatten layer which will flattens the input and does not affect the batch size.
# Finally the Adam optimizer.
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
# The Keras-RL library implements some state of the art deep reinforcement learning algorithm
# It seamlessly integrates with deep learning keras library, DQNAgent, a polocy and a memory model are imported
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
# See the seed value
# The seed() function sets the seed of the random number generator which is useful for creating simulations or random objects that can be reproduced.
# You have to use this function every time you want to get a reproduciable random result.
np.random.seed(1)
env.seed(1)
# Extract the actions that available to the agent
# The nb_actions variable now contains all of the actions that are availabel in the selected environment.
# The Gym will not always tell you what these actions mean, only which ones are available.
nb_actions = env.action_space.n
# Simple neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
# One problem with using the DQN is that the neural network used in the algorithms tends to forge previous experiences because it overwrites them with new experiences
# Therefore, we do need a list of previous experiences and observations to reform the model from the previous experiences
# For this reason, a Memory variable is defined which will contain the previous experiences
memory = SequentialMemory(limit=50000, window_length=1)
# Set the policy variable
policy = BoltzmannQPolicy()
# We define the agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
            nb_steps_warmup=10,target_model_update=1e-2,
            policy=policy)
# Compiling the model
# The compile command will comppile an agent and the underlying models to be used for training and testing
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# The fit() function trains the agent on the given environment
# At the end of training, it is important to save the obtained weights
dqn.fit(env, nb_steps=1000, visualize=True, verbose=2)
# Saving the weight of a network or an entire structure takes place in an HDF5 files which is efficient and flexible storage system that support complex multidimensional datasets.
# Finally we will evaluate our algorithm for 5 episodes
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
dqn.test(env, nb_episodes=5, visualize=True)
# In this way, the balance of the system is assured.
