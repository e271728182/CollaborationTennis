# CollaborationTennis
Third Assignment of DRL
Credit to Alexis Cook's DRL repository for the DDPG algorithm and the Model.py set up.
Credit to this blog for discussing the general architecture about higher number of nodes and using dropouts :https://medium.com/@kinwo/learning-to-play-tennis-from-scratch-with-self-play-using-ddpg-ac7389eb980e


A DDPG agent algorithm trained on the Tennis environment of Unity

This reportory contains files used to build a deep reinforcement learning network to solve the continuous control task . 2 agetns

Tennis game description: the state space is a vector of 24 dimensions that describe the environment where the agent is. The action space consist of a 3 dimensional vector with continuous values between -1 and 1. the reward is +0.1 for every time step the arm is at the right place

Description of files used
the 5 files below are used to build & train the models and have dependencies from Pytorch, Numpy, namedTuples,UnityEnvironment,Gym and Collections.

checkpoint_actor.pth : A trained actor for the DDPG model 
checkpoint_critic.pth :  trained critic for the DDPG model
Model.py:An actor-critic model build on a Deep neural network architecture with a tentative batch normalization and modified NN parameters. 

ddpg_agent4.py: An agent class that implements the DDPG algorithm. Agent class inherits the Qnetwork class in Model.py. this has been modified from the orifinal ddpg_agent.py from the Alexis Cook's DRL repository. 
Collaboration DDPG.pdf: Report describing model implementation, results and potential improvements

the forward method overrides the nn.module forward method as prescribed in the documentation the forward method does not need to be called explicitly

Tennis.ipynb: A jupyter notebook that connects to Unity environment to run the UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64'). the notebook contains the function ddpg which is used to train the ddpg network architecture with shared memory using the state,actions,reward as inputs. 

How to run the model
Everything is self contained in the Tennis.ipynb notebook. Simply run every cell and the dqn function will create a trained model as an ouput in the directory. Simply change the name of the output file in the ddpg function to save a new copy

How to set up the dependencies Detailed instructions are available on Alexis Cook's DRL repository https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md
