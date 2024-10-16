#first we have to import the libraries.
import warnings
import numpy as np
import gym

#first we must suppress the specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#loading the enviroment with a rendering mode specified
env = gym.make('CartPole-v1', render_mode="human")

#now we have to intialize the environment.This will be required to get the start/initial state.
state = env.reset()

#print the state space and action space :
print("State space:", env.observation_space)
print("Action space:", env.action_space)

#now we have intialzed the envionment and got the intial state.

#for the second part we have to create the actions.
#we ate going to create few steps by using random actions

for i in range(10) :
    env.render() #this is for rendering the environment for the visualization of the process.
    action = env.action_space.sample() #implement a random actions

    #then taking a step in the enviroment for the random action.
    step_result = env.step(action)

    #checking the number of values returned
    #then unpack accordingly
    if len(step_result) == 4:
        next_state, reward, done, info = step_result
        terminated = False
    else:
        next_state, reward, done, truncated, info = step_result
        terminated = done or truncated

    print(f"Action: {action}, Reward: {reward}, Next_state: {next_state}, Done: {done}, Info: {info}")

    if terminated:
        #This function is to reset the environment if the episode is finshed
        state = env.reset()

#finally we must create the function to close the enviroment when the problem is solved.
env.close()
