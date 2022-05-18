# Autonomous-driving

Safe-Driving requires the knowledge of the
environment around and taking the actions accordingly.
Autonomous driving can be considered reliable when it
is able to do so in the real-time. In this project, we
have implemented value based and policy gradient based
Reinforcemnt Learning algorithms for making a vehicle
learn to drive on road using discrete and continuous
actions space. The Metadrive vehicle simulator is used
to implement RL algorithms like DQN, Double DQN,
Dueling DQN which are values based reinforcement
learning algorithms for an optimal policy to drive on road.PPO, a gradient based method is used for discrete as well
as continuous action space learning.


## Simulator

MetaDrive is an efficient and compositional driving
simulator which has the capability of generating infinite scenes
with various road maps and traffic settings for research of
generalize RL. It is very easy to install and has the capability to
achieve 300 FPS on a standard PC. The simulator has accurate
physics with multiple sensory input including Lidar, RGB
images, top-down semantic map and first-person view images.

### Observation Space

MetaDrive provides various kinds of sensory inputs. For
low-level sensors, RGB cameras, depth cameras and Lidar can
be placed anywhere in the scene with adjustable parameters
such as view field and the laser number. Meanwhile, the high-
level scene information including the road information and
nearby vehicles’ information like velocity and heading can
also be provided as the observation. Three types of existing
Observation are

• State Vector

• Top-down Semantic Maps

• First-View Images

C. Action Space

MetaDrive receives normalized action as input to control
each target vehicle. At each environmental time step,
MetaDrive converts the normalized action into the steering
(degree), acceleration (hp) and brake signal (hp). MetaDrive
provides the functionality of adding more dimensions in the
action space. This allow the user to write environment wrapper
that introduce more input action dimensions.

### Rewards

The default reward function in MetaDrive only contains a
dense driving reward and a sparse terminal reward. The dense
reward is the longitudinal movement toward destination in
Frenet coordinates. The sparse reward +20 is given when the
agent arrives the destination. MetaDrive calculates a complex
reward function that enables user to customize their reward
functions from config dict. The complete reward function is
composed of four parts as follows The driving reward denote the longitudinal coordinates of the
target vehicle in the current lane of two consecutive time steps,
providing dense reward to encourage agent to move forward.
Speed reward incentives agent to drive fast. The
termination reward contains a set of sparse rewards. At the
end of episode, other dense rewards will be disabled and only
one sparse reward will be given to the agent at the end of the
episode according to its termination state.
It also implements success reward,out of the road penalty
and crash vehicle penalty.

## Results

A. DQN

https://user-images.githubusercontent.com/94932358/168958985-5a9594c9-fc67-4bc3-ba08-8f2f8dc329a2.mp4


B. PPO


https://user-images.githubusercontent.com/94932358/168959737-192b3589-87dc-4610-8a7a-5ee7752e0adc.mp4




