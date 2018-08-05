[//]: # (Image References)

[image1]: ./banana.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, an agent has been trained to navigate (and collect bananas!) in a large, square world.


![Trained Agent][image1]

**Action and State space**

The state space consists of 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent learns how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

**Goal**

The goal of the agent is to smartly collect as many yellow bananas as possible while avoiding blue bananas.
A reward of +1 is provided to the agent for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

The environment is considered solved if the agent gets an average score of **+13** over 100 consecutive episodes.

**Algorithm**

A deep neural network termed as a deep Q-network is used to train a agent that learns successful policies to solve the environment.

### Getting Started

1. Follow the [instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install Unity ML-Agents. 

2. Navigate to the `p1_navigation/` folder, and run the command below to obtain a few more packages.
  ```
  pip3 install -r requirements.txt
  ```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Report.ipynb` to get started with training your own agent!  To use a Jupyter notebook, run the following command from the `p1_navigation/` folder:
```
jupyter notebook
```
and open `Report.ipynb` from the list of files.  Alternatively, you may prefer to work with the [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) interface.  To do this, run this command instead:
```
jupyter-lab
```
The agent can also be trained from the command line.
From the p1_navigation folder, run the following command:

```
python3 train_agent.py  
```
