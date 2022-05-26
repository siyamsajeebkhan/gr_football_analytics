#### Creating a virtual environment using conda (recommended)
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) using these instructions, if it is not already installed in your system
```shell
conda create -n football-env python=3.9
conda activate football-env
```
### On your computer

#### 1. Install required packages
#### Linux
```shell
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

python3 -m pip install --upgrade pip setuptools psutil wheel
```

#### 2. Install GFootball
#### Option a. From PyPi package (recommended)
```shell
python3 -m pip install gfootball
```

#### Option b. Installing from sources using GitHub repository 
(On Windows you have to install additional tools and set an environment variable, see 
[Compiling Engine](gfootball/doc/compile_engine.md#windows) for detailed instructions.)

```shell
git clone https://github.com/google-research/football.git
cd football
```

Optionally you can use [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```shell
python3 -m venv football-env
source football-env/bin/activate
```

Next, build the game engine and install dependencies:

```shell
python3 -m pip install .
```
This command can run for a couple of minutes, as it compiles the C++ environment in the background.
If you face any problems, first check [Compiling Engine](gfootball/doc/compile_engine.md) documentation and search GitHub issues.

#### 3. Install stable-baselines3
```shell
pip install stable-baselines3==1.5.0
```

#### 4. Install jupyter notebook 
```shell
pip install notebook
```

#### Playing around with the notebooks
1. gfootball-stable-baselines3-DQN.ipynb - for running DQN experiments including DDQN.
2. gfootball-stable-baselines3-PPO.ipynb - for running PPO experiments.
3. gfootball-stable-baselines3-test-agent.ipynb - for testing how an agent performs
