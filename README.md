#### Creating a virtual environment using conda (recommended)
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) if it is not already installed in your system. Then create and activate a virtual environment using the following commands:
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

#### macOS
First install [brew](https://brew.sh/). It should automatically install Command Line Tools.
Next install required packages:

```shell
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

python3 -m pip install --upgrade pip setuptools psutil wheel
```


#### Windows
Install [Git](https://git-scm.com/download/win) and [Python 3](https://www.python.org/downloads/).
Update pip in the Command Line (here and for the **next steps** type `python` instead of `python3`)
```commandline
python -m pip install --upgrade pip setuptools psutil wheel
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

#### Special Note
Please replace the original ```stable_baselines3/common/offpolicyalgorithm.py``` and ```stable_baselines3/common/torch_layers.py``` files of the sb3 library with the files provided in the ```changes to_sb3``` directory. I will incorporate these changes in a smarter way later instead of this brute force way. The reason behind this is that I have changed the code of the ```offpolicyalgorithm.py``` for changing the action sampling policy of DQN and the ```torch_layers.py``` file has been changed to incorporate changes such as dropout in the policy network both for DQN and PPO. 


#### 4. Install jupyter notebook 
```shell
pip install notebook
```

#### Visualization with Tensorboard
```shell
pip install tensorboard
```

#### Trained IL agent checkpoints
[IL agent checkpoints](https://drive.google.com/drive/folders/1QwyPsWdGfJMhjEcBIhNot15iij_VRx_U?usp=sharing)
* epoch=146-step=479366.ckpt - IL agent without batch normalization

#### Playing around with the notebooks
1. [gfootball-stable-baselines3-DQN.ipynb](https://github.com/siyamsajeebkhan/gr_football_analytics/blob/6c68e826b418d083eff010f9cce01cc52acfe54a/notebooks/gfootball-stable-baselines3-DQN.ipynb) - for running DQN experiments including DDQN.
2. gfootball-stable-baselines3-PPO.ipynb - for running PPO experiments.
3. gfootball-stable-baselines3-test-agent.ipynb - for testing how an agent performs
