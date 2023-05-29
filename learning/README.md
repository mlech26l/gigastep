# Learning in Gigastep

## MARLlib

### Installation
```
$ cd MARLlib
$ pip install -e .
$ pip install ma_gym
```
```
$ cd ray
$ python python/ray/setup-dev.py -y
```

### Run
```
$ python examples/train_gigastep.py
```

## evosax

### Installation
```
$ pip install evosax
$ pip install tqdm
```

### Run
```
$ python3 learning/evosax/scripts/run_open_es.py
```

### Notes
Evosax is under development and some features may move / not be synced across functions.\
For Tensorboard logging this setup should work but probably not all steps necessary.
```
$ pip install tensorboard==2.7.0
$ pip install torch==1.10.0
$ pip install moviepy
$ pip install setuptools==59.5.0
```
This reads images from vmap scan. Consider doing single rollout if memory efficiency becomes an issue.

### MAVA

### Installation
```
$ sudo apt-get install swig
$ cd Mava
$ pip install -e . # don't do pip install id-mava[reverb,jax,envs] since that is out-of-date
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/gigastep/lib # you need this for every runs; perhaps put it in bashrc

$ pip install --upgrade dm-haiku # to fix "AttributeError: 'Config' object has no attribute 'jax_experimental_name_stack'"
```
