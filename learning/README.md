# Learning in Gigastep

## MARLlib

### Installation
```
$ cd MARLlib
$ pip install -e .
$ pip install ma_gym
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