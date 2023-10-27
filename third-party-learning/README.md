# Learning in Gigastep

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

### Torch PPO
#### Installation
```
$ pip install pyqt5==5.12.3
$ pip install ruamel-yaml==0.17.21 gym
$ pip install  matplotlib
$ pip install pybullet
$ pip install stable-baselines3
$ pip install h5py
$ pip install torch==1.13.1
$ pip install wandb
```

## PureJaxRL
### Run
Train
```
$ python ippo.py --env-name identical_5_vs_5 \
                 --all-total-timesteps 20000000 \
                 --eval-every 1000000 \
                 --base-dirname "gigastep_out" \
                 --self-play
```
Eval
```
$ python cross_eval.py --env-name identical_5_vs_5 \
                       --ckpt1 <path-to-ckpt-1> \
                       --ckpt2 <path-to-ckpt-2> \
                       --ckpt-mode "11" \
                       --n-episodes 1000 \
                       --min-ep-len 30
```
