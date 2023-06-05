import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['JAX_DISABLE_JIT'] = '1'

import jax
import jax.numpy as jnp
from evosax import Strategies, ParameterReshaper
from learning.evosax.networks import NetworkMapperGiga
from learning.evosax.problems.gigastep import GigastepFitness
from evosax.utils import ESLog
import tqdm

import time
from datetime import datetime
import numpy as np

from collections import deque
import random

import yaml
import copy


def network_setup(train_evaluator, test_evaluator, rng, env_cfg, net_cfg, net_type="CNN"):
    
    rng_ego, rng_ado = jax.random.split(rng, 2)
    
    output_activation = "tanh_gaussian"  # "tanh_gaussian"
    output_dimensions = 3
    if "discrete_actions" in env_cfg:
        if env_cfg["discrete_actions"]:
            output_activation = "categorical"
            output_dimensions = 9

    if net_type=="MLP":
        ### Ego
        network_ego = NetworkMapperGiga["MLP"](
            num_hidden_units=256,
            num_hidden_layers=2,
            num_output_units=output_dimensions,
            hidden_activation="relu",
            output_activation=output_activation,
        )
        pholder_ego = jnp.zeros((1, *train_evaluator.env.observation_space.low.shape))
        net_ego_params = network_ego.init(
            rng_ego,
            x=pholder_ego,
            rng=rng_ego,
        )

        ### Ado
        network_ado = NetworkMapperGiga["MLP"](
            num_hidden_units=256,
            num_hidden_layers=2,
            num_output_units=output_dimensions,
            hidden_activation="relu",
            output_activation=output_activation,
        )
        pholder_ado = jnp.zeros((1, *train_evaluator.env.observation_space.low.shape))
        net_ado_params = network_ado.init(
            rng_ado,
            x=pholder_ado,
            rng=rng_ado,
        )

        train_evaluator.set_apply_fn(network_ego.apply, network_ado.apply)
        test_evaluator.set_apply_fn(network_ego.apply, network_ado.apply)

    elif net_type=="CNN":
        ### Ego
        network_ego = NetworkMapperGiga["CNN"](
            depth_1=1,
            depth_2=1,
            features_1=32,
            features_2=64,
            kernel_1=8,
            kernel_2=4,
            strides_1=2,
            strides_2=1,
            num_linear_layers=1,
            num_hidden_units=64,
            num_output_units=output_dimensions,
            output_activation=output_activation,
        )
        # Channel last configuration for conv!
        pholder_ego = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        net_ego_params = network_ego.init(
            rng_ego,
            x=pholder_ego,
            rng=rng_ego,
        )

        ### Ado
        network_ado = NetworkMapperGiga["CNN"](
            depth_1=1,
            depth_2=1,
            features_1=32,
            features_2=64,
            kernel_1=8,
            kernel_2=4,
            strides_1=2,
            strides_2=1,
            num_linear_layers=1,
            num_hidden_units=64,
            num_output_units=output_dimensions,
            output_activation=output_activation,
        )
        # Channel last configuration for conv!
        pholder_ado = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        net_ado_params = network_ado.init(
            rng_ado,
            x=pholder_ado,
            rng=rng_ado,
        )

        train_evaluator.set_apply_fn(network_ego.apply, network_ado.apply)
        test_evaluator.set_apply_fn(network_ego.apply, network_ado.apply)

    elif net_type=='LSTM':
        ### Ego
        network_ego = NetworkMapperGiga["LSTM"](
            num_hidden_units=net_cfg["lstm_units"],  # 32,
            num_output_units=output_dimensions,
            output_activation=output_activation,
        )
        pholder_ego = jnp.zeros((1, *train_evaluator.env.observation_space.low.shape))
        carry_ego_init = network_ego.initialize_carry()
        net_ego_params = network_ego.init(
            rng_ego,
            x=pholder_ego,
            carry=carry_ego_init,
            rng=rng_ego,
        )

        ### Ado
        network_ado = NetworkMapperGiga["LSTM"](
            num_hidden_units=32,  # 32,
            num_output_units=output_dimensions,
            output_activation=output_activation,
        )
        pholder_ado = jnp.zeros((1, *train_evaluator.env.observation_space.low.shape))
        carry_ado_init = network_ado.initialize_carry()
        net_ado_params = network_ado.init(
            rng_ado,
            x=pholder_ado,
            carry=carry_ado_init,
            rng=rng_ado,
        )

        train_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network_ego.initialize_carry, network_ado.initialize_carry)
        test_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network_ego.initialize_carry, network_ado.initialize_carry)

    elif net_type=="LSTM_CNN":
        ### Ego
        network_ego = NetworkMapperGiga["LSTM_CNN"](
            # CNN
            num_output_units_cnn=128,  # 64,
            depth_1=1,
            depth_2=1,
            features_1=32,
            features_2=64,
            kernel_1=8,
            kernel_2=4,
            strides_1=2,
            strides_2=1,
            num_linear_layers_cnn=1,
            num_hidden_units_cnn=64,
            output_activation_cnn="identity",
            # LSTM
            num_hidden_units_lstm=64,
            num_output_units_lstm=output_dimensions,
            output_activation_lstm=output_activation,
        )
        # Channel last configuration for conv!
        pholder_ego = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        carry_ego_init = network_ego.initialize_carry(batch_dims=(1,))
        net_ego_params = network_ego.init(
            rng_ego,
            x=pholder_ego,
            carry=carry_ego_init,
            rng=rng_ego,
        )

        ### Ado
        network_ado = NetworkMapperGiga["LSTM_CNN"](
            # CNN
            num_output_units_cnn=128,  # 64,
            depth_1=1,
            depth_2=1,
            features_1=32,
            features_2=64,
            kernel_1=8,
            kernel_2=4,
            strides_1=2,
            strides_2=1,
            num_linear_layers_cnn=1,
            num_hidden_units_cnn=64,
            output_activation_cnn="identity",
            # LSTM
            num_hidden_units_lstm=64,
            num_output_units_lstm=output_dimensions,
            output_activation_lstm=output_activation,
        )
        # Channel last configuration for conv!
        pholder_ado = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        carry_ado_init = network_ado.initialize_carry(batch_dims=(1,))
        net_ado_params = network_ado.init(
            rng_ado,
            x=pholder_ado,
            carry=carry_ado_init,
            rng=rng_ado,
        )

        train_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network_ego.initialize_carry, network_ado.initialize_carry)
        test_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network_ego.initialize_carry, network_ado.initialize_carry)
      
    else:
        raise NotImplementedError

    return net_ego_params, net_ado_params, train_evaluator, test_evaluator


def run_gigastep_fitness(
  scenario_names: str = "identical_5_vs_5", 
  # strategy_name: str = ["OpenES"],
  alg_cfgs: dict = {},
  network_types: str = "LSTM_CNN",
  net_cfgs: dict = {},
  env_cfgs: dict = {}
  ):
    # for s_name in strategy_name:
    for scenario_name in scenario_names:
        for alg_cfg in alg_cfgs:
            for env_cfg in env_cfgs:
                for network_type in network_types:
                    for net_cfg in net_cfgs:

                        alg_cfg = copy.deepcopy(alg_cfg)
                        env_cfg = copy.deepcopy(env_cfg)
                
                        s_name = alg_cfg["strategy_name"]
                        c_name = alg_cfg["config_name"]
                        e_name = env_cfg["config_name"]
                        n_name = net_cfg["config_name"]

                        num_generations = 2000
                        evaluate_every_gen = 100
                        n_devices = 1  # 1 # 2

                        checkpoint_freq = 200

                        update_team1 = 50
                        update_team2 = 50

                        iter_switch = [0]
                        while iter_switch[-1] < num_generations:
                            iter_switch.append(iter_switch[-1] + update_team1)
                            iter_switch.append(iter_switch[-1] + update_team2)
                        iter_switch.pop(0)
                        train_team1 = True
                        train_team2 = False

                        update_ado_freq = 10  # 100
                        extend_ado_freq = 10  # 100
                        max_ado_params = 10
                        params_ado_buffer = deque(maxlen=max_ado_params)

                        log_tensorboard = True

                        debug_reward = env_cfg["debug_reward"]

                        subdir_name = "alg"+c_name.lower() + "_" + "env"+e_name.lower() + "_" + network_type.lower() + n_name.lower()

                        # if log_tensorboard:
                        from torch.utils.tensorboard import SummaryWriter
                        logdir = './logdir/evosax/eval/' + s_name.lower() + '/' + scenario_name.lower() + '/' +  subdir_name + '/' + datetime.now().strftime("%y_%m_%d_%H_%M_%S")
                        writer = SummaryWriter(log_dir=logdir, flush_secs=10)

                        with open(logdir + '/cfg.yml', 'w') as outfile:
                            yaml.dump(alg_cfg, outfile, default_flow_style=False)


                        rng = jax.random.PRNGKey(0)

                        del env_cfg["config_name"]
                        train_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=False, n_devices=n_devices, env_cfg=env_cfg)
                        test_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=True, n_devices=1, env_cfg=env_cfg)

                        # Initialize network depending on type
                        out = network_setup(train_evaluator, test_evaluator, rng, env_cfg, net_cfg, net_type=network_type)
                        (net_ego_params, net_ado_params, train_evaluator, test_evaluator) = out

                        train_param_ego_reshaper = ParameterReshaper(net_ego_params, n_devices=n_devices)
                        test_param_ego_reshaper = ParameterReshaper(net_ego_params, n_devices=1)
                        train_param_ado_reshaper = ParameterReshaper(net_ado_params, n_devices=n_devices)
                        test_param_ado_reshaper = ParameterReshaper(net_ado_params, n_devices=1)

                        if "strategy" in alg_cfg.keys():
                            strategy_params = alg_cfg["strategy"]
                        else:
                            strategy_params = {}
                        strategy_params["centered_rank"] = True
                        # strategy_params["popsize"] = 512  # 512  # 32
                        # # strategy_params["mean_decay"] = 1.e-2
                        # # if s_name=="DES":
                        # #     strategy_params["sigma_init"] = 0.1  # 5.0
                        # if s_name=="PGPE":
                        #     strategy_params["elite_ratio"] = 1.0
                        # if s_name=='OpenES':
                        #     strategy_params["sigma_init"] = 0.1  # 03
                        # else:
                        #     pass

                        strategy_ego = Strategies[s_name](**strategy_params, num_dims=train_param_ego_reshaper.total_params, maximize=True)
                        es_ego_params = strategy_ego.default_params
                        if "params" in alg_cfg.keys():
                            es_ego_params = es_ego_params.replace(**alg_cfg["params"])
                        es_ego_state = strategy_ego.initialize(rng, es_ego_params)
                        es_ego_state_mean = es_ego_state.mean.copy()

                        strategy_ado = Strategies[s_name](**strategy_params, num_dims=train_param_ado_reshaper.total_params, maximize=True)
                        es_ado_params = strategy_ado.default_params
                        if "params" in alg_cfg.keys():
                            es_ado_params = es_ado_params.replace(**alg_cfg["params"])
                        es_ado_state = strategy_ado.initialize(rng, es_ado_params)
                        es_ado_state_mean = es_ado_state.mean.copy()

                        es_ego_logging = ESLog(
                            num_dims=train_param_ego_reshaper.total_params,
                            num_generations=num_generations,
                            top_k=5,
                            maximize=True,
                        )
                        es_ego_log = es_ego_logging.initialize()

                        es_ado_logging = ESLog(
                            num_dims=train_param_ado_reshaper.total_params,
                            num_generations=num_generations,
                            top_k=5,
                            maximize=True,
                        )
                        es_ado_log = es_ado_logging.initialize()

                        # # Initialize opponent buffer
                        # x_mean = jnp.repeat(es_state_mean[None], x.shape[0], 0)
                        # x_mean_re = train_param_reshaper.reshape(x_mean)
                        # params_ado_buffer.append(x_mean_re)

                        # Run the ask-eval-tell loop
                        log_steps, log_return = [], []
                        t = tqdm.tqdm(range(1, num_generations + 1), desc=s_name, leave=True)
                        for gen in t:
                            rng, rng_gen_ego, rng_gen_ado, rng_eval = jax.random.split(rng, 4)

                            if gen in iter_switch:
                                train_team1 = not train_team1
                                train_team2 = not train_team2

                            # Sporadically evaluate the mean & best performance on test evaluator.
                            if gen % evaluate_every_gen == 0:
                                rng, rng_test = jax.random.split(rng)
                                # Stack best params seen & mean strategy params for eval
                                best_params_ego = es_ego_log["top_params"][0]
                                mean_params_ego = es_ego_state.mean

                                best_params_ado = es_ado_log["top_params"][0]
                                mean_params_ado = es_ado_state.mean

                                x_ego_test = jnp.stack([best_params_ego, mean_params_ego], axis=0)  # FIXME: this is sketchy
                                x_ego_test_re = test_param_ego_reshaper.reshape(x_ego_test)

                                x_ado_mean_test = jnp.repeat(es_ado_state_mean[None], x_ego_test.shape[0], 0)  # TODO: check if repeat can be removed
                                x_ado_mean_test_re = test_param_ado_reshaper.reshape(x_ado_mean_test)

                                test_returns = test_evaluator.rollout(
                                    rng_test, x_ego_test_re, x_ado_mean_test_re
                                )
                                if debug_reward:
                                    test_scores_ego, test_scores_ado, test_images_global, _, _ = test_returns
                                else:
                                    test_scores_ego, test_scores_ado, test_images_global, _ = test_returns

                                if log_tensorboard:
                                    ### Video
                                    id_prm = 0
                                    id_run = 0
                                    id_ego = 0
                                    frames_glb = np.array(test_images_global[id_prm, id_run])

                                    frames_glb = np.transpose(frames_glb, (0, 3, 1, 2))  # (T, C, W, H)
                                    frames_glb = np.expand_dims(frames_glb, axis=0)      # (N, T, C, W, H)
                                    writer.add_video("test/video-all", frames_glb, global_step=gen, fps=15)
                                
                                test_return_to_log = test_scores_ego[0]
                                log_steps.append(train_evaluator.total_env_steps)
                                log_return.append(test_return_to_log)
                                t.set_description(f"R: " + "{:.3f}".format(test_return_to_log.item()))
                                t.refresh()
                            
                            # Sample parameters for the ego team
                            x_ego, es_ego_state = strategy_ego.ask(rng_gen_ego, es_ego_state)
                            x_ego_re = train_param_ego_reshaper.reshape(x_ego)

                            if (gen-1) % update_ado_freq == 0:
                                x_ado, es_ado_state = strategy_ado.ask(rng_gen_ado, es_ado_state)
                                x_ado_re = train_param_ado_reshaper.reshape(x_ado)

                            # Get mean parameters for the ado team
                            if (gen-1) % extend_ado_freq == 0:
                                es_ado_state_mean = es_ado_state.mean.copy()
                                x_ado_mean = jnp.repeat(es_ado_state_mean[None], x_ego.shape[0], 0)  # TODO: check if repeat can be removed
                                params_ado = x_ado_mean  # train_param_ado_reshaper.reshape(x_ado_mean)
                                params_ado_buffer.append(params_ado)
                            if (gen-1) % update_ado_freq == 0:
                                x_ado_mean = random.sample(list(params_ado_buffer), 1)[0]
                                x_ado_mean_re = train_param_ado_reshaper.reshape(x_ado_mean)

                            # Rollout fitness and update parameter distribution
                            # scores, reward_info = train_evaluator.rollout(rng_eval, x_re, x_mean_re)
                            train_returns = train_evaluator.rollout(rng_eval, x_ego_re, x_ado_re)  # x_ado_mean_re)
                            if debug_reward:
                                scores_ego, scores_ado, reward_info, act_info = train_returns
                            else:
                                scores_ego, scores_ado, act_info = train_returns
                                reward_info = None
                            es_ego_state = strategy_ego.tell(x_ego, scores_ego, es_ego_state)
                            es_ado_state = strategy_ado.tell(x_ado, scores_ado, es_ado_state)

                            # Update the logging instance.
                            es_ego_log = es_ego_logging.update(es_ego_log, x_ego, scores_ego)
                            es_ado_log = es_ado_logging.update(es_ado_log, x_ado, scores_ado)

                            if gen % checkpoint_freq == 0:
                                es_ego_logging.save(log=es_ego_log, filename=logdir + '/log_iter_' + str(gen))

                            if log_tensorboard:
                                for key, value in es_ego_log.items():
                                    if not "params" in key:
                                        val = value[gen-1] if len(value.shape) else value
                                        writer.add_scalar('train/' + key, np.array(val), gen)
                                if reward_info:
                                    for key, value in reward_info.items():
                                        val = value[gen-1] if len(value.shape) else value
                                        writer.add_scalar('train/' + key, np.array(val), gen)
                                for key, value in act_info.items():
                                        for idx, val in enumerate(value):
                                            val = val[gen-1] if len(val.shape) else val
                                            writer.add_scalar('train/' + key + '_dim' + str(idx), np.array(val), gen)

                        # Sporadically evaluate the mean & best performance on test evaluator.
                        if True:
                            rng, rng_test = jax.random.split(rng)
                            # Stack best params seen & mean strategy params for eval
                            best_params_ego = es_ego_log["top_params"][0]
                            mean_params_ego = es_ego_state.mean

                            best_params_ado = es_ado_log["top_params"][0]
                            mean_params_ado = es_ado_state.mean

                            x_ego_test = jnp.stack([best_params_ego, mean_params_ego], axis=0)  # FIXME: this is sketchy
                            x_ego_test_re = test_param_ego_reshaper.reshape(x_ego_test)

                            x_ado_mean_test = jnp.repeat(mean_params_ado[None], x_ego_test.shape[0], 0)  # TODO: check if repeat can be removed
                            x_ado_mean_test_re = test_param_ado_reshaper.reshape(x_ado_mean_test)

                            test_returns = test_evaluator.rollout(
                                rng_test, x_ego_test_re, x_ado_mean_test_re
                            )
                            if debug_reward:
                                test_scores_ego, test_scores_ado, test_images_global, _, _ = test_returns
                            else:
                                test_scores_ego, test_scores_ado, test_images_global, _ = test_returns

                            if log_tensorboard:
                                ### Video
                                id_prm = 0
                                id_run = 0
                                id_ego = 0
                                frames_glb = np.array(test_images_global[id_prm, id_run])

                                frames_glb = np.transpose(frames_glb, (0, 3, 1, 2))  # (T, C, W, H)
                                frames_glb = np.expand_dims(frames_glb, axis=0)      # (N, T, C, W, H)
                                writer.add_video("test/video-all", frames_glb, global_step=gen, fps=15)
                        
                        test_return_to_log = test_scores_ego[0]
                        log_steps.append(train_evaluator.total_env_steps)
                        log_return.append(test_return_to_log)
                        t.set_description(f"R: " + "{:.3f}".format(test_return_to_log.item()))
                        t.refresh()
        

if __name__ == "__main__":
    
    ### Environment config
    # Base config
    env_cfg0 = {}
    env_cfg0["config_name"] = "config0"
    env_cfg0["obs_type"] = "vector"  # "rgb"
    env_cfg0["discrete_actions"] = True  # True 
    env_cfg0["reward_game_won"] = 100
    env_cfg0["reward_defeat_one_opponent"] = 100
    env_cfg0["reward_detection"] = 0
    env_cfg0["reward_damage"] = 0  # 0
    env_cfg0["reward_idle"] = 0
    env_cfg0["reward_collision_obstacle"] = 100  # 0
    env_cfg0["cone_depth"] = 15.0
    env_cfg0["cone_angle"] = jnp.pi * 1.99  # 1.99
    env_cfg0["enable_waypoints"] = False
    env_cfg0["use_stochastic_obs"] = False
    env_cfg0["use_stochastic_comm"] = False
    env_cfg0["max_agent_in_vec_obs"] = 100
    env_cfg0["debug_reward"] = False
    env_cfg0["max_episode_length"] = 500
    # Config 1
    env_cfg1 = copy.deepcopy(env_cfg0)
    env_cfg1["config_name"] = "config1"
    env_cfg1["max_episode_length"] = 200
    # Config 2
    env_cfg2 = copy.deepcopy(env_cfg0)
    env_cfg2["config_name"] = "config2"
    env_cfg2["max_episode_length"] = 100
    # Config 3
    env_cfg3 = copy.deepcopy(env_cfg0)
    env_cfg3["config_name"] = "config3"
    env_cfg3["max_episode_length"] = 50

    ### Network config
    ### LSTM
    # Base Config
    net_cfg0 = {}
    net_cfg0["config_name"] = "config0"
    net_cfg0["lstm_units"] = 32
    # Config 1
    net_cfg1 = copy.deepcopy(net_cfg0)
    net_cfg1["config_name"] = "config1"
    net_cfg1["lstm_units"] = 16
    # Config 2
    net_cfg2 = copy.deepcopy(net_cfg0)
    net_cfg2["config_name"] = "config2"
    net_cfg2["lstm_units"] = 64

    # ### Algorithm config
    ### DES
    # Base Config
    alg_des_cfg0 = {}
    alg_des_cfg0["strategy_name"] = "DES"
    alg_des_cfg0["config_name"] = "config0"
    strategy_cfg = {}
    strategy_cfg["popsize"] = 512
    strategy_cfg["temperature"] = 12.5
    strategy_cfg["sigma_init"] = 0.1
    strategy_cfg["mean_decay"] = 0.0
    params_cfg = {}
    params_cfg["lrate_sigma"] = 0.1
    params_cfg["lrate_mean"] = 1.0
    params_cfg["init_min"] = 0.0
    params_cfg["init_max"] = 0.0
    alg_des_cfg0["strategy"] = strategy_cfg
    alg_des_cfg0["params"] = params_cfg
    # Config 1
    alg_des_cfg1 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg1["config_name"] = "config1"
    alg_des_cfg1["strategy"]["temperature"] = 6.25
    # Config 2
    alg_des_cfg2 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg2["config_name"] = "config2"
    alg_des_cfg2["strategy"]["sigma_init"] = 0.5
    # Config 3
    alg_des_cfg3 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg3["config_name"] = "config3"
    alg_des_cfg3["params"]["lrate_mean"] = 0.1
    # Config 4
    alg_des_cfg4 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg4["config_name"] = "config4"
    alg_des_cfg4["params"]["lrate_sigma"] = 1.0
    # # Config 5
    # alg_cfg5 = copy.deepcopy(alg_des_cfg0)
    # alg_cfg5["config_name"] = "config5"
    # alg_cfg5["strategy"]["popsize"] = 1024
    # Config 6
    alg_des_cfg6 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg6["config_name"] = "config6"
    alg_des_cfg6["strategy"]["temperature"] = 6.25
    alg_des_cfg6["params"]["lrate_sigma"] = 1.0
    # Config 7
    alg_des_cfg7 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg7["config_name"] = "config7"
    alg_des_cfg7["strategy"]["temperature"] = 3.0
    # Config 8
    alg_des_cfg8 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg8["config_name"] = "config8"
    alg_des_cfg8["strategy"]["temperature"] = 9.0
    # Config 9
    alg_des_cfg9 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg9["config_name"] = "config9"
    alg_des_cfg9["params"]["lrate_sigma"] = 2.0
    # Config 10
    alg_des_cfg10 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg10["config_name"] = "config10"
    alg_des_cfg10["params"]["lrate_sigma"] = 0.5
    # # Config 11 --> MLP: MLPs not good right now
    # alg_cfg11 = copy.deepcopy(alg_cfg)
    # alg_cfg11["config_name"] = "config11"
    # alg_cfg11["strategy"]["temperature"] = 6.25
    # # Config 12 --> MLP
    # alg_cfg12 = copy.deepcopy(alg_cfg)
    # alg_cfg12["config_name"] = "config12"
    # alg_cfg12["strategy"]["sigma_init"] = 0.5
    # # Config 13 --> MLP
    # alg_cfg13 = copy.deepcopy(alg_cfg)
    # alg_cfg13["config_name"] = "config13"
    # alg_cfg13["params"]["lrate_mean"] = 0.1
    # # Config 14 --> MLP
    # alg_cfg14 = copy.deepcopy(alg_cfg)
    # alg_cfg14["config_name"] = "config14"
    # alg_cfg14["params"]["lrate_sigma"] = 1.0
    # # Config 15 --> 100 horizon
    # alg_cfg15 = copy.deepcopy(alg_cfg)
    # alg_cfg15["config_name"] = "config15"
    # alg_cfg15["params"]["lrate_sigma"] = 1.0
    # # Config 16 --> 100 horizon
    # alg_cfg16 = copy.deepcopy(alg_cfg)
    # alg_cfg16["config_name"] = "config16"
    # # Config 17 --> continuous
    # alg_cfg17 = copy.deepcopy(alg_cfg)
    # alg_cfg17["config_name"] = "config17_continuous"
    # # Config 18 --> continuous
    # alg_cfg18 = copy.deepcopy(alg_cfg)
    # alg_cfg18["config_name"] = "config18_continuous"
    # alg_cfg18["params"]["lrate_sigma"] = 1.0
    # # Config 19
    # alg_cfg19 = copy.deepcopy(alg_cfg)
    # alg_cfg19["config_name"] = "config19"
    # alg_cfg19["strategy"]["popsize"] = 1024
    # Config 20
    alg_des_cfg20 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg20["config_name"] = "config20"
    alg_des_cfg20["strategy"]["temperature"] = 20.0
    # Config 21
    alg_des_cfg21 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg21["config_name"] = "config21"
    alg_des_cfg21["params"]["lrate_sigma"] = 0.05
    # Config 22
    alg_des_cfg22 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg22["config_name"] = "config22"
    alg_des_cfg22["strategy"]["sigma_init"] = 3.0
    # Config 23
    alg_des_cfg23 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg23["config_name"] = "config23"
    alg_des_cfg23["strategy"]["temperature"] = 20.0
    alg_des_cfg23["params"]["lrate_sigma"] = 0.5
    # Config 24
    alg_des_cfg24 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg24["config_name"] = "config24"
    alg_des_cfg24["strategy"]["temperature"] = 20.0
    alg_des_cfg24["strategy"]["sigma_init"] = 3.0
    # Config 25
    alg_des_cfg25 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg25["config_name"] = "config25"
    alg_des_cfg25["strategy"]["temperature"] = 20.0
    alg_des_cfg25["strategy"]["sigma_init"] = 3.0
    alg_des_cfg25["params"]["lrate_sigma"] = 0.05
    # Config 26
    alg_des_cfg26 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg26["config_name"] = "config26"
    alg_des_cfg26["strategy"]["temperature"] = 20.0
    alg_des_cfg26["strategy"]["sigma_init"] = 3.0
    alg_des_cfg26["params"]["lrate_sigma"] = 0.2
    # Config 27
    alg_des_cfg27 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg27["config_name"] = "config27"
    alg_des_cfg27["strategy"]["temperature"] = 20.0
    alg_des_cfg27["params"]["lrate_sigma"] = 0.2
    # Config 28
    alg_des_cfg28 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg28["config_name"] = "config28"
    alg_des_cfg28["strategy"]["temperature"] = 20.0
    alg_des_cfg28["params"]["lrate_sigma"] = 0.5
    # Config 29
    alg_des_cfg29 = copy.deepcopy(alg_des_cfg0)
    alg_des_cfg29["config_name"] = "config29"
    alg_des_cfg29["strategy"]["temperature"] = 20.0
    alg_des_cfg29["strategy"]["sigma_init"] = 3.0
    alg_des_cfg29["params"]["lrate_sigma"] = 0.5

    # ### Open-ES
    # Base Config
    alg_openes_cfg0 = {}
    alg_openes_cfg0["strategy_name"] = "OpenES"
    alg_openes_cfg0["config_name"] = "config0"
    strategy_cfg = {}
    strategy_cfg["popsize"] = 512
    strategy_cfg["lrate_init"] = 0.05
    strategy_cfg["sigma_init"] = 0.03
    strategy_cfg["mean_decay"] = 0.0
    params_cfg = {}
    params_cfg["init_min"] = 0.0
    params_cfg["init_max"] = 0.0
    alg_openes_cfg0["strategy"] = strategy_cfg
    alg_openes_cfg0["params"] = params_cfg
    # Config 1
    alg_openes_cfg1 = copy.deepcopy(alg_openes_cfg0)
    alg_openes_cfg1["config_name"] = "config1"
    alg_openes_cfg1["strategy"]["lrate_init"] = 0.2
    # Config 2
    alg_openes_cfg2 = copy.deepcopy(alg_openes_cfg0)
    alg_openes_cfg2["config_name"] = "config2"
    alg_openes_cfg2["strategy"]["lrate_init"] = 0.02
    # Config 3
    alg_openes_cfg3 = copy.deepcopy(alg_openes_cfg0)
    alg_openes_cfg3["config_name"] = "config3"
    alg_openes_cfg3["strategy"]["sigma_init"] = 0.3
    # Config 4
    alg_openes_cfg4 = copy.deepcopy(alg_openes_cfg0)
    alg_openes_cfg4["config_name"] = "config4"
    alg_openes_cfg4["strategy"]["sigma_init"] = 1.0
    # # Config 5
    # alg_cfg5 = copy.deepcopy(alg_cfg)
    # alg_cfg5["config_name"] = "config5"
    # alg_cfg5["strategy"]["popsize"] = 1024

    ### CMA-ES
    # Base Config
    alg_cmaes_cfg0 = {}
    alg_cmaes_cfg0["strategy_name"] = "CMA_ES"
    alg_cmaes_cfg0["config_name"] = "config0"
    strategy_cfg = {}
    strategy_cfg["popsize"] = 100  # 512
    params_cfg = {}
    alg_cmaes_cfg0["strategy"] = strategy_cfg
    alg_cmaes_cfg0["params"] = params_cfg

    ### CR-FM-NES
    # Base Config
    alg_crfmnes_cfg0 = {}
    alg_crfmnes_cfg0["strategy_name"] = "CR_FM_NES"
    alg_crfmnes_cfg0["config_name"] = "config0"
    strategy_cfg = {}
    strategy_cfg["popsize"] = 512
    params_cfg = {}
    alg_crfmnes_cfg0["strategy"] = strategy_cfg
    alg_crfmnes_cfg0["params"] = params_cfg

    ### PGPE
    # Base Config
    alg_pgpe_cfg0 = {}
    alg_pgpe_cfg0["strategy_name"] = "PGPE"
    alg_pgpe_cfg0["config_name"] = "config0"
    strategy_cfg = {}
    strategy_cfg["popsize"] = 512
    params_cfg = {}
    alg_pgpe_cfg0["strategy"] = strategy_cfg
    alg_pgpe_cfg0["params"] = params_cfg


    scenario_names = [
        # "identical_20_vs_20",
        # "special_20_vs_20",
        # "identical_10_vs_10",
        # "special_10_vs_10",
        "identical_5_vs_5",
        # "special_5_vs_5",
        # "identical_2_vs_2",
        # "identical_5_vs_1",
        # "special_5_vs_1",
        # "identical_10_vs_3",
        # "special_10_vs_3",
        # "identical_20_vs_5",
        # "special_20_vs_5",
        # "identical_20_vs_20_center_block",
        # "identical_20_vs_20_two_rooms1",
        # "identical_10_vs_10_center_block",
        # "identical_10_vs_10_two_rooms1",
        # "identical_5_vs_5_center_block",
        # "identical_2_vs_2_center_block",
        # "identical_5_vs_5_two_rooms1",
    ]


    t_start = time.time()
    run_gigastep_fitness(
        scenario_names=scenario_names, 
        # strategy_name=["DES"],  # ["OpenES", "DES", "CR_FM_NES"],
        alg_cfgs=[
            # alg_des_cfg1,
            # alg_des_cfg2,
            # alg_des_cfg3,
            # alg_des_cfg4,
            # alg_des_cfg6,
            # alg_des_cfg7,
            # alg_des_cfg8,
            # alg_des_cfg9,
            # alg_des_cfg10,
            # alg_des_cfg20,
            # alg_des_cfg21,
            # alg_des_cfg22,
            alg_des_cfg23,
            alg_des_cfg24,
            alg_des_cfg25,
            alg_des_cfg26,
            alg_des_cfg27,
            alg_des_cfg28,
            alg_des_cfg29,
            # alg_openes_cfg0,
            # alg_openes_cfg1,
            # alg_openes_cfg2,
            # alg_openes_cfg3,
            # alg_openes_cfg4,
            # alg_crfmnes_cfg0,
            # alg_pgpe_cfg0,
            # alg_cmaes_cfg0,
        ],
        network_types=["LSTM"],  # "LSTM",
        net_cfgs = [
            net_cfg0,
            # net_cfg1,
            # net_cfg2,
        ],
        env_cfgs=[
            # env_cfg3,
            env_cfg2,
            # env_cfg1,
            # env_cfg0,
        ])
    t_end = time.time()
    print("Runtime = " + str(round(t_end-t_start, 3)))
