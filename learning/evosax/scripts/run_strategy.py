import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def network_setup(train_evaluator, test_evaluator, rng, env_cfg, net_type="CNN"):
    
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
            num_hidden_units=32,  # 32,
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

        train_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network.initialize_carry)
        test_evaluator.set_apply_fn(network_ego.apply, network_ado.apply, network.initialize_carry)

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
  scenario_name: str = "identical_5_vs_5", 
  # strategy_name: str = ["OpenES"],
  alg_cfgs: dict = {},
  network_type: str = "LSTM_CNN",
  env_cfg: dict = {}
  ):
    # for s_name in strategy_name:
    for alg_cfg in alg_cfgs:
        
        s_name = alg_cfg["strategy_name"]
        c_name = alg_cfg["config_name"]
        num_generations = 1000
        evaluate_every_gen = 100
        n_devices = 1  # 1 # 2

        update_ado_freq = 100
        extend_ado_freq = 100
        max_ado_params = 10
        params_ado_buffer = deque(maxlen=max_ado_params)

        log_tensorboard = True

        debug_reward = env_cfg["debug_reward"]

        # if log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logdir = './logdir/evosax/' + s_name.lower() + '/' + scenario_name.lower() + '/' +  c_name.lower() + '/' + datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        writer = SummaryWriter(log_dir=logdir, flush_secs=10)

        with open(logdir + '/cfg.yml', 'w') as outfile:
            yaml.dump(alg_cfg, outfile, default_flow_style=False)


        rng = jax.random.PRNGKey(0)

        train_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=False, n_devices=n_devices, env_cfg=env_cfg)
        test_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=True, n_devices=1, env_cfg=env_cfg)

        # Initialize network depending on type
        out = network_setup(train_evaluator, test_evaluator, rng, env_cfg, net_type=network_type)
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
            es_params = es_params.replace(**alg_cfg["params"])
        # if s_name=="DES":
        #     es_params = es_params.replace(lrate_mean=0.01, lrate_sigma=0.01, init_max=1.0)
        es_ego_state = strategy_ego.initialize(rng, es_params)
        es_ego_state_mean = es_ego_state.mean.copy()

        es_logging = ESLog(
            num_dims=train_param_ego_reshaper.total_params,
            num_generations=num_generations,
            top_k=5,
            maximize=True,
        )
        es_log = es_logging.initialize()

        # # Initialize opponent buffer
        # x_mean = jnp.repeat(es_state_mean[None], x.shape[0], 0)
        # x_mean_re = train_param_reshaper.reshape(x_mean)
        # params_ado_buffer.append(x_mean_re)

        # Run the ask-eval-tell loop
        log_steps, log_return = [], []
        t = tqdm.tqdm(range(1, num_generations + 1), desc=s_name, leave=True)
        for gen in t:
            rng, rng_gen, rng_eval = jax.random.split(rng, 3)

            # Sporadically evaluate the mean & best performance on test evaluator.
            if gen % evaluate_every_gen == 0:
                rng, rng_test = jax.random.split(rng)
                # Stack best params seen & mean strategy params for eval
                best_params = es_log["top_params"][0]
                mean_params = es_state.mean

                x_test = jnp.stack([best_params, mean_params], axis=0)
                x_test_re = test_param_reshaper.reshape(x_test)

                x_mean_test = jnp.repeat(es_state_mean[None], x_test.shape[0], 0)  # TODO: check if repeat can be removed
                x_mean_test_re = test_param_reshaper.reshape(x_mean_test)

                test_returns = test_evaluator.rollout(
                    rng_test, x_test_re, x_mean_test_re
                )
                if debug_reward:
                    test_scores, test_images_global, _, _ = test_returns
                else:
                    test_scores, test_images_global, _ = test_returns

                if log_tensorboard:
                    ### Video
                    id_prm = 0
                    id_run = 0
                    id_ego = 0
                    frames_glb = np.array(test_images_global[id_prm, id_run])

                    frames_glb = np.transpose(frames_glb, (0, 3, 1, 2))  # (T, C, W, H)
                    frames_glb = np.expand_dims(frames_glb, axis=0)      # (N, T, C, W, H)
                    writer.add_video("test/video-all", frames_glb, global_step=gen, fps=15)
                
                test_return_to_log = test_scores[1]
                log_steps.append(train_evaluator.total_env_steps)
                log_return.append(test_return_to_log)
                t.set_description(f"R: " + "{:.3f}".format(test_return_to_log.item()))
                t.refresh()
            
            # Sample parameters for the ego team
            x, es_state = strategy.ask(rng_gen, es_state)
            x_re = train_param_reshaper.reshape(x)

            # Get mean parameters for the ado team
            if (gen-1) % extend_ado_freq == 0:
                es_state_mean = es_state.mean.copy()
                x_mean = jnp.repeat(es_state_mean[None], x.shape[0], 0)  # TODO: check if repeat can be removed
                params_ado = train_param_reshaper.reshape(x_mean)
                params_ado_buffer.append(params_ado)
            if (gen-1) % update_ado_freq == 0:
                x_mean_re = random.sample(list(params_ado_buffer), 1)[0]

            # Rollout fitness and update parameter distribution
            # scores, reward_info = train_evaluator.rollout(rng_eval, x_re, x_mean_re)
            train_returns = train_evaluator.rollout(rng_eval, x_re, x_mean_re)
            if debug_reward:
                scores, reward_info, act_info = train_returns
            else:
                scores, act_info = train_returns
                reward_info = None
            es_state = strategy.tell(x, scores, es_state)

            # Update the logging instance.
            es_log = es_logging.update(es_log, x, scores)

            if log_tensorboard:
                for key, value in es_log.items():
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
            best_params = es_log["top_params"][0]
            mean_params = es_state.mean

            x_test = jnp.stack([best_params, mean_params], axis=0)
            x_test_re = test_param_reshaper.reshape(x_test)

            x_mean_test = jnp.repeat(es_state_mean[None], x_test.shape[0], 0)  # TODO: check if repeat can be removed
            x_mean_test_re = test_param_reshaper.reshape(x_mean_test)

            test_returns = test_evaluator.rollout(
                rng_test, x_test_re, x_mean_test_re
            )
            if debug_reward:
                test_scores, test_images_global, _, _ = test_returns
            else:
                test_scores, test_images_global, _ = test_returns

            if log_tensorboard:
                ### Video
                id_prm = 0
                id_run = 0
                id_ego = 0
                frames_glb = np.array(test_images_global[id_prm, id_run])

                frames_glb = np.transpose(frames_glb, (0, 3, 1, 2))  # (T, C, W, H)
                frames_glb = np.expand_dims(frames_glb, axis=0)      # (N, T, C, W, H)
                writer.add_video("test/video-all", frames_glb, global_step=gen, fps=15)
            
            test_return_to_log = test_scores[1]
            log_steps.append(train_evaluator.total_env_steps)
            log_return.append(test_return_to_log)
            t.set_description(f"R: " + "{:.3f}".format(test_return_to_log.item()))
            t.refresh()
        

if __name__ == "__main__":
    
    ### Environment config
    env_cfg = {}
    env_cfg["obs_type"] = "vector"  # "rgb"
    env_cfg["discrete_actions"] = True  # True 
    env_cfg["reward_game_won"] = 100
    env_cfg["reward_defeat_one_opponent"] = 100
    env_cfg["reward_detection"] = 0
    env_cfg["reward_damage"] = 0  # 0
    env_cfg["reward_idle"] = 0
    env_cfg["reward_collision"] = 0  # 0
    env_cfg["cone_depth"] = 15.0
    env_cfg["cone_angle"] = jnp.pi * 1.99  # 1.99
    env_cfg["enable_waypoints"] = False
    env_cfg["use_stochastic_obs"] = False
    env_cfg["use_stochastic_comm"] = False
    env_cfg["max_agent_in_vec_obs"] = 100
    env_cfg["debug_reward"] = True

    ### Algorithm config
    ### DES
    # Base Config
    alg_cfg = {}
    alg_cfg["strategy_name"] = "DES"
    alg_cfg["config_name"] = "config0_discrete"
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
    alg_cfg["strategy"] = strategy_cfg
    alg_cfg["params"] = params_cfg
    # # Config 1
    # alg_cfg1 = alg_cfg.copy()
    # alg_cfg1["config_name"] = "config1"
    # alg_cfg1["strategy"]["temperature"] = 6.25
    # # Config 2
    # alg_cfg2 = alg_cfg.copy()
    # alg_cfg2["config_name"] = "config2"
    # alg_cfg2["strategy"]["sigma_init"] = 0.5
    # # Config 3
    # alg_cfg3 = alg_cfg.copy()
    # alg_cfg3["config_name"] = "config3"
    # alg_cfg3["params"]["lrate_mean"] = 0.1
    # Config 4
    alg_cfg4 = alg_cfg.copy()
    alg_cfg4["config_name"] = "config4_discrete"
    alg_cfg4["params"]["lrate_sigma"] = 1.0
    # # Config 5
    # alg_cfg5 = alg_cfg.copy()
    # alg_cfg5["config_name"] = "config5"
    # alg_cfg5["strategy"]["popsize"] = 1024
    # # Config 6
    # alg_cfg6 = alg_cfg.copy()
    # alg_cfg6["config_name"] = "config6"
    # alg_cfg6["strategy"]["temperature"] = 6.25
    # alg_cfg6["params"]["lrate_sigma"] = 1.0
    # # Config 7
    # alg_cfg7 = alg_cfg.copy()
    # alg_cfg7["config_name"] = "config7"
    # alg_cfg7["strategy"]["temperature"] = 3.0
    # # Config 8
    # alg_cfg8 = alg_cfg.copy()
    # alg_cfg8["config_name"] = "config8"
    # alg_cfg8["strategy"]["temperature"] = 9.0
    # # Config 9
    # alg_cfg9 = alg_cfg.copy()
    # alg_cfg9["config_name"] = "config9"
    # alg_cfg9["params"]["lrate_sigma"] = 2.0
    # # Config 10
    # alg_cfg10 = alg_cfg.copy()
    # alg_cfg10["config_name"] = "config10"
    # alg_cfg10["params"]["lrate_sigma"] = 0.5
    # # Config 11 --> MLP: MLPs not good right now
    # alg_cfg11 = alg_cfg.copy()
    # alg_cfg11["config_name"] = "config11"
    # alg_cfg11["strategy"]["temperature"] = 6.25
    # # Config 12 --> MLP
    # alg_cfg12 = alg_cfg.copy()
    # alg_cfg12["config_name"] = "config12"
    # alg_cfg12["strategy"]["sigma_init"] = 0.5
    # # Config 13 --> MLP
    # alg_cfg13 = alg_cfg.copy()
    # alg_cfg13["config_name"] = "config13"
    # alg_cfg13["params"]["lrate_mean"] = 0.1
    # # Config 14 --> MLP
    # alg_cfg14 = alg_cfg.copy()
    # alg_cfg14["config_name"] = "config14"
    # alg_cfg14["params"]["lrate_sigma"] = 1.0
    # # Config 15 --> 100 horizon
    # alg_cfg15 = alg_cfg.copy()
    # alg_cfg15["config_name"] = "config15"
    # alg_cfg15["params"]["lrate_sigma"] = 1.0
    # # Config 16 --> 100 horizon
    # alg_cfg16 = alg_cfg.copy()
    # alg_cfg16["config_name"] = "config16"

    # ### Open-ES
    # # Base Config
    # alg_cfg = {}
    # alg_cfg["strategy_name"] = "OpenES"
    # alg_cfg["config_name"] = "config0"
    # strategy_cfg = {}
    # strategy_cfg["popsize"] = 512
    # strategy_cfg["lrate_init"] = 0.05
    # strategy_cfg["sigma_init"] = 0.03
    # strategy_cfg["mean_decay"] = 0.0
    # params_cfg = {}
    # params_cfg["init_min"] = 0.0
    # params_cfg["init_max"] = 0.0
    # alg_cfg["strategy"] = strategy_cfg
    # alg_cfg["params"] = params_cfg
    # # Config 1
    # alg_cfg1 = alg_cfg.copy()
    # alg_cfg1["config_name"] = "config1"
    # alg_cfg1["strategy"]["lrate_init"] = 0.2
    # # Config 2
    # alg_cfg2 = alg_cfg.copy()
    # alg_cfg2["config_name"] = "config2"
    # alg_cfg2["strategy"]["lrate_init"] = 0.02
    # # Config 3
    # alg_cfg3 = alg_cfg.copy()
    # alg_cfg3["config_name"] = "config3"
    # alg_cfg3["strategy"]["sigma_init"] = 0.3
    # # Config 4
    # alg_cfg4 = alg_cfg.copy()
    # alg_cfg4["config_name"] = "config4"
    # alg_cfg4["strategy"]["sigma_init"] = 1.0
    # # Config 5
    # alg_cfg5 = alg_cfg.copy()
    # alg_cfg5["config_name"] = "config5"
    # alg_cfg5["strategy"]["popsize"] = 1024


    t_start = time.time()
    run_gigastep_fitness(
        scenario_name="identical_5_vs_5", 
        # strategy_name=["DES"],  # ["OpenES", "DES", "CR_FM_NES"],
        alg_cfgs=[
            alg_cfg,
            alg_cfg4,
        ],
        network_type="LSTM",  # "LSTM",
        env_cfg=env_cfg)
    t_end = time.time()
    print("Runtime = " + str(round(t_end-t_start, 3)))
