import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
import jax.numpy as jnp
from evosax import CR_FM_NES, ParameterReshaper
from learning.evosax.networks import NetworkMapperGiga
from learning.evosax.problems.gigastep import GigastepFitness
from evosax.utils import ESLog
from evosax.utils import FitnessShaper  # NOTE: will move to evosax.core
import tqdm

import time
from datetime import datetime
import numpy as np


def network_setup(train_evaluator, test_evaluator, rng, net_type="CNN"):

    if net_type=="CNN":
        network = NetworkMapperGiga["CNN"](
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
            num_output_units=3,
            output_activation="gaussian",
        )
        # Channel last configuration for conv!
        pholder = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        net_params = network.init(
            rng,
            x=pholder,
            rng=rng,
        )

        train_evaluator.set_apply_fn(network.apply)
        test_evaluator.set_apply_fn(network.apply)

    elif net_type=="LSTM_CNN":
        network = NetworkMapperGiga["LSTM_CNN"](
            # CNN
            num_output_units_cnn=64,
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
            num_output_units_lstm=3,
            output_activation_lstm="gaussian",
        )
        # Channel last configuration for conv!
        pholder = jnp.zeros((1, *train_evaluator.env.resolution, 3))
        carry_init = network.initialize_carry(batch_dims=(1,))
        net_params = network.init(
            rng,
            x=pholder,
            carry=carry_init,
            rng=rng,
        )

        train_evaluator.set_apply_fn(network.apply, network.initialize_carry)
        test_evaluator.set_apply_fn(network.apply, network.initialize_carry)
      
    else:
        raise NotImplementedError

    return net_params, train_evaluator, test_evaluator


def run_gigastep_fitness(
  scenario_name: str = "identical_5_vs_5", 
  network_type: str = "LSTM_CNN"
  ):
    num_generations = 1000
    evaluate_every_gen = 10
    log_tensorboard = True

    if log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logdir = './logdir/evosax/cr_fm_nes/' + datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        writer = SummaryWriter(log_dir=logdir, flush_secs=10)

    rng = jax.random.PRNGKey(0)

    train_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=False)
    test_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=True, n_devices=1)

    # Initialize network depending on type
    out = network_setup(train_evaluator, test_evaluator, rng, net_type=network_type)
    (net_params, train_evaluator, test_evaluator) = out

    train_param_reshaper = ParameterReshaper(net_params)
    test_param_reshaper = ParameterReshaper(net_params, n_devices=1)

    # fitness_shaper = FitnessShaper(maximize=True)

    strategy = CR_FM_NES(popsize=30, num_dims=train_param_reshaper.total_params, maximize=True)
    es_state = strategy.initialize(rng)

    es_logging = ESLog(
        num_dims=train_param_reshaper.total_params,
        num_generations=num_generations,
        top_k=5,
        maximize=True,
    )
    es_log = es_logging.initialize()

    # Run the ask-eval-tell loop
    log_steps, log_return = [], []
    t = tqdm.tqdm(range(1, num_generations + 1), desc="CR-FM-NES", leave=True)
    for gen in t:
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        
        # Sample parameters for the ego team
        x, es_state = strategy.ask(rng_gen, es_state)
        x_re = train_param_reshaper.reshape(x)

        # Get mean parameters for the ado team
        x_mean = jnp.repeat(es_state.mean[None], x.shape[0], 0)  # TODO: check if repeat can be removed
        x_mean_re = train_param_reshaper.reshape(x_mean)

        # Rollout fitness and update parameter distribution
        scores = train_evaluator.rollout(rng_eval, x_re, x_mean_re)
        # fitness = fitness_shaper.apply(x, scores)
        es_state = strategy.tell(x, scores, es_state)

        # Update the logging instance.
        es_log = es_logging.update(es_log, x, scores)

        if log_tensorboard:
            for key, value in es_log.items():
                if not "params" in key:
                    val = value[gen-1] if len(value.shape) else value
                    writer.add_scalar('train/' + key, np.array(val), gen)

        # Sporadically evaluate the mean & best performance on test evaluator.
        if (gen + 1) % evaluate_every_gen == 0:
            rng, rng_test = jax.random.split(rng)
            # Stack best params seen & mean strategy params for eval
            best_params = es_log["top_params"][0]
            mean_params = es_state.mean

            x_test = jnp.stack([best_params, mean_params], axis=0)
            x_test_re = test_param_reshaper.reshape(x_test)

            x_mean_test = jnp.repeat(es_state.mean[None], x_test.shape[0], 0)  # TODO: check if repeat can be removed
            x_mean_test_re = train_param_reshaper.reshape(x_mean_test)

            test_scores, test_images_global, test_images_ego = test_evaluator.rollout(
                rng_test, x_test_re, x_mean_test_re
            )
            # test_fitness = fitness_shaper.apply(x, test_scores)

            if log_tensorboard:
                ### Video
                id_prm = 0
                id_run = 0
                id_ego = 0
                frames_ego = np.array(test_images_ego[id_prm, id_run, :, id_ego])  # NOTE: this for agent-specific obs
                frames_glb = np.array(test_images_global[id_prm, id_run])

                frames_ego = np.transpose(frames_ego, (0, 3, 1, 2))  # (T, C, W, H)
                frames_ego = np.expand_dims(frames_ego, axis=0)      # (N, T, C, W, H)
                writer.add_video("test/video-ego", frames_ego, global_step=gen, fps=15)

                frames_glb = np.transpose(frames_glb, (0, 3, 1, 2))  # (T, C, W, H)
                frames_glb = np.expand_dims(frames_glb, axis=0)      # (N, T, C, W, H)
                writer.add_video("test/video-all", frames_glb, global_step=gen, fps=15)
            
            test_return_to_log = test_scores[1]
            log_steps.append(train_evaluator.total_env_steps)
            log_return.append(test_return_to_log)
            t.set_description(f"R: " + "{:.3f}".format(test_return_to_log.item()))
            t.refresh()
    

if __name__ == "__main__":
    t_start = time.time()
    run_gigastep_fitness("identical_5_vs_5", "LSTM_CNN")
    t_end = time.time()
    print("Runtime = " + str(round(t_end-t_start, 3)))
