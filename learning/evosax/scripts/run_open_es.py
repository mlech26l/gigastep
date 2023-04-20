import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from evosax import OpenES, ParameterReshaper, NetworkMapper
from learning.evosax.problems.gigastep import GigastepFitness
from evosax.utils import ESLog
from evosax.utils import FitnessShaper  # NOTE: will move to evosax.core
import tqdm


def run_gigastep_fitness(scenario_name: str = "identical_5_vs_5"):
    num_generations = 100
    evaluate_every_gen = 10

    rng = jax.random.PRNGKey(0)

    train_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=False)
    test_evaluator = GigastepFitness(scenario_name, num_rollouts=20, test=True, n_devices=1)

    network = NetworkMapper["CNN"](
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
    pholder = jnp.zeros((1, 84, 84, 3))
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )

    train_param_reshaper = ParameterReshaper(net_params)
    test_param_reshaper = ParameterReshaper(net_params, n_devices=1)

    train_evaluator.set_apply_fn(network.apply)
    test_evaluator.set_apply_fn(network.apply)

    fitness_shaper = FitnessShaper(maximize=True)

    strategy = OpenES(popsize=30, num_dims=train_param_reshaper.total_params)
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
    t = tqdm.tqdm(range(1, num_generations + 1), desc="Open-ES", leave=True)
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
        fitness = fitness_shaper.apply(x, scores)
        es_state = strategy.tell(x, fitness, es_state)

        # Update the logging instance.
        es_log = es_logging.update(es_log, x, fitness)

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

            test_scores = test_evaluator.rollout(
                rng_test, x_test_re, x_mean_test_re
            )
            test_fitness = fitness_shaper.apply(x, test_scores)

            test_fitness_to_log = test_fitness[1]
            log_steps.append(train_evaluator.total_env_steps)
            log_return.append(test_fitness_to_log)
            t.set_description(f"R: " + "{:.3f}".format(test_fitness_to_log.item()))
            t.refresh()
    

if __name__ == "__main__":
    run_gigastep_fitness()
