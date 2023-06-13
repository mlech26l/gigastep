import os
import sys
import time

import cv2
import jax
import numpy as np

import copy

from gigastep import GigastepViewer, GigastepEnv, make_scenario, ScenarioBuilder
from gigastep.evaluator import Evaluator, loop_env_vectorized_evosax
import jax.numpy as jnp

# from learning.evosax.scripts.run_strategy import network_setup
from learning.evosax.networks import NetworkMapperGiga


def network_setup(obs_shape, rng, env_cfg, net_cfg, net_type="LSTM"):
    
    rng_ego, rng_ado = jax.random.split(rng, 2)
    
    output_activation = "tanh_gaussian"  # "tanh_gaussian"
    output_dimensions = 3
    if "discrete_actions" in env_cfg:
        if env_cfg["discrete_actions"]:
            output_activation = "categorical"
            output_dimensions = 9

    if net_type=='LSTM':
        ### Ego
        network_ego = NetworkMapperGiga["LSTM"](
            num_hidden_units=net_cfg["lstm_units"],  # 32,
            num_output_units=output_dimensions,
            output_activation=output_activation,
        )
        pholder_ego = jnp.zeros((1, *obs_shape))
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
        pholder_ado = jnp.zeros((1, *obs_shape))
        carry_ado_init = network_ado.initialize_carry()
        net_ado_params = network_ado.init(
            rng_ado,
            x=pholder_ado,
            carry=carry_ado_init,
            rng=rng_ado,
        )

    else:
        raise NotImplementedError
    
    network_dict = {}
    network_dict["params"] = net_ego_params
    network_dict["apply"] = network_ado.apply
    network_dict["carry"] = network_ego.initialize_carry

    return network_dict





if __name__ == "__main__":
    # convert -delay 3 -loop 0 video/scenario/frame_*.png video/scenario.webp

    trained_agent_dict = {
        "identical_5_vs_5": {
            "discrete_actions": True,
            "obs_type": "vector",
            "path": [
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig10_envconfig2_lstmconfig0/23_06_05_11_00_37/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig20_envconfig2_lstmconfig0/23_06_05_11_10_04/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig22_envconfig2_lstmconfig0/23_06_05_11_19_19/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_0",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig23_envconfig2_lstmconfig0/23_06_05_10_50_26/log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig27_envconfig2_lstmconfig0/23_06_05_11_28_34/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/des/identical_5_vs_5/algconfig28_envconfig2_lstmconfig0/23_06_05_11_37_49/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/ars/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_22_04/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/ars/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_22_04/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gesmr_ga/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_23_54/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gesmr_ga/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_23_54/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gld/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_23_11_20/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gld/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_23_11_20/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/guidedes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_30_09/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/guidedes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_30_09/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/persistentes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_21_38/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/persistentes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_16_21_38/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/samr_ga/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_19_04_01/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/samr_ga/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_19_04_01/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/simplega/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_15_56_59/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/simplega/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_15_56_59/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_17_46_57/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_17_46_57/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/xnes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_20_44_18/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/xnes/identical_5_vs_5/algconfig0_envconfig2_lstmconfig0/23_06_05_20_44_18/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/ars/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_31_19/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/ars/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_31_19/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gesmr_ga/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_33_14/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gesmr_ga/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_33_14/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gld/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_23_20_36/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/gld/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_23_20_36/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/guidedes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_39_28/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/guidedes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_39_28/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/persistentes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_30_56/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/persistentes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_16_30_56/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/samr_ga/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_19_13_18/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/samr_ga/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_19_13_18/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/simplega/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_15_56_05/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/simplega/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_15_56_05/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/xnes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_21_16_54/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/xnes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_21_16_54/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig0_envconfig1_lstmconfig0/23_06_05_17_56_14/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_0",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/snes/identical_5_vs_5/algconfig1_envconfig2_lstmconfig0/23_06_06_06_06_03/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ego_log_iter_2000",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_0",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_200",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_400",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_600",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_800",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_1000",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_1200",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_1400",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_1600",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_1800",
                "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ego_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_0",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig3_envconfig2_lstmconfig0/23_06_06_10_48_33/ado_log_iter_2000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_0",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_1000",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_1200",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_1400",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_1600",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_1800",
                # "/home/timlocal/projects/gigastep/logdir/evosax/eval/v2/openes/identical_5_vs_5/algconfig4_envconfig2_lstmconfig0/23_06_06_10_20_50/ado_log_iter_2000",
            ]
        },
    }
    
    for env_name, load_config in trained_agent_dict.items():

        ### Environment config
        # Base config
        env_cfg0 = {}
        # env_cfg0["config_name"] = "config0"
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
        ### Network config
        ### LSTM
        # Base Config
        net_cfg0 = {}
        net_cfg0["config_name"] = "config0"
        net_cfg0["lstm_units"] = 32

        ## create env
        env = make_scenario(env_name, **env_cfg0)


        print(f"env_name: {env_name}")

        obs_shape = env.observation_space.low.shape

        rng = jax.random.PRNGKey(0)
        network_dict = network_setup(obs_shape, rng, env_cfg2, net_cfg0, net_type="LSTM")


        from evosax.utils import ESLog, ParameterReshaper
        param_ego_reshaper = ParameterReshaper(network_dict["params"], n_devices=1)

        chkpt_nums = []
        winrates = []
        tierates = []
        loserates = []

        for chkpt_path in trained_agent_dict["identical_5_vs_5"]["path"]:

            path_name = chkpt_path.split("/")
            alg_name = path_name[-5]
            cfg_name = path_name[-3].split("_")[0]
            agt_name = path_name[-1].split("_")[0]
            ckp_name = path_name[-1].split("_")[-1]

            chkpt_nums.append(int(ckp_name))

            print("Evaluating " + alg_name + " on alg " + cfg_name + " with " + agt_name + " agent checkpoint " + ckp_name)

            es_ego_logging = ESLog(
                                num_dims=param_ego_reshaper.total_params,
                                num_generations=1,
                                top_k=5,
                                maximize=True,
                            )
            es_ego_log = es_ego_logging.initialize()
            if int(ckp_name) == 0:
                params_flat = param_ego_reshaper.flatten_single(network_dict["params"])
                network_dict["params"] = param_ego_reshaper.reshape(jnp.repeat(params_flat[None], 20, axis=0))
            else:
                es_ego_log = es_ego_logging.load(filename=chkpt_path)
                network_dict["params"] = param_ego_reshaper.reshape(jnp.repeat(es_ego_log["top_params"][0][None], 20, axis=0))


            # actor_critic,_ = torch.load(
            #     load_config["path"],
            #     map_location=torch.device('cpu')
            # )
            # device = torch.device("cuda:0")
            # actor_critic.to(device)

            wining_rate_vec = loop_env_vectorized_evosax(
                env=env,
                network_dict=network_dict,
                device = "cuda:0"
            )
            # wining_rate = loop_env(
            #     env=env,
            #     policy=actor_critic,
            #     device = "cuda:0",
            #     headless = False
            # )

            winrates.append(wining_rate_vec[0])
            loserates.append(wining_rate_vec[1])
            tierates.append(wining_rate_vec[2])

        results = np.stack((chkpt_nums, winrates, tierates, loserates), axis=0)
        np.savetxt("/".join(path_name[:-1]) + "/" + agt_name + "_results.csv", results, delimiter=",")
