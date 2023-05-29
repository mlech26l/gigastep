# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple

import dm_env

from mava.utils.jax_training_utils import set_jax_double_precision
from mava.wrappers.gigastep import GigastepParallelEnvWrapper
from mava.wrappers.env_preprocess_wrappers import (
    ConcatAgentIdToObservation,
    StackObservations,
)

from gigastep import make_scenario


def make_environment(
    env_name: str = "identical_5_vs_5",
    obs_type: str = "vector",
    use_stochastic_obs: bool = False,
    discrete_actions: bool = True,
    random_seed: Optional[int] = 42,
    **kwargs: Any,
) -> Tuple[dm_env.Environment, Dict[str, str]]:
    # from jax import config
    if discrete_actions:
        # Env uses int64 action space due to the use of spac.Discrete.
        set_jax_double_precision()

    env_module = make_scenario(
        env_name,
        obs_type=obs_type,
        use_stochastic_obs=use_stochastic_obs,
        discrete_actions=discrete_actions,
    )
    environment = GigastepParallelEnvWrapper(
        env_module, random_seed=random_seed,
    )

    environment_task_name = {
        "environment_name": "gigastep",
        "task_name": env_name,
    }

    return environment, environment_task_name
