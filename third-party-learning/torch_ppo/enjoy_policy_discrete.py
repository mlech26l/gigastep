# import wandb
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["export XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from gigastep import make_scenario, GigastepViewer
import jax
import jax.numpy as jnp
import cv2
import numpy as np
import sys
import time
import torch



def get_action_jax(actor_critic, obs,n_ego_agent,recurrent_hidden_states,device):
    obs = torch.tensor(np.asarray(obs),device=device)
    obs = torch.unsqueeze(obs, 0)
    obs = torch.moveaxis(obs, -1,2)
    obs = obs[:,:n_ego_agent,::]
    obs = obs.float().contiguous()
    masks = torch.ones(1, 1).to(device)
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs.float(), recurrent_hidden_states, masks, deterministic=True)

    action_view = torch.squeeze(action, 0)
    # action_view = torch.squeeze(action_view, -1) # double check
    return action_view, recurrent_hidden_states
def get_action_jax_opponent(actor_critic, actor_critic_opponent, obs,n_ego_agent, recurrent_hidden_states,device):
    obs = torch.tensor(np.asarray(obs),device=device)
    obs = torch.unsqueeze(obs, 0)
    obs = torch.moveaxis(obs, -1,2)
    obs = obs[:,n_ego_agent:,::]
    obs = obs.float().contiguous()
    masks = torch.ones(1, 1).to(obs.device)
    if actor_critic_opponent is None:
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs.float(), recurrent_hidden_states, masks, deterministic=True)
        action_view = torch.squeeze(action, 0)
        action_view = torch.zeros_like(action_view)
    else:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic_opponent.act(
                obs.float(), recurrent_hidden_states, masks, deterministic=True)
        action_view = torch.squeeze(action, 0)

    return action_view, recurrent_hidden_states


    
def evaluation_jax(env_name,obs_type,discrete_actions, actor_critic,actor_critic_opponent, num_of_evaluation = 1, device = None,
                   headless = True, env_cfg = None, show_num_agents = 0, random_action = False):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    viewer = GigastepViewer(frame_size=200, show_num_agents=show_num_agents, headless= headless)


    env = make_scenario(env_name,
                        **vars(env_cfg)
                        )
    rng = jax.random.PRNGKey(3)
    key_reset, rng = jax.random.split(rng, 2)

    recurrent_hidden_states = torch.zeros(env.n_teams[0],actor_critic.recurrent_hidden_state_size,device=device)
    recurrent_hidden_states_opponent = torch.zeros(env.n_teams[1],actor_critic.recurrent_hidden_state_size,device=device)

    state, obs = env.reset(key_reset)
    ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)

    wining_team = 0
    log_dict = {}
    for i in range(num_of_evaluation):
        images = []
        reward_episode = torch.zeros((2, 1)).to(device)
        rng = jax.random.PRNGKey(np.random.randint(0,100,1)[0])
        key_reset, rng = jax.random.split(rng, 2)
        state, obs = env.reset(key_reset)
        step_number = 0

        while True:
            rng,key_action = jax.random.split(rng, 2)
            if not random_action:
                action, recurrent_hidden_states = get_action_jax(actor_critic,
                                                                 obs,
                                                                 env.n_teams[0],
                                                                 recurrent_hidden_states,
                                                                 device)

                action_opponent, recurrent_hidden_states_opponent = get_action_jax_opponent(actor_critic,
                                                                        actor_critic_opponent,
                                                                        obs,
                                                                        env.n_teams[0],
                                                                        recurrent_hidden_states_opponent,
                                                                        device)

                if discrete_actions:
                    action_jax = jnp.array(action.int().detach().cpu().numpy().astype(np.int32))
                    action_opponent_jax = jnp.array(action_opponent.int().detach().cpu().numpy().astype(np.int32))
                else:
                    action_jax = jnp.array(action.detach().cpu().numpy().astype(np.float32))
                    action_opponent_jax = jnp.array(action_opponent.detach().cpu().numpy().astype(np.float32))

                action = jnp.concatenate([action_jax, action_opponent_jax], axis=0)
            else:
                action = jax.random.uniform(key_action, shape=(env.n_agents, 3), minval=-1, maxval=1,dtype=jnp.float32)


            rng,key_step = jax.random.split(rng, 2)
            state, obs, rewards, dones, ep_dones = env.step(state, action, key_step)
            # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
            # rewards is a float32 array of shape [batch_size, n_agents]
            # dones is a bool array of shape [batch_size, n_agents]
            # ep_done is a bool array of shape [batch_size]


            reward = torch.unsqueeze(torch.tensor(np.asarray(rewards)).to(device),0)
            reward_ego = torch.sum(reward[:,:env.n_teams[0]],dim=-1,keepdim=True)
            reward_opponent = torch.sum(reward[:,env.n_teams[0]:],dim=-1,keepdim=True)
            reward_episode += torch.concatenate([reward_ego,reward_opponent],dim=0)
            img = viewer.draw(env, state, obs)
            images.append(img)
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(0.05)
            if jnp.all(ep_dones):

                break

            step_number += 1
        

        alive_team1 = jnp.sum(state[0]["alive"] * (state[0]["team"] == 0))
        alive_team2 = jnp.sum(state[0]["alive"] * (state[0]["team"] == 1))
        win_or_loss = float(alive_team1 > alive_team2)
        wining_team += win_or_loss

        # print("evaluation")
        # print(f"team1 reward{reward_episode[0]}, team2 reward{reward_episode[1]}")
        # print(f"step number {step_number}")
        # print(f"team1 win {win_or_loss}, team2 win { 1 - win_or_loss}")

    # print(f"In total {i+1} evaluation")
    print(f"team1 win {wining_team}, team2 win {num_of_evaluation - wining_team} in total {num_of_evaluation}")

    return np.moveaxis(np.array(images),-1,1), wining_team/num_of_evaluation



if __name__ == "__main__":
    class Cfg():
        def __init__(self):
            pass
    env_cfg = Cfg()
    env_cfg.reward_game_won = 0
    env_cfg.reward_defeat_one_opponent = 0
    env_cfg.reward_detection = 0
    env_cfg.reward_idle = 0
    env_cfg.reward_collision = 0
    env_cfg.reward_damage = 1
    import jax.numpy as jnp
    env_cfg.cone_depth = 15.0
    env_cfg.cone_angle = jnp.pi * 1.99
    env_cfg.enable_waypoints = False
    env_cfg.use_stochastic_obs = False
    env_cfg.use_stochastic_comm = False
    env_cfg.max_agent_in_vec_obs = 40
    env_cfg.resolution_x = 84
    env_cfg.resolution_y = 84
    env_cfg.discrete_actions = True
    env_cfg.obs_type = "vector"

    actor_critic,_ = torch.load("trained_models/2v2/ppogigastepgiga20230523-105128.027506.pt",
                                # "ppogigastepgiga20230521-144137.667049.pt",
                                map_location=torch.device('cpu'))
    device = torch.device("cuda:0")
    actor_critic.to(device)
    evaluation_jax("identical_2_vs_2", obs_type= "vector",
                   discrete_actions = True,
                   actor_critic = actor_critic,
                   actor_critic_opponent= None ,
                   num_of_evaluation = 6,
                   device = device,
                   headless= False,
                   env_cfg = env_cfg,
                   show_num_agents= 0)