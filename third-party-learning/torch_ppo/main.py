import os
def main(**sweep_dict):
    import os
    os.environ["JAX_PLATFORM_NAME"]="cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["PYTORCH_CUDA_ALLOC_CONN"]="max_spt_size_mb:20024"

    import time
    from collections import deque
    import numpy as np

    import gigastep
    from gigastep import make_scenario

    import jax
    import jax.numpy as jnp


    import torch

    from a2c_ppo_acktr import algo, utils
    from a2c_ppo_acktr.algo import gail
    from a2c_ppo_acktr.arguments import get_args
    from a2c_ppo_acktr.envs import make_vec_envs
    from a2c_ppo_acktr.model import Policy
    from a2c_ppo_acktr.storage import RolloutStorage
    from evaluation import evaluate
    from gigastep.evaluator import Evaluator, loop_env_vectorized, loop_env

    from enjoy_policy_discrete import evaluation_jax


    args = get_args()
    if sweep_dict is not None:
        args = tune_args(args=args,sweep_dict = sweep_dict)
    else:
        args = tune_args(args=args)
    class Cfg():
        def __init__(self):
            pass
    env_cfg = Cfg()
    env_cfg = tune_cfg(env_cfg,sweep_dict)
    

    from datetime import datetime as dt
    if args.log_to_wandb:
        import wandb
        import yaml
        project_name = f'{args.project_name}-{args.env_name}'
        exp_prefix = f'{args.variant_name}-{dt.now():%D-%H%M%S.%f}'

        wandb.login(
            key=yaml.safe_load(
                open("learning/torch_ppo/.secret.yml", 'r'))["WANDB_KEY"]
        )

        config_del = vars(args)
        config_del.update(vars(env_cfg))

        wandb.init(
            name=exp_prefix,
            group=args.variant_name,
            project= project_name,
            config=config_del
        )
        
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.log_dir is None:
        log_dir = args.log_dir
        eval_log_dir = log_dir
    else:
        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)


    torch.set_num_threads(1)

    device = torch.device("cuda:"+str(args.cuda_id) if args.cuda else "cpu")
    # device = torch.device("cpu")
    print(args.env_name)
    print(f"device: {device}")


    if "gigastep" in args.env_name:
        args.num_processes = args.batch_size
        envs = make_scenario(args.env_sub_name,
                             **vars(env_cfg))

        from gigastep.jax_utils import Box
        if envs._obs_type == "rgb":
            envs.observation_space = Box(
                low=jnp.zeros([3, envs.resolution[0], envs.resolution[1]], dtype=jnp.uint8),
                high=255
                * jnp.ones([3, envs.resolution[0], envs.resolution[1]], dtype=jnp.uint8),
            )
        n_ego_agents = 0 
        n_opponents = 0
        for i in envs._per_agent_team:
            if i==0:
                n_ego_agents += 1 
            else:
                n_opponents += 1 


    
    PATH=os.getcwd()+ args.pretrained_dir
    if os.path.exists(PATH):
        actor_critic, obs_rms = \
            torch.load(PATH,
                        map_location='cpu')
    else:
        if "gigastep" in args.env_name:
            if "_distr" in args.model_name:
                base_kwargs = {'recurrent': args.recurrent_policy,
                               'num_agent': n_ego_agents,
                               "hidden_size": 64,
                               "num_inputs": envs.observation_space.shape,
                               "device": device,
                               "value_decomposition": True}
                base_kwargs_opponent = base_kwargs.copy()
                base_kwargs_opponent["num_agent"] = n_opponents
            else:
                base_kwargs = {'recurrent': args.recurrent_policy,
                               "hidden_size": 64,
                               }
                base_kwargs_opponent = base_kwargs.copy()
            model_kwargs ={"model": args.model_name,
                            'num_agent': n_ego_agents,
                            "distribution": args.distribution_name,
                            "max_output": 1
                        }
            model_kwargs_opponent = model_kwargs.copy()
            model_kwargs_opponent["num_agent"] = n_opponents

            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs = base_kwargs,
                model_kwargs= model_kwargs
            )
            actor_critic_opponent = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs = base_kwargs_opponent,
                model_kwargs= model_kwargs_opponent
            )

        else:
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy},
                model_kwargs={'model': args.model_name}
            )

    actor_critic.to(device)
    actor_critic_opponent.to(device)

    if args.log_to_wandb:
        wandb.watch(actor_critic)
        wandb.watch(actor_critic_opponent)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        agent_opponent= algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        agent_opponent = algo.PPO(
            actor_critic_opponent,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr = args.lr,
            eps = args.eps,
            max_grad_norm = args.max_grad_norm)
        


    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size


    if "gigastep" in args.env_name:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                             envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,n_ego_agents)
        rollouts_opponent = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic_opponent.recurrent_hidden_state_size,
                                           n_opponents)
    rng = jax.random.PRNGKey(args.seed)
    rng, key_reset = jax.random.split(rng, 2)
    key_reset = jax.random.split(key_reset, int(args.batch_size))
    state, obs = envs.v_reset(key_reset)
    obs = torch.tensor(np.asarray(obs))
    obs = torch.moveaxis(obs, -1,2)
    obs_opponent = obs[:,n_ego_agents:,:]
    done_opponent = torch.zeros((obs_opponent.shape[0],1),device=device)

    obs_opponent = obs_opponent.to(device)
    done_opponent = done_opponent.to(device)

    obs = obs[:,:n_ego_agents,:]
    rollouts.obs[0].copy_(obs)
    rollouts_opponent.obs[0].copy_(obs_opponent)

    episode_rewards = deque(maxlen=args.num_steps)
    episode_rewards_opponent = deque(maxlen=args.num_steps)
    episode_step = deque(maxlen=args.num_steps)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    reward_episode_env = torch.zeros((obs.shape[0],1),device=device)
    reward_episode_env_opponent = torch.zeros((obs_opponent.shape[0],1),device=device)
    step_agent = torch.rand((int(args.batch_size)),device=device)*args.max_steps
    saving_path_vec = []
    saving_path_vec_rank = []

    # main training parts
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent_opponent.optimizer,j, num_updates,
                agent_opponent.optimizer.lr if args.algo == "acktr" else args.lr)



        for step in range(args.num_steps):

            with torch.no_grad():
                actor_critic = actor_critic.to(device) # debug
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device),masks_agent=rollouts.masks_agent[step].to(device))
                value_opponent, action_opponent, action_log_prob_opponent, recurrent_hidden_states_opponent = actor_critic_opponent.act(
                    rollouts_opponent.obs[step].to(device), rollouts_opponent.recurrent_hidden_states[step].to(device),
                    rollouts_opponent.masks[step].to(device), masks_agent=rollouts_opponent.masks_agent[step].to(device))

            # action = action.int()
            if "gigastep" in args.env_name:

                action_view = torch.concatenate([action, action_opponent], dim=1)

                if envs.action_space.__class__.__name__ == 'Discrete':
                    action_jax = jnp.array(action_view.detach().cpu().numpy().astype(np.int32))
                else:
                    action_jax = jnp.array(action_view.detach().cpu().numpy().astype(np.float32))
                rng, key_action, key_step = jax.random.split(rng, 3)
                key_step = jax.random.split(key_step, int(args.batch_size))

                state, obs, reward, agent_done, episode_done = envs.v_step(state, action_jax, key_step)

                step_agent += 1

                reward = torch.tensor(np.asarray(reward)).to(device)
                reward_raw = reward.clone()

                reward = torch.sum(reward_raw[:,:n_ego_agents], dim=-1, keepdim=True)
                reward_opponent = torch.sum(reward_raw[:,n_ego_agents:], dim=-1, keepdim=True)


                reward_episode_env += reward
                reward_episode_env_opponent +=torch.sum(reward_raw[:,n_ego_agents:],dim=-1,keepdim=True)

                done = torch.tensor(np.asarray(episode_done)).to(device)
                masks_agent_all = torch.tensor(np.asarray(1 - agent_done)).to(device)


                done = torch.logical_or(done, torch.greater(step_agent,args.max_steps)).detach()
                infos = [dict() for i in range(obs.shape[0])]
                for i in range(done.shape[0]):
                    if done[i]:
                        infos[i] = {"episode": {"r": reward_episode_env[i,0].detach().cpu().numpy(),
                                                "r_opponent": reward_episode_env_opponent[i,0].detach().cpu().numpy(),
                                                "step": step_agent[int(i)].detach().cpu().numpy()}}
                        reward_episode_env[i,0] = 0
                        reward_episode_env_opponent[i,0] = 0
                        step_agent[int(i)] = 0
                if jnp.any(episode_done):
                    rng, key = jax.random.split(rng, 2)
                    state, obs = envs.reset_done_episodes(state, obs, episode_done, key)
                obs = torch.tensor(np.asarray(obs)).to(device)
                obs = torch.moveaxis(obs, -1, 2)

                obs_opponent = obs[:,-n_opponents:, :]
                done_opponent = episode_done
                obs = obs[:, :n_ego_agents, :]
                masks_agent = masks_agent_all[:, :n_ego_agents]
                masks_agent_opponent = masks_agent_all[:, n_ego_agents:]
            else:
                obs, reward, done, infos = envs.step(torch.squeeze(action))

            masks_agent = masks_agent.unsqueeze(-1)
            masks_agent_opponent = masks_agent_opponent.unsqueeze(-1)

            for info in infos:
                if info.get('episode') is not None:
                    episode_rewards.append(info['episode']["r"])
                    episode_rewards_opponent.append(info['episode']["r_opponent"])
                    episode_step.append(info['episode']["step"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])


            bad_masks = torch.FloatTensor(
                 [[1.0]
                  for info in infos])

            rollouts.insert(obs.to("cpu"),
                            recurrent_hidden_states.to("cpu"),
                            action.to("cpu"),
                            action_log_prob.to("cpu"),
                            value.to("cpu"),
                            reward.to("cpu"),
                            masks.to("cpu"),
                            bad_masks.to("cpu"),
                            masks_agent=masks_agent.to("cpu"))
            rollouts_opponent.insert(obs_opponent.to("cpu"),
                                    recurrent_hidden_states_opponent.to("cpu"),
                                    action_opponent.to("cpu"),
                                    action_log_prob_opponent.to("cpu"),
                                    value_opponent.to("cpu"),
                                    reward_opponent.to("cpu"),
                                    masks.to("cpu"),
                                    bad_masks.to("cpu"),
                                    masks_agent=masks_agent_opponent.to("cpu"))


        with torch.no_grad():

            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device), rollouts.masks_agent[-1].to(device)).detach()
            next_value_opponent = actor_critic_opponent.get_value(
                rollouts_opponent.obs[-1].to(device), rollouts_opponent.recurrent_hidden_states[-1].to(device),
                rollouts_opponent.masks[-1].to(device), rollouts_opponent.masks_agent[-1].to(device)).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        rollouts_opponent.compute_returns(next_value_opponent, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)

        rollouts.to(device)
        rollouts_opponent.to(device)
        agent.actor_critic.to(device)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        value_loss_opponent, action_loss_opponent, dist_entropy_oppoent = agent_opponent.update(rollouts_opponent)


        rollouts.after_update()
        rollouts_opponent.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                if not os.path.exists(save_path):
                   os.makedirs(save_path)
            except OSError:
                pass

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            if args.log_to_wandb:
                model_pre_temp_name = f"{dt.now():%Y%m%d-%H%M%S.%f}"
                model_save_path = args.save_dir + args.algo + args.env_name + args.save_name_prex + model_pre_temp_name + ".pt"
                model_save_path_opponent = args.save_dir + args.algo + args.env_name + args.save_name_prex + "_oppo_"+ model_pre_temp_name + ".pt"

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], model_save_path)
                torch.save([actor_critic_opponent,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], model_save_path_opponent)
                print(f"model saved to {model_save_path}")
                wandb.save(model_save_path)
                wandb.save(model_save_path_opponent)
                saving_path_vec.append(model_save_path)
                saving_path_vec_rank.append(np.mean(episode_rewards))
            else:
                model_pre_temp_name = f"{dt.now():%Y%m%d-%H%M%S.%f}"
                model_save_path = args.save_dir + args.algo + args.env_name + args.save_name_prex + model_pre_temp_name + ".pt"
                model_save_path_opponent = args.save_dir + args.algo + args.env_name + args.save_name_prex + "_oppo_"+ model_pre_temp_name + ".pt"
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], model_save_path)
                torch.save([actor_critic_opponent,
                            getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                            ], model_save_path_opponent)
                saving_path_vec.append(model_save_path)
                saving_path_vec_rank.append(np.mean(episode_rewards))


            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n\
                    dist_entropy: {}, value_loss {}, action_loss {}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            if args.log_to_wandb:
                wand_log = {"total_num_steps":total_num_steps,"FPS": int(total_num_steps / (end - start)),
                       "last training episode":len(episode_rewards),"reward": np.mean(episode_rewards),
                       "dist_entropy":dist_entropy,"value_loss":value_loss,"action_loss":action_loss,
                          "value_loss_opponent":value_loss_opponent,"action_loss_opponent":action_loss_opponent,
                        "step":np.mean(episode_step),
                        "reward_opponent":np.mean(episode_rewards_opponent),
                        "replay_index": 0 # replay_index
                        }

                print(wand_log)
                wandb.log(wand_log)



        if (j % args.eval_interval == 0) and ("giga" in args.env_name): # and len(episode_rewards)>1:
            if "giga" in args.env_name:
                for i in range(2):
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>evaluating in cpu>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    videos,wining_rate = evaluation_jax(args.env_sub_name, env_cfg.obs_type,
                                            env_cfg.discrete_actions,
                                            actor_critic,
                                            actor_critic_opponent= actor_critic_opponent,
                                            device=device,
                                            headless= True,
                                            env_cfg = env_cfg)
                    rate_elo = False
                    if rate_elo:
                        # # load rating system
                        from gigastep.evaluator import Rating
                        rate_cls = Rating(device, args.path_pool, args.env_sub_name, env_cfg, num_match=5)
                        for i in range(1):
                            rate_0 = rate_cls.rate_policy(actor_critic, rate_0,add_policy_to_pool=False)
                        if args.log_to_wandb:
                            wandb.log({
                                    "video":[wandb.Video(videos, fps=4, format="gif")],
                                    "wining_rate_simple":wining_rate,
                                    "Elo": rate_0
                            }
                            )
                    else:
                        if args.log_to_wandb:
                            wandb.log({
                                    "video":[wandb.Video(videos, fps=4, format="gif")],
                                    "wining_rate_simple":wining_rate,
                                    # "Elo": rate_0
                            }
                            )
                        
                #  evaluate ego and opponent 
                env_evaluation = make_scenario(args.env_sub_name,
                                               **vars(env_cfg)
                                               )
                wining_rate_vec_evaluation = loop_env_vectorized(env=env_evaluation,
                                                                 policy=actor_critic,
                                                                 device=device
                                                                 )
                if args.log_to_wandb:
                    wandb.log({
                        "wining_rate_team_ego_eval_0": wining_rate_vec_evaluation[0],
                        "wining_rate_team_ego_eval_1": wining_rate_vec_evaluation[1],
                        "wining_rate_tie_ego_eval": wining_rate_vec_evaluation[2],
                    }
                    )

                env_evaluation = make_scenario(args.env_sub_name,
                                               **vars(env_cfg)
                                               )
                wining_rate_vec_evaluation = loop_env_vectorized(env=env_evaluation,
                                                                 policy=actor_critic_opponent,
                                                                 switch_side= True,
                                                                 device=device
                                                                 )
                if args.log_to_wandb:
                    wandb.log({
                        "wining_rate_team_opp_eval_0": wining_rate_vec_evaluation[0],
                        "wining_rate_team_opp_eval_1": wining_rate_vec_evaluation[1],
                        "wining_rate_tie_opp_eval": wining_rate_vec_evaluation[2],
                    }
                    )


def tune_cfg(env_cfg,sweep_dict = None):

    resolution = 84 #
    env_cfg.resolution_x = resolution
    env_cfg.resolution_y = resolution

    env_cfg.obs_type = "vector"
    env_cfg.discrete_actions = True

    env_cfg.reward_game_won = 100
    env_cfg.reward_defeat_one_opponent = 100
    env_cfg.reward_detection = 0
    env_cfg.reward_damage = 0
    env_cfg.reward_idle = 0
    env_cfg.reward_hit_waypoint = 0

    import jax.numpy as jnp
    env_cfg.cone_depth = 15.0
    env_cfg.cone_angle = jnp.pi * 1.99
    env_cfg.enable_waypoints = False
    env_cfg.use_stochastic_obs = False
    env_cfg.use_stochastic_comm = False
    env_cfg.max_agent_in_vec_obs = 100

    
    return env_cfg


def tune_args(args, sweep_dict = None):
    args.env_name = "gigastep"
    # args.model_name = "Giga_distri_cnn"
    args.model_name = "Giga_distri_mlp"
    args.distribution_name = "MultiCategorical"
    # args.distribution_name = "DiagGaussianMulti"
    args.project_name = "Gigastep"
    args.algo = "ppo"
    args.use_gae = False
    args.gail = False
    args.gail_experts_dir = "gail_experts/"
    args.lr = 1e-3
    args.value_loss_coef = 0.5
    args.num_processes = args.batch_size
    args.num_mini_batch = 4
    args.log_interval = 20
    args.use_linear_lr_decay = True
    args.entropy_coef = 0.02
    args.log_dir = "/logs/gigasteps/"
    args.save_dir = "/logs/gigasteps/"
    args.path_pool = os.path.dirname(os.path.abspath(__file__)) \
                     + "/gigastep/evaluation/trained_models/5v5_torch/"
    args.max_grad_norm = 0.1
    args.num_env_steps = 5e8
    args.save_name_prex = "giga"
    args.variant_name = "selfplay"
    args.log_to_wandb = True
    args.num_frame_stack = 1
    args.eval_interval = args.num_env_steps \
                         // args.num_steps \
                         // args.num_processes \
                         // 100
    args.num_step_to_load_self_play = int(10)
    args.recurrent_policy = False
    args.max_steps = 490
    args.selfplay_simultaneously = True

    return args


if __name__ == "__main__":
    main()
