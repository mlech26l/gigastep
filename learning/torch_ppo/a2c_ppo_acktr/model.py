import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from time import time
import cv2

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian,\
                    DiagGaussianMulti, MultiCategorical
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None,model_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if True:
                    if "model" in model_kwargs.keys():
                        try:
                            base = eval(model_kwargs["model"])
                        except: 
                            print(base_kwargs["model"]+" is not defined")
                else:
                    base = CNNBase
            elif len(obs_shape) == 2:
                if "model" in model_kwargs.keys():
                    try:
                        base = eval(model_kwargs["model"])
                    except:
                        print(base_kwargs["model"] + " is not defined")

            elif len(obs_shape) == 1:
                if "model" in model_kwargs.keys():
                    base= eval(model_kwargs["model"])
                else:
                    base = MLPBase
            else:
                raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
            if "Giga_distri" in model_kwargs.keys():
                if model_kwargs["giga_distri"]:
                    self.dist = eval(model_kwargs["distribution"])(self.base.output_size, num_outputs,
                                            model_kwargs["num_agent"])

        elif action_space.__class__.__name__ == "Box":
            if "Giga_distri" in model_kwargs["model"]:
                num_outputs = action_space.shape[0]
                if "Multi" in model_kwargs["distribution"]:
                    self.dist = eval(model_kwargs["distribution"])(self.base.output_size, num_outputs,
                                              model_kwargs["num_agent"], model_kwargs["max_output"])
                else:
                    self.dist = eval(model_kwargs["distribution"])(self.base.output_size, num_outputs)
            else:
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, masks_agent = None, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, masks_agent)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, masks_agent = None):

        value, _, _ = self.base(inputs, rnn_hxs, masks, masks_agent)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, masks_agent = None):
        # Todo mask the action loss and dist_entropy for died agents: rem .
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, masks_agent)
        dist = self.dist(actor_features) # batch, num_agent, action_dim

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        # Change to reset 
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, kernel_size = 2, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size= 2, stride = 2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, kernel_size = 2, stride=1)), nn.ReLU(), Flatten())

        self.main_down = nn.Sequential(
            init_(nn.Linear(96, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        x = self.main_down(x)
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs




class MLPBase(NNBase):
    def __init__(self, num_inputs = None, recurrent=False, recurrent_input_size = None, hidden_size=64):
        recurrent_input_size = num_inputs if recurrent_input_size is None else recurrent_input_size
        super(MLPBase, self).__init__(recurrent, recurrent_input_size, hidden_size)
        
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class Giga_distri_cnn(MLPBase):
    def __init__(self, not_used, num_inputs, recurrent=False, hidden_size=64, num_agent=700,
                 device="cuda", value_decomposition=True, observation_type="vec position", use_cnn=True):
        num_inputs_agent = int(num_inputs[0]) if len(num_inputs)==1 else int(num_inputs[0]*num_inputs[1])
        super(Giga_distri_cnn, self).__init__( num_inputs=num_inputs_agent,
                                      recurrent=recurrent,
                                      recurrent_input_size = hidden_size,
                                      hidden_size=hidden_size)

        self.num_inputs = num_inputs
        self.use_cnn = use_cnn
        if recurrent:
            num_inputs = hidden_size
            self.num_inputs_agent = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.use_dual_input = True

        if self.use_cnn:

            self.backbone = nn.Sequential(
                init_(nn.Conv2d(3, 128, kernel_size=8, stride=4)), nn.ELU(),
                init_(nn.Conv2d(128, 64, kernel_size=8, stride=6)), nn.ELU(),
                Flatten(),
                init_(nn.Linear(576, hidden_size)), nn.ELU(),
            )
            self.actor = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.ELU()
            )

            self.critic_output_is_1 = nn.Sequential(
                init_(nn.Linear(hidden_size, 1)), nn.ELU()
            )
            self.critic_linear = init_(nn.Linear(hidden_size * num_agent, 1))

        self.num_agent = num_agent
        self.device = device
        self.train()

        self.shift_matrix = None
        self.value_decomposition = value_decomposition
        self.observation_type = observation_type

    def init_hidden_variable(self, inputs):
        rnn_hxs = torch.zeros((inputs.shape[0]*inputs.shape[1], self.hidden_size),device=self.device)
        return rnn_hxs


    def forward(self, x, rnn_hxs, masks, masks_agent = None):

        x = (x / 255).float()
        # batches, n, channel, w, height
        x_shape = x.shape
        x = x.view(x_shape[0] * x_shape[1], *self.num_inputs)

        x = self.backbone(x)
        if len(x.shape)>=3:
            raise Exception("backbone")
        
        x = x.view(x_shape[0], x_shape[1],-1)
        if masks_agent is not None:
            x = x * masks_agent



        if self.is_recurrent:
            x_shape = x.shape
            x = x.view( x_shape[0]*x_shape[1], x_shape[2])

            x, rnn_hxs = self._forward_gru(x, rnn_hxs.view(x_shape[0]*x_shape[1],-1),
                                           torch.repeat_interleave(masks,x_shape[1],dim=0))

            rnn_hxs = rnn_hxs.view(x_shape[0], x_shape[1], -1)
            x = x.view(x_shape[0], x_shape[1], x_shape[2])
        x_shape = x.shape
        x = x.view(x_shape[0] * x_shape[1], x_shape[2])
        hidden_actor = self.actor(x).view(x_shape[0], x_shape[1], -1)
        if self.value_decomposition:
            hidden_critic = self.critic_output_is_1(x).view(x_shape[0], x_shape[1])
            if masks_agent is not None:
                hidden_critic = hidden_critic * masks_agent.squeeze(-1)
            hidden_critic = torch.unsqueeze(torch.sum(hidden_critic, dim=1), dim=1)
        else:
            hidden_critic = self.critic(x).view(x_shape[0], x_shape[1], -1)
            hidden_critic = torch.flatten(hidden_critic, start_dim=1, end_dim=-1)
            hidden_critic = self.critic_linear(hidden_critic)

        if len((torch.isnan(hidden_actor).nonzero())) != 0:
            import pdb; pdb.set_trace()

        return hidden_critic, hidden_actor, rnn_hxs


class Giga_distri_mlp(MLPBase):

    def __init__(self, not_used, num_inputs, recurrent=False, hidden_size=64, num_agent=700,
                 device="cuda", value_decomposition=True, observation_type="vec position", use_cnn=True):
        num_inputs_agent = int(num_inputs[0])
        super(Giga_distri_mlp, self).__init__( num_inputs=num_inputs_agent,
                                               recurrent = recurrent,
                                              recurrent_input_size =hidden_size,
                                              hidden_size = hidden_size)
        self.num_inputs = num_inputs
        # if recurrent:
        #     num_inputs = hidden_size
        #     num_inputs_agent = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.use_dual_input = True
        
        self.backbone = nn.Sequential(
            init_(nn.Linear(self.num_inputs[0], hidden_size)), nn.ELU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ELU(),
        init_(nn.Linear(hidden_size, hidden_size)), nn.ELU()
        )
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.ELU()
        )

        self.critic_output_is_1 = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)), nn.ELU()
        )
        self.critic_linear = init_(nn.Linear(hidden_size * num_agent, 1))

        self.num_agent = num_agent
        self.device = device
        self.train()

        self.shift_matrix = None
        self.value_decomposition = value_decomposition
        self.observation_type = observation_type

    def forward(self, inputs, rnn_hxs, masks, masks_agent = None):
        # masks_agent: batch,num_agent,1
        x = (inputs).float()
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        #     
        # mask the died agent as zeros.
        if masks_agent is not None:
            x = x*masks_agent
        
        x_re = x.view(x.shape[0] * x.shape[1], *self.num_inputs)
        x = self.backbone(x_re).view(x.shape[0], x.shape[1],-1)
        
        if self.is_recurrent:
            x_shape = x.shape
            x = x.view(x_shape[0]*x_shape[1], x_shape[2])
            x, rnn_hxs = self._forward_gru(x, rnn_hxs.view(x_shape[0]*x_shape[1],-1),
                                           torch.repeat_interleave(masks,x_shape[1],dim=0))

            rnn_hxs = rnn_hxs.view(x_shape[0], x_shape[1], -1)
            x = x.view(x_shape[0], x_shape[1], x_shape[2])

        x_re = x.view(x.shape[0] * x.shape[1], x.shape[2])
        hidden_actor = self.actor(x_re).view(x.shape[0], x.shape[1], -1)

        if self.value_decomposition:
            hidden_critic = self.critic_output_is_1(x_re).view(x.shape[0], x.shape[1])
            # Remove the values for died agents using individual mask
            if masks_agent is not None:
                hidden_critic = hidden_critic * masks_agent.squeeze(-1)
            hidden_critic = torch.unsqueeze(torch.sum(hidden_critic, dim=1), dim=1)
        else:
            hidden_critic = self.critic(x_re).view(x.shape[0], x.shape[1], -1)
            hidden_critic = torch.flatten(hidden_critic, start_dim=1, end_dim=-1)
            hidden_critic = self.critic_linear(hidden_critic)

        if len((torch.isnan(hidden_actor).nonzero())) != 0:
            import pdb; pdb.set_trace()

        return hidden_critic, hidden_actor, rnn_hxs