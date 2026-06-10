from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule
from omegaconf import DictConfig

class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(PPOActor, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        self.num_actions = {'lower_body': 15, 'upper_body': 14}  # TODO: Hardcode
        # Action noise
        if isinstance(init_noise_std, dict) or isinstance(init_noise_std, DictConfig):
            std_list = []
            for key in self.num_actions.keys():
                if key not in init_noise_std:
                    raise ValueError(f"Key {key} not found in init_noise_std.")
                std_value = init_noise_std[key]
                num = self.num_actions[key]
                std_list.append(std_value * torch.ones(num))
            std_tensor = torch.cat(std_list)
        else:
            std_tensor = init_noise_std * torch.ones(num_actions)

        self.std = nn.Parameter(std_tensor)
        self.min_noise_std = module_config_dict.get('min_noise_std', None)
        self.min_mean_noise_std = module_config_dict.get('min_mean_noise_std', None)
        self.max_noise_std = module_config_dict.get('max_noise_std', None)  # 可选 std 上限,默认关(不影响现有训练);weak 扩展可设如 1.0 防膨胀
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        std = self.std
        if self.min_noise_std:
            std = torch.clamp(std, min=self.min_noise_std)
        elif self.min_mean_noise_std:
            current_mean = std.mean()
            if current_mean < self.min_mean_noise_std:
                std = std * (self.min_mean_noise_std / (current_mean + 1e-6))
        if self.max_noise_std is not None:
            std = torch.clamp(std, max=self.max_noise_std)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')



class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

# Deprecated: TODO: Let Wenli Fix this
class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                network_dict,
                network_load_dict,
                num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0)
        
    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution
    
    def get_actions_log_prob(self, actions):
        raise NotImplementedError
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution