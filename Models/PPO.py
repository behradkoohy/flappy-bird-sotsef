import numpy as np
import pfrl.policies
import torch
from torch import nn
from pfrl.action_value import DistributionalDiscreteActionValue
from pfrl.agents import PPO


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer

class PPOWrapper:
    def __new__(self, obs_size, env, n_epochs, n_buf=50000, *args, **kwargs):

        policy = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU6(),
            nn.Linear(128, 256),
            nn.ReLU6(),
            nn.Linear(256, 512),
            nn.ReLU6(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Linear(256, 1),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=1,
                var_type='diagonal',
                # var_func=lambda x: torch.exp(2*x),
                # var_param_init=0
            )
        )
        vf = torch.nn.Sequential(
            nn.Linear(obs_size,64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        model = pfrl.nn.Branched(policy, vf)

        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        # print(policy.dtype, vf.dtype)
        agent = PPO(
            model,
            opt,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            # gamma=0.99
        )
        return agent