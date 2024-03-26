import numpy as np
import pfrl.policies
import torch
from pfrl.policies import SoftmaxCategoricalHead
from torch import nn
from pfrl.action_value import DistributionalDiscreteActionValue
from pfrl.agents import A2C

class A2CWrapper:
    def __new__(self, obs_size, env, n_epochs, n_buf=50000, *args, **kwargs):
        model = nn.Sequential(
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
            pfrl.nn.Branched(
                nn.Sequential(
                    nn.Linear(256, 2),
                    SoftmaxCategoricalHead()
                ),
                nn.Linear(256, 1)
            )
        )
        optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
            model.parameters(),
            lr=7e-4
        )
        agent = A2C(
                model,
                optimizer,
            gamma=0.99,
            num_processes=1,
            phi=lambda x: np.asarray(x, dtype=np.float32),
        )
        return agent


