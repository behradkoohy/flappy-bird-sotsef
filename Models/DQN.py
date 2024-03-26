import numpy as np
from pfrl.action_value import DistributionalDiscreteActionValue
from pfrl.agents import DQN, CategoricalDoubleDQN
from pfrl import replay_buffers, utils, explorers
from pfrl.q_functions import DiscreteActionValueHead, DistributionalFCStateQFunctionWithDiscreteAction
from torch import nn, optim


class DQNWrapper:
    def __new__(self, obs_size, env, n_epochs, n_buf=50000, *args, **kwargs):
        print("obs size", obs_size)
        # obs_space = obs_size[0]
        # act_space = obs_act[1]

        # def conv2d_size_out(size, kernel_size=3, stride=1):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
        #
        # h = conv2d_size_out(obs_size[1])
        # w = conv2d_size_out(obs_size[2])

        self.model = nn.Sequential(
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
            nn.Linear(256, 64),
            nn.ReLU6(),
            DiscreteActionValueHead(),
            # DistributionalFCStateQFunctionWithDiscreteAction(64, 2, 100, -10, 10, n_hidden_channels=16,
                                                             # n_hidden_layers=2)
        )
        print(self, (sum([len(x) for x in self.model.parameters()])))

        self.opt = optim.Adam(self.model.parameters())

        # self.replay_buffer = replay_buffers.ReplayBuffer(n_buf)
        # betasteps = timesteps / 50
        replay_size = 10 ** 18
        replay_alpha0 = 0.6
        replay_beta0 = 0.4
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            # n_buf, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=1
            replay_size,
            alpha=replay_alpha0,
            beta0=replay_beta0,
            # betasteps=betasteps,
            num_steps=1000,
        )

        # decay_timestep = int(10000 * n_epochs * 0.9)
        decay_timestep = 1e4
        explr_start = 1.0
        explr_end = 0.01
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            explr_start,
            explr_end,
            # 0.99,
            decay_timestep,
            lambda: np.random.randint(2),
            # 1.0,
            # 0.1,
            # 1000000,
            # lambda: np.random.randint(3),
        )
        return CategoricalDoubleDQN(
            self.model,
            self.opt,
            self.replay_buffer,
            0.99,
            self.explorer,
            minibatch_size=512,
            replay_start_size=1024,
            target_update_interval=512,
            phi=lambda x: np.asarray(x, dtype=np.float32)
        )

