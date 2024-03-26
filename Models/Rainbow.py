import numpy as np
from pfrl.agents import DQN, CategoricalDoubleDQN, CategoricalDQN
from pfrl import replay_buffers, utils, explorers
from pfrl.q_functions import DiscreteActionValueHead, DistributionalDuelingDQN, \
    DistributionalFCStateQFunctionWithDiscreteAction
from torch import nn, optim


class RBWrapper:
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
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            DistributionalFCStateQFunctionWithDiscreteAction(64, 2, 100, -10, 10, n_hidden_channels=16, n_hidden_layers=2),
        )
        print(self, (sum([len(x) for x in self.model.parameters()])))

        self.opt = optim.Adam(self.model.parameters())
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            10 ** 6,
            alpha=0.5,
            beta0=0.4,
            betasteps=250,
            num_steps=3,
            normalize_by_max="memory",
        )
        decay_timestep = 200
        explr_start = 1.0
        explr_end = 0.0
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            explr_start,
            explr_end,
            decay_timestep,
            lambda: np.random.randint(2)
        )
        # Agent = CategoricalDoubleDQN
        agent = CategoricalDQN(
            self.model,
            self.opt,
            rbuf,
            gamma=0.99,
            explorer=self.explorer,
            minibatch_size=2,
            replay_start_size=10**6,
            target_update_interval=32000,
            batch_accumulator="mean",
            phi=lambda x: np.asarray(x, dtype=np.float32),
        )
        return agent

