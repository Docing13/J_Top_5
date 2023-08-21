from collections import deque, namedtuple
import random
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self,
                 s_size: int,
                 a_size: int,
                 linear_dims: list[int],
                 act_func=nn.Sigmoid,
                 a_func=None):

        super().__init__()

        self.s_size = s_size
        self.a_size = a_size
        self.act_func = act_func

        if a_func:
            self.a_func = a_func()
        else:
            self.a_func = a_func

        if linear_dims:
            layers = [nn.Linear(self.s_size,
                                linear_dims[0]),
                      self.act_func()]

            for idx in range(len(linear_dims) - 1):
                layers.append(nn.Linear(linear_dims[idx],
                                        linear_dims[idx + 1]))
                layers.append(self.act_func())

            layers.append(nn.Linear(linear_dims[-1],
                                    self.a_size))
        else:
            layers = [
                nn.Linear(self.s_size, self.a_size)
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)

        if self.a_func:
            x = self.a_func(x)

        return x


