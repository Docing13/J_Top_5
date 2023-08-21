import gymnasium as gym
import tianshou as ts
import torch
import numpy as np
from tianshou.data import Batch
from torch import nn
import time

np.random.seed(42)
torch.manual_seed(42)
class Net(nn.Module):
    def __init__(self, state_size:int, action_size:int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, action_size)
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

#creation
env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=5e-3)


policy = ts.policy.DQNPolicy(net,
                             optim,
                             discount_factor=0.95,
                             estimation_step=1,
                             target_update_freq=0,
                             is_double=False)
policy.set_eps(0.1)
buffer = ts.data.VectorReplayBuffer(20000, 1)

s, _ = env.reset()
s_, r, term, trunc, info = env.step(np.random.randint(0,1))
done = np.logical_or(term, trunc)
data = Batch(
    obs={},
    act={},
    rew={},
    done={},
    obs_next={},
    terminated={},
    truncated={},
)

data.update(
            obs=s,
            act=0,
            rew=r,
            done=False,
            obs_next=s_,
            terminated=False,
            truncated=False)

print(data)
buffer.add(data,buffer_ids=None)
print(buffer)
# buffer.

# training
# for i in range(500):
#     losses = policy.update(200, buffer)
#     print(losses)

# # save
# torch.save(policy.state_dict(), 'dqn.pth')

# load
# policy.load_state_dict(torch.load('dqn.pth'))
# # eval
# policy.eval()

# collector = ts.data.Collector(policy, env)
# collector.collect(n_episode=3, render=1 / 35)

# with torch.no_grad():
#     while True:
#         env = gym.make('CartPole-v1',
#                        render_mode="human"
#                        )
#         s, _ = env.reset()
#         d = False
#         while not d:
#             # {'obs': s}
#             # env.render()
#             action = policy.forward(ts.data.Batch(obs=np.array([s]),
#                                                   info=None)).act[0]
#             print((action))
#
#             time.sleep(1/50)
#             s, r, term, trunc, _ = env.step(action)
#             d = term
#
#         env.close()