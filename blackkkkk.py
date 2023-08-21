import numpy as np

from agents import GreedyCore, SensorDefault, TicketActuator

#
# a = {
#     0:[0,1,2,3],
#     1:[0,21],
#     2:[0,20],
#     3:[0,5,11,5]
# }
#
# means = [(i,np.mean(np.array(j)))for i,j in zip(a.keys(),a.values())]
# print(means)
# action = sorted(means,key=lambda x:x[1],reverse=True)[0][0]
# print(action)

# print(np.random.binomial(1,0.1))
sen = SensorDefault
act = TicketActuator
core = GreedyCore()
core.setup(sen,act)
a = core.act(0,2)
print(a)