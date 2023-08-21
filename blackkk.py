import numpy as np

from agents import SensorDefault, RandomCore, DefaultActuator, ExponentialSkill, AgentFactory, TicketActuator
from envs.envs import SimpleRatingTableEnv
from table_methods import PlacesFactor
from tickets import LegendaryTicket, CommonTicket, SuperRareTicket

players_count = 5
config = [{
    'sensor': SensorDefault,
    'sensor_params': {'places_amount':players_count,'cash_max':50},
    'core': RandomCore,
    'core_params': {},
    'actuator': TicketActuator,
    'actuator_params': {},
    'skill': ExponentialSkill,
    'skill_params': {},
    'count': players_count
}, ]

factory = AgentFactory(config=config)


places_factor = PlacesFactor(
    [.6, .2, .10, .07, .03]
)
env = SimpleRatingTableEnv(places_factor=places_factor,
                           players_count=players_count)
zero_s = env.reset()
a = factory.act(zero_s,np.zeros(zero_s.shape[0]),False)
for i in range(3):


    s, r = env.step(a)

    print(r)
    print('='*8)
    print(s)
    a = factory.act(s,r,False)
    print('~~'*5)
    print(a)

agent,s,a,r = factory.logger[2]
print(r)