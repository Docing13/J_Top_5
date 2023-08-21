from agents import SensorDefault, DefaultActuator, ExponentialSkill, RandomCore
from envs.envs import SimpleRatingTableEnv
from table_methods import PlacesFactor
from tickets import LegendaryTicket,\
    SuperRareTicket,\
    CommonTicket


players_count = 4
places_factor = PlacesFactor(
    [.6, .2, .10, .07, .03]
)
env = SimpleRatingTableEnv(places_factor=places_factor,
                           players_count=players_count)
zero_s = env.reset()

for i in range(10):

    a = [[LegendaryTicket(), 0.6],
         [SuperRareTicket(), 0.4],
         [CommonTicket(), 0.8],
         [CommonTicket(), 0.5],
         [CommonTicket(), 0.1],
         [CommonTicket(), 0.1]]

    s, r = env.step(a)

    print(r)
    print('='*8)
    print(s)
    # print(s.ndim)
    # print(s[0].shape)
