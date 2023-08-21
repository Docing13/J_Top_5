from abc import ABC, abstractmethod
import random
import numpy as np
from numpy import ndarray
from typing import Optional, Any, Union
import tianshou as ts

from tickets import CommonTicket, RareTicket, SuperRareTicket, UncommonTicket, EpicTicket, LegendaryTicket

'''
https://coax.readthedocs.io/en/latest/
https://tianshou.readthedocs.io/en/master/
https://stable-baselines.readthedocs.io/en/master/
'''


# todo implement dummy, simple greedy, more complex but nn??

# ----------------------------------SKILLS--------------------------------------------
class SkillBase(ABC):
    def __init__(self):
        self._iteration = 0

    def __call__(self, *args, **kwargs) -> float:
        self._iteration += 1
        return self._skill

    @property
    @abstractmethod
    def _skill(self) -> float:
        pass


class RandomSkill(SkillBase):
    def _skill(self) -> float:
        return random.random()


class LinearSkill(SkillBase):
    def __init__(self,
                 factor: float = 1e-4,
                 bias: float = .0):
        super().__init__()

        self._factor = factor
        self._bias = bias

    def _skill(self) -> float:
        return self._iteration * self._factor + self._bias


class ExponentialSkill(SkillBase):

    def __init__(self,
                 skill_scale: float = 0.03,
                 skill_target: float = 0.9,
                 skill_limit: float = 1.0):
        super().__init__()

        self._skill_scale = skill_scale
        self._skill_target = skill_target
        self._skill_limit = skill_limit

    def _skill(self) -> ndarray:
        y = self._skill_target * (np.exp(self._iteration) /
                                  (1 + np.exp(self._iteration)))

        y = np.random.normal(y, self._skill_scale)

        y = np.clip(y, 0, self._skill_limit)

        return y


# ----------------------------------SENSORS------------------------------------------
class SensorBase(ABC):

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @abstractmethod
    def read(self, s: np.ndarray) -> np.ndarray:
        pass


# this basic sensor looks only for the agent activity
class SensorDefault(SensorBase):
    def __init__(self,
                 places_amount: int,
                 cash_max: float):
        self._places_amount = places_amount
        self._cash_max = cash_max

    @property
    def shape(self) -> tuple:
        return 4,

    def read(self, s: np.ndarray) -> np.ndarray:
        # s = [ses_id,score,skill,place,cash_total]

        if s.ndim != 1:
            raise Exception(f'Wrong state {s}'
                            f'with shape {s.shape}'
                            f' 1d array expected')
        # s = [score,skill,place,cash_total]
        s = s[1:]
        s[2] /= self._places_amount
        s[3] /= self._cash_max

        return s


# ----------------------------------ACTUATORS--------------------------------------------
class ActuatorBase(ABC):

    shape: Union[tuple, int]

    @abstractmethod
    def actuate(self, x: np.ndarray) -> Any:
        pass


class DefaultActuator(ActuatorBase):

    shape = 5

    def actuate(self, x: np.ndarray) -> Any:
        return x


class TicketActuator(ActuatorBase):
    _map = {
        0: CommonTicket(),
        1: UncommonTicket(),
        2: RareTicket(),
        3: SuperRareTicket(),
        4: EpicTicket(),
        5: LegendaryTicket(),
    }

    shape = 6

    def actuate(self, x: np.ndarray) -> Any:
        return self._map[x]


# --------------------------------CORES------------------------------------------


class AgentCoreBase(ABC):
    # todo maybe add train method
    def __init__(self):
        self._input_shape = None
        self._output_shape = None

    def setup(self,
              sensor: SensorBase,
              actuator: ActuatorBase) -> None:
        self._input_shape = sensor.shape
        self._output_shape = actuator.shape

        self._init_core()

    @abstractmethod
    def _init_core(self) -> Any:
        pass

    # todo maybe rename
    @abstractmethod
    def act(self, s, r) -> Union[np.ndarray, int]:
        pass


class RandomCore(AgentCoreBase):
    def __init__(self):
        super().__init__()

    def _init_core(self) -> Any:
        pass

    def act(self, s, r) -> Union[np.ndarray, int]:
        act = np.random.randint(0, self._output_shape)
        return act


class GreedyCore(AgentCoreBase):
    def __init__(self, explore_p: float = 0.1):

        super(GreedyCore, self).__init__()
        self.explore_p = explore_p
        self.act_rewards = {}
        self.prev_action = None

    def _init_core(self) -> Any:
        # print(self._output_shape)
        actions_len = self._output_shape
        for _ in range(actions_len):
            self.act_rewards[_] = [0]

    @property
    def _greedy(self):
        means = [(i, np.mean(np.array(j))) for i, j in zip(self.act_rewards.keys(),
                                                           self.act_rewards.values())]
        action = sorted(means, key=lambda x: x[1], reverse=True)[0][0]
        return action

    def act(self, s, r) -> Union[np.ndarray, int]:

        if self.prev_action:
            self.act_rewards[self.prev_action].append(r)

        if np.random.binomial(1, self.explore_p) or self.prev_action is None:
            action = np.random.randint(0, self._output_shape)

        else:
            action = self._greedy

        self.prev_action = action
        return action


class DQNCore(AgentCoreBase):
    pass


# ----------------------------------AGENTS--------------------------------------------
# class AgentBase(ABC):
class Agent:

    def __init__(self,
                 sensor: SensorBase,
                 core: AgentCoreBase,
                 actuator: ActuatorBase,
                 skill: SkillBase):
        self._sensor = sensor
        self._core = core
        self._actuator = actuator
        self._skill = skill()

        self._core.setup(sensor=self._sensor,
                         actuator=self._actuator)

    def __call__(self, s: np.ndarray, r: np.ndarray) -> Any:
        s = self._sensor.read(s)
        s = self._core.act(s, r)
        s = self._actuator.actuate(s)
        skill = self._skill()
        return s, skill


class FactoryLogger:
    def __init__(self, agents: list[Agent]):
        self._agents = agents
        self._s = [[] for _ in self._agents]
        self._a = [[] for _ in self._agents]
        self._r = [[] for _ in self._agents]

    @staticmethod
    def _log(buffer,
             values_like_arr):
        for buf_elem, value in zip(buffer, values_like_arr):
            buf_elem.append(value)

    def log_s(self, s):
        self._log(self._s, s)

    def log_r(self, r):
        self._log(self._r, r)

    def log_a(self, a):
        self._log(self._a, a)

    def __getitem__(self, item):
        s = self._s[item]
        a = self._a[item]
        r = self._r[item]
        agent = self._agents[item]

        return agent, s, a, r


class AgentFactory:
    def __init__(self, config: list[tuple]):
        self._config = config
        self._agents = []
        self._create_agents()
        self.logger = FactoryLogger(agents=self._agents)

    def _create_agents(self):

        for config_item in self._config:

            for _ in range(config_item['count']):
                sensor = config_item['sensor'](**config_item['sensor_params'])
                core = config_item['core'](**config_item['core_params'])
                actuator = config_item['actuator'](**config_item['actuator_params'])
                skill = config_item['skill'](**config_item['skill_params'])

                agent = Agent(sensor=sensor,
                              core=core,
                              actuator=actuator,
                              skill=skill)

                self._agents.append(agent)

    def act(self,
            s: np.ndarray,
            r: np.ndarray,
            d: bool):

        acts = []

        for s_, r_, agent in zip(s, r, self._agents):
            act = agent(s_, r_)
            acts.append(act)

        self.logger.log_s(s)
        self.logger.log_a(acts)
        self.logger.log_r(r)

        return acts
