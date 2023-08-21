from __future__ import annotations

from typing import Iterable, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType

from table_methods import RatingTable, PlacesFactor, MoneyPool
from scipy.special import softmax
from tickets import TicketBase


class SimpleRatingTableEnv(gym.Env):

    def __init__(self,
                 places_factor: PlacesFactor,
                 players_count: int,
                 game_scale: float = 0.1):
        self.players_count = players_count
        self.places_factor = places_factor

        # aka std for sampling
        self.game_scale = game_scale

        self.top_players_count = len(self.places_factor.factors)
        self.money_pool = MoneyPool(0)

        self.table = RatingTable(places_factor=self.places_factor,
                                 money_pool=self.money_pool)

    @staticmethod
    def _tickets(actions) -> list[TicketBase]:
        tickets = [action[0] for action in actions]
        return tickets

    @staticmethod
    def _total_cash(tickets: Iterable[TicketBase]) -> float:
        cash = sum([ticket.price for ticket in tickets])
        return cash

    @staticmethod
    def _skills(actions) -> np.ndarray:
        skills = [action[1] for action in actions]
        skills = np.array(skills)
        return skills

    def step(self,
             actions: Iterable[Iterable[TicketBase, float]]):
        '''

        a: [card,skill]

        action is selelcted card and skill
         action is selected cards by players
         [cardt1 cardt2 cardt2 cardt3]

        '''

        # todo figure out about state space...
        # todo add logging about places and total received cash(not here but upper)
        session_tickets = self._tickets(actions)

        session_cash = self._total_cash(session_tickets)
        self.money_pool.value = session_cash

        session_skills = self._skills(actions)

        session_idxs = np.arange(0, len(session_tickets))

        # simulation of playing
        scores = np.random.normal(session_skills,
                                  scale=self.game_scale)

        rating = sorted(zip(session_idxs,
                            session_tickets,
                            scores,
                            session_skills),
                        key=lambda x: x[2],
                        reverse=True)

        # expanding rating with places
        places = [place for place in range(1, len(session_tickets) + 1)]
        rating = [(*r_item, place) for r_item, place in zip(rating, places)]

        favourites_tickets = [f[1] for f in rating[:self.top_players_count]]
        top_summary = self.table.result_table(favourites_tickets)

        profits = np.zeros(len(session_tickets))
        top_profits = top_summary['rest_share'].to_numpy()
        profits[:top_profits.size] = top_profits

        tickets_costs = np.array([r_item[1].price for r_item in rating])

        r = profits - tickets_costs

        # ses id , scores, skills, places, money_pool, r
        final_summary = np.array([
            [r[0] for r in rating],  # ses id
            [r[2] for r in rating],  # score
            [r[3] for r in rating],  # skill
            [r[4] for r in rating],  # place
            np.full(len(session_tickets), session_cash),  # cash
            r,
         ]).T

        sorted_id_final_summary = final_summary[final_summary[:, 0].argsort()]

        s = sorted_id_final_summary[:, :-1]
        r = sorted_id_final_summary[:, -1].T

        return s, r

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        self.money_pool.value = 0
        shape = (self.players_count, 5)

        return np.zeros(shape)
