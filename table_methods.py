from dataclasses import dataclass
from typing import Iterable,Tuple,Type,Union,List
from tickets import TicketBase
import pandas as pd

# todo alternative we can create custom place factors percents depends on card type:
#  for example first player with legendary pretends on the 90% of money amount (think about this alternative)


@dataclass
class PlacesFactor:
    factors: list[float]


@dataclass
class MoneyPool:
    value: float



class RatingTable:

    def __init__(self,
                 places_factor: PlacesFactor,
                 money_pool: MoneyPool):

        self.places_factor = places_factor
        self.money_pool = money_pool

    def result_table(self,
                     winners: Union[Tuple[Type[TicketBase]],
                                    List[Type[TicketBase]]]) -> pd.DataFrame:

        if len(winners) != len(self.places_factor.factors):
            raise Exception(f"Wrong winner count: {len(winners)}, "
                            f"when {len(self.places_factor.factors)}"
                            f" was expected")

        places = []
        places_factors = []
        winners_strs = []
        winners_percent = []
        money_parts = []

        for idx, _ in enumerate(zip(winners, self.places_factor.factors)):

            winner, factor = _

            places_factors.append(factor)

            win_money = self.money_pool.value * factor * winner.percent

            places.append(idx + 1)
            winners_strs.append(winner.__class__.__name__)
            winners_percent.append(winner.percent)
            money_parts.append(win_money)

        summary_table = pd.DataFrame({
            'place': places,
            'place_factor': places_factors,
            'player': winners_strs,
            'ticket scale': winners_percent,
            'share': money_parts
        })

        sum_total = summary_table['share'].sum()
        diff = self.money_pool.value - sum_total
        rest = [i * diff for i in self.places_factor.factors]

        summary_table['rest_share'] = summary_table['share'] + rest
        return summary_table
