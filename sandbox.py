from table_methods import RatingTable, PlacesFactor, MoneyPool
from tickets import TicketBase, CommonTicket, RareTicket, EpicTicket, \
    LegendaryTicket

cash_amount = 1
places_factor = PlacesFactor(
    [.6, .2, .10, .07, .03]
)

money_pool = MoneyPool(value=cash_amount)

sample_winners = [
    CommonTicket(),
    EpicTicket(),
    CommonTicket(),
    EpicTicket(),
    CommonTicket()
]

rt = RatingTable(places_factor=places_factor,
                 money_pool=money_pool)

result = rt.result_table(winners=sample_winners)

print(result)
print(result[['share',
              'rest_share']].sum())

s = result['place']
print(s.to_numpy())