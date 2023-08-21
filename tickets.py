
from abc import ABC
from dataclasses import dataclass


# @dataclass
class TicketBase(ABC):
    percent: float
    price: float


class CommonTicket(TicketBase):
    percent = 0.5
    price = 0.1


class UncommonTicket(TicketBase):
    percent = 0.7
    price = 0.2


class RareTicket(TicketBase):
    percent = 0.75
    price = 0.3


class SuperRareTicket(TicketBase):
    percent = 0.8
    price = 0.4


class EpicTicket(TicketBase):
    percent = 0.85
    price = 0.5


class LegendaryTicket(TicketBase):
    percent = 0.9
    price = 0.6

