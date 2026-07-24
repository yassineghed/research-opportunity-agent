from abc import ABC, abstractmethod


class OpportunityBuilder(ABC):

    @abstractmethod
    def build(self, opportunity):
        pass