from abc import ABC, abstractmethod


class ProfileBuilder(ABC):

    @abstractmethod
    def build(self, researcher):
        pass