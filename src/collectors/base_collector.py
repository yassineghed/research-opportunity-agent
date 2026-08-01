from abc import ABC, abstractmethod


class BaseCollector(ABC):
    """
    Abstract interface for all opportunity collectors.
    """

    @abstractmethod
    def collect(self):
        """
        Collect opportunities from an external source.

        Returns:
            list[Opportunity]
        """
        pass