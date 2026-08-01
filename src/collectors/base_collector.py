class BaseCollector(ABC):
    """
    Abstract interface for opportunity collectors.

    Every collector (CORDIS, OpenAlex, etc.)
    must implement the collect() method.
    """

    @abstractmethod
    def collect(self):
        """
        Collect opportunities from an external source.

        Returns:
            list[Opportunity]
        """
        pass