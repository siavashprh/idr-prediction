import abc


class Config(abc.ABC):
    @abc.abstractmethod
    def get_configuration(self):
        pass

    @abc.abstractmethod
    def get_summary(self):
        pass
