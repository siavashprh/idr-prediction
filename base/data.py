import abc


class Data(abc.ABC):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    @abc.abstractmethod
    def extend(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def subset(self, indices) -> tuple:
        raise NotImplementedError


class TrainTestSplit(abc.ABC):
    @abc.abstractmethod
    def split(self, train_indices, test_indices):
        raise NotImplementedError
