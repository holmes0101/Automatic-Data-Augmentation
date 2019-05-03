# TODO: rethink dataset interface

from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_input_shape():
        pass

    @abstractmethod
    def get_batch(self, data = "Train", policy = None):
        pass
