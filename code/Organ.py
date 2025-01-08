from abc import ABC, abstractmethod

class Organ(ABC):
    @abstractmethod
    def prepare_train():
        pass
    @abstractmethod
    def prepare_test():
        pass
    @abstractmethod
    def results():
        pass
