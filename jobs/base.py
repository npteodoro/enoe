from abc import ABC, abstractmethod

class Job(ABC):
    @abstractmethod
    def execute(self, step, config, logger):
        pass

class Step(ABC):
    @abstractmethod
    def process(self):
        pass
