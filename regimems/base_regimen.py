from abc import abstractmethod
from typing import Dict

class Regimen(abs):
    def __init__(self, *args, **kwargs):
        pass

    def init(self, args:Dict):
        for k, v in args.items():
            setattr(self, k, v)

    @abstractmethod
    def train(self)->Dict:
        pass

    @abstractmethod
    def test(self)->Dict:
        pass

