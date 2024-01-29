
import abc
from datasets import Dataset

class SplitDataset():
    @abc.abstractclassmethod
    def get_training() -> Dataset:
        raise NotImplementedError
    @abc.abstractclassmethod
    def get_eval() -> Dataset:
        raise NotImplementedError