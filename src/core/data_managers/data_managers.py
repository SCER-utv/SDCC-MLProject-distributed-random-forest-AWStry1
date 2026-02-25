import os
import sys
from abc import abstractmethod, ABC
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class BaseDataManager(ABC):

    def __init__(self, strategy):
        # Salva la strategy come attributo dell'istanza
        self.strategy = strategy

    # L'unica informazione che serve a runtime è il nome della label
    @abstractmethod
    def get_target_column(self): pass