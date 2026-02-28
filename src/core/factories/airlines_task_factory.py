from src.core.data_managers.airlines_data_manager import AirlinesDataManager
from src.core.factories.factories import TaskFactory
from src.core.ml_strategies.classification_ml_strategy import ClassificationMLStrategy
from src.core.strategies.classification_strategy import ClassificationStrategy

class AirlinesTaskFactory(TaskFactory):
    def create_strategy(self): return ClassificationStrategy()
    def create_data_manager(self, strategy): return AirlinesDataManager(strategy)
    def create_ml_strategy(self): return ClassificationMLStrategy()

