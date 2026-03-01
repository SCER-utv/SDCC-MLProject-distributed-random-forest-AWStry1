import os
from src.core.data_managers.data_managers import BaseDataManager

class AirlinesDataManager(BaseDataManager):
    def get_target_column(self):

        return 'Label'
