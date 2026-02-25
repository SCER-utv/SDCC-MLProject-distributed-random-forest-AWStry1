import os
# [MODIFICA] Rimossi gli import di pandas e numpy, non servono più qui!
from src.core.data_managers.data_managers import BaseDataManager

class HiggsDataManager(BaseDataManager):

    def get_target_column(self):
        return 'Label'  