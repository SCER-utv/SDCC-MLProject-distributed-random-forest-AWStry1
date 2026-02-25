import os
# [MODIFICA] Rimossi gli import di pandas e numpy
from src.core.data_managers.data_managers import BaseDataManager

class TaxiDataManager(BaseDataManager):
    
    def get_target_column(self):
        return 'Label'  
