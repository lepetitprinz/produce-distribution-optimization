import numpy as np
import pandas as pd

from Dao import Dao
from Preprocessing import Preprocessing


# MARKET_KOR = '가락'
# MARKET_ENG = 'garak'

MARKET_KOR = '대구'
MARKET_ENG = 'daegu'

def main():
    dao: Dao = Dao.instance()

    # Query Data
    df_prdt: pd.DataFrame = dao.get_prdt_data()

    preprocessing = Preprocessing()
    df_prdt_remap = preprocessing.get_prdt_remap(data=df_prdt)
    df_prdt_remap.to_csv('dist_prep.csv')

    print("Debug Point")



main()