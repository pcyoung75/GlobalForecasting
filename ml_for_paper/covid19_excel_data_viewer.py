import numpy as np
import pandas as pd
import os
from os import walk

mypath = '../ml_outputs'

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

for filename in filenames:
    if 'CnetctFoiaMieMcia_isuiNvd_e esy' in filename:
        df_data = pd.read_excel(mypath+'/'+filename)
        print(f"CnetctFoiaMieMcia_isuiNvd_e esy\t{df_data['X Variables'][0]}\t{df_data['avg_score'][len(df_data)-1]}")


