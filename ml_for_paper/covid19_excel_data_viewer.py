import numpy as np
import pandas as pd
import os
from os import walk

mypath = '../ml_outputs'

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

# AioaCnetct
# Aak_aaiLus
# AaaaAkna_a

for filename in filenames:
    if 'AioaCnetct' in filename:
        df_data = pd.read_excel(mypath+'/'+filename)
        print(f"AioaCnetct\t{df_data['X Variables'][0]}\t{df_data['avg_score'][len(df_data)-1]}")


