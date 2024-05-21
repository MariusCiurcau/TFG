import os

import numpy as np
import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy import stats

from Scripts.utils import read_label

file = "votesMulticlass.txt"
url = "https://gaia.fdi.ucm.es/files/research/xai/xai-experiments/experiment2/" + file
download = False

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

if download:
    download_file(url, file)

df = pd.read_csv(file, delimiter=';', names=["date", "ip", "knowledge", "role", "image", "utility", "feedback"])

df['utility'] = df.utility.astype(int)

role_stats = df.groupby('role')['utility'].agg(['mean', 'min', 'max', 'std', 'count'])
role_stats['sem'] = role_stats['std'] / np.sqrt(role_stats['count'])
role_stats['ci'] = role_stats.apply(lambda row: tuple(round(x, 3) for x in stats.norm.interval(0.95, loc=row['mean'], scale=row['sem'])), axis=1)
#pivot_table = pd.DataFrame({'utility': average_utility})

# round all columns to 2 decimals and count column to int
role_stats = role_stats.round(3)
role_stats['count'] = role_stats['count'].astype(int)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(role_stats[['mean', 'ci', 'count']])