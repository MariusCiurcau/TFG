import os

import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt

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

average_utility = df.groupby('role')['utility'].mean()
pivot_table = pd.DataFrame({'utility': average_utility})

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pivot_table)