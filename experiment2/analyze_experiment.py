import os

import numpy as np
import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib

from Scripts.utils import read_label
rc_params = {
    "text.usetex": True,
    "font.size": 18,
    "font.family": "sans-serif",
    "text.latex.preamble": r'\usepackage[T1]{fontenc}'
}
matplotlib.rcParams.update(rc_params)

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

def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.median()

def q3(x):
    return x.quantile(0.75)

role_stats = df.groupby('role')['utility'].agg(['mean', 'min', 'max', 'std', 'count', q1, q2, q3])
role_stats['sem'] = role_stats['std'] / np.sqrt(role_stats['count'])
role_stats['ci'] = role_stats.apply(lambda row: tuple(round(x, 3) for x in stats.norm.interval(0.95, loc=row['mean'], scale=row['sem'])), axis=1)
#pivot_table = pd.DataFrame({'utility': average_utility})

# round all columns to 2 decimals and count column to int
role_stats = role_stats.round(3)
role_stats['count'] = role_stats['count'].astype(int)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(role_stats[['mean', 'ci', 'count', 'std', 'sem', 'min', 'max', 'q1', 'q2', 'q3']])

box_data = []

for role in role_stats.index:
    q1 = role_stats.at[role, 'q1']
    q2 = role_stats.at[role, 'q2']
    q3 = role_stats.at[role, 'q3']
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    box_data.append({
        'label': role,
        'med': q2,
        'q1': q1,
        'q3': q3,
        'whislo': lower_whisker,
        'whishi': upper_whisker,
        'fliers': []
    })

fig, ax = plt.subplots(figsize=(10, 6))
ax.bxp(box_data, showfliers=False, showmeans=False)

# Añadir medias y medianas
for i, role in enumerate(role_stats.index, start=1):
    mean = role_stats.at[role, 'mean']
    median = role_stats.at[role, 'q2']
    ax.plot(i, mean, 'D', markersize=8, color='blue', label='Mean' if i == 1 else "")
    ax.plot(i, median, 'o', markersize=8, color='red', label='Median' if i == 1 else "")

# Añadir leyenda
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.set_xlabel('Role')
ax.set_ylabel('Values')
ax.set_title('Boxplot by Role with Means and Medians')
plt.xticks(range(1, len(role_stats.index) + 1), role_stats.index)
plt.show()