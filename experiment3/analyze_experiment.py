import os

import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from Scripts.utils import read_label

file = "votesMulticlass.txt"
url = "https://gaia.fdi.ucm.es/files/research/xai/xai-experiments/experiment3/" + file
IMG_DIR = '../Datasets/ULTIMAS/images'
LABEL_DIR = '../Datasets/ULTIMAS/labels'
download = True

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

if download:
    download_file(url, file)


# Read the text file into a DataFrame
df = pd.read_csv(file, delimiter=';', names=["date", "ip", "knowledge", "role", "image", "human_label"])


true_labels = {}
model_labels = {}
with open('./images/images.csv', 'r') as f:
    for line in f:
        image = line.split(';')[0]
        true_label = line.split(';')[1]
        model_label = line.split(';')[2].strip()
        true_labels[image] = true_label
        model_labels[image] = model_label

for i, row in df.iterrows():
    image = row['image']
    label = read_label(os.path.join(LABEL_DIR, os.path.splitext(image)[0] + '.txt'), 3)
    print(image, label)


df['true_label'] = df['image'].apply(lambda x: true_labels[x])

dtypes = {
    'human_label': int,
    'true_label': int
}
df = df.astype(dtypes)


df['p'] = (df['true_label'] == 1) | (df['true_label'] == 2)
df['n'] = (df['true_label'] == 0)
df['hp'] = (df['human_label'] == 1) | (df['human_label'] == 2)
df['hn'] = df['human_label'] == 0

df['correct'] = (((df['human_label'] == 1) | (df['human_label'] == 2)) & (df['p'])) | ((df['human_label'] == 0) & (df['n']))


df['tp'] = df['hp'] & df['p']
df['fp'] = df['hp'] & df['n']
df['tn'] = df['hn'] & df['n']
df['fn'] = df['hn'] & df['p']

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df[['human_label', 'true_label', 'correct', 'p', 'tp', 'fn']])
human_precision = (df.groupby('role')['tp'].sum() + df.groupby('role')['tn'].sum()) / (df.groupby('role')['tp'].sum() + df.groupby('role')['fp'].sum() + df.groupby('role')['tn'].sum() + df.groupby('role')['fn'].sum())
human_sensitivity = df.groupby('role')['tp'].sum() / (df.groupby('role')['tp'].sum() + df.groupby('role')['fn'].sum())
pivot_table = pd.DataFrame({'accuracy': human_precision, 'sensitivity': human_sensitivity})


model_tp = 0
model_fp = 0
model_tn = 0
model_fn = 0
for image, model_label in model_labels.items():
    model_label = int(model_label)
    true_label = int(true_labels[image])
    if (model_label == 1 or model_label == 2) and (true_label == 1 or true_label == 2):
        model_tp += 1
    if model_label == 0 and true_label == 0:
        model_tn += 1
    if (model_label == 1 or model_label == 2) and true_label == 0:
        model_fp += 1
    if model_label == 0 and (true_label == 1 or true_label == 2):
        model_fn += 1

model_sensitivity = model_tp / (model_tp + model_fn)
model_precision = (model_tp + model_tn) / (model_tp + model_fp + model_tn + model_fn)
pivot_table.loc['Model'] = [model_precision, model_sensitivity]
print(pivot_table)

def calculate_ci(count, nobs):
    return proportion_confint(count=count, nobs=nobs, alpha=0.05, method='normal')

# Calculate confidence intervals for accuracy and sensitivity for each role
role_ci_human = {}
for role, data in df.groupby('role'):
    accuracy_ci = calculate_ci(data['correct'].sum(), len(data))
    sensitivity_ci = calculate_ci(data['tp'].sum(), data['tp'].sum() + data['fn'].sum())
    role_ci_human[role] = {
        'Accuracy CI': accuracy_ci,
        'Sensitivity CI': sensitivity_ci
    }

model_accuracy_ci = calculate_ci(model_tp + model_tn, model_tp + model_fp + model_tn + model_fn)
model_sensitivity_ci = calculate_ci(model_tp, model_tp + model_fn)

# Convert the dictionaries to DataFrames
ci_df_human = pd.DataFrame.from_dict(role_ci_human, orient='index')
ci_df_model = pd.DataFrame({
    'Accuracy CI': [model_accuracy_ci],
    'Sensitivity CI': [model_sensitivity_ci]
}, index=['Model'])

# Combine the DataFrames
ci_df_combined = pd.concat([ci_df_human, ci_df_model])
pivot_table = pd.DataFrame({'accuracy': human_precision, 'sensitivity': human_sensitivity})
pivot_table.loc['Model'] = [(model_tp + model_tn) / (model_tp + model_fp + model_tn + model_fn), model_tp / (model_tp + model_fn)]
# Append model confidence intervals to the pivot table
pivot_table['accuracy CI'] = ci_df_combined['Accuracy CI']
pivot_table['sensitivity CI'] = ci_df_combined['Sensitivity CI']

# reordenamos columnas
pivot_table = pivot_table.reindex(columns=['accuracy', 'accuracy CI', 'sensitivity', 'sensitivity CI'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    print(pivot_table)