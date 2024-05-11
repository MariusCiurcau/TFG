import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt

file = "votesMulticlass.txt"
url = "https://gaia.fdi.ucm.es/files/research/xai/xai-experiments/experiment1/" + file

download = True

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

if download:
    download_file(url, file)



# Read the text file into a DataFrame
df = pd.read_csv(file, delimiter=';', names=["date", "ip", "knowledge", "role", "image", "method"])

df['dataset'] = df['image'].str.split('_').str[0]

df_group = df.groupby(['dataset', 'method']).size().reset_index(name='count')
print(df_group)

pivot_df = df_group.pivot(index='dataset', columns='method', values='count')
relative_freq_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)


relative_freq_df.plot(kind='bar', stacked=True)
plt.xlabel('Dataset')
plt.ylabel('Percentage')
plt.title('Percentage by Method and Dataset')
plt.xticks(rotation=0)  # Rotate x labels
plt.legend(title='Method')
plt.tight_layout()
plt.show()
