import matplotlib
import pandas
import pandas as pd
import requests
from matplotlib import pyplot as plt

rc_params = {
    "text.usetex": True,
    "font.size": 18,
    "font.family": "sans-serif",
    "text.latex.preamble": r'\usepackage[T1]{fontenc}'
}
matplotlib.rcParams.update(rc_params)

file = "votesMulticlass.txt"
url = "https://gaia.fdi.ucm.es/files/research/xai/xai-experiments/experiment1/" + file

download = False

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

# set Dark2 colormap for all plots

fig, ax = plt.subplots(figsize=(10, 6))

relative_freq_df.plot(kind='bar', stacked=True, ax=ax, color=['darkred', 'darkblue', 'darkgreen', 'purple'])
ax.tick_params(axis='x', which='both', direction='in', length=0, pad=15)
ax.tick_params(axis='y', which='minor', direction='in', length=0, pad=15)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
legend_map = {'saliency': 'Saliency', 'two_class_gradcam': 'GradCAM 2 clases', 'three_class_gradcam': 'GradCAM 3 clases', 'vargrad': 'VarGrad'}
ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
plt.xlabel('Dataset', labelpad=10)
plt.ylabel('Frecuencia relativa', labelpad=10)
plt.title('Frecuencia relativa de métodos de explicación visual', pad=20)
plt.xticks(rotation=0)  # Rotate x labels
labels = [f'{i:.0%}' for i in relative_freq_df.to_numpy().flatten(order='F')]

for i, patch in enumerate(ax.patches):
    fontsize = 14
    x, y = patch.get_xy()
    x += patch.get_width() / 2
    y += patch.get_height() / 2
    # patch color
    color = patch.get_facecolor()
    annotation = ax.annotate(labels[i][:-1] + '\%', (x, y), ha='center', va='center', fontsize=fontsize, color='white')
    if annotation.get_window_extent().height > patch.get_window_extent().height:
        print(annotation.get_window_extent(), '>', patch.get_window_extent().height)
        annotation.set_bbox(dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=1))
        #annotation.set_color('black')
        # tight background frame
        #annotation.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none'))
        #annotation.set_backgroundcolor('white')

        #new_fontsize = fontsize * patch.get_window_extent().height / annotation.get_window_extent().height
        #annotation.remove()
        #annotation = ax.annotate(labels[i][:-1] + '\%', (x, y), ha='center', va='center', fontsize=new_fontsize, color='white')

        #annotation.set_fontsize(new_fontsize)
        #annotation.remove()
        #annotation = ax.annotate(labels[i][:-1] + '\%', (x, y), ha='center', va='center', fontsize=new_fontsize, color='white')

plt.legend(loc='upper center', ncol=4, frameon=False, labels=[legend_map[label] for label in ax.get_legend_handles_labels()[1]], columnspacing=0.5)
#plt.legend()
plt.tight_layout()
plt.savefig('./experiment1.pdf')
#plt.show()

# distribution by role
df_role = df.groupby(['role', 'method']).size().reset_index(name='count')
print(df_role)

pivot_role_df = df_role.pivot(index='role', columns='method', values='count')
relative_freq_role_df = pivot_role_df.div(pivot_role_df.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

relative_freq_role_df.plot(kind='bar', stacked=True, ax=ax, color=['darkred', 'darkblue', 'darkgreen', 'purple'])
ax.tick_params(axis='x', which='both', direction='in', length=0, pad=15)
ax.tick_params(axis='y', which='minor', direction='in', length=0, pad=15)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
ticks_map = {'Orthopedics expert': 'Experto en ortopedia', 'Other': 'Otro', 'Student': 'Estudiante'}
legend_map = {'saliency': 'Saliency', 'two_class_gradcam': 'GradCAM 2 clases', 'three_class_gradcam': 'GradCAM 3 clases', 'vargrad': 'VarGrad'}
print(ax.get_xticklabels())
ax.set_xticklabels([ticks_map[tick.get_text()] for tick in ax.get_xticklabels()])
plt.xlabel('Rol', labelpad=10)
plt.ylabel('Frecuencia relativa', labelpad=10)
plt.title('Frecuencia relativa de métodos de explicación visual por rol', pad=20)
plt.xticks(rotation=0)  # Rotate x labels
for i, patch in enumerate(ax.patches):
    fontsize = 14
    x, y = patch.get_xy()
    x += patch.get_width() / 2
    y += patch.get_height() / 2
    # patch color
    color = patch.get_facecolor()
    annotation = ax.annotate(labels[i][:-1] + '\%', (x, y), ha='center', va='center', fontsize=fontsize, color='white')
    if annotation.get_window_extent().height > patch.get_window_extent().height:
        annotation.set_bbox(dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=1))
plt.legend(loc='upper center', ncol=4, frameon=False, labels=[legend_map[label] for label in ax.get_legend_handles_labels()[1]], columnspacing=0.5)
plt.tight_layout()
plt.savefig('./experiment1_role.pdf')
plt.show()
