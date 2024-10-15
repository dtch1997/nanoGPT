
import torch 
import pathlib
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

save_dir = pathlib.Path('kl_div')

# load all files in the directory
kl_divs = {}
for file in save_dir.iterdir():
    if file.is_file():
        kl_div = torch.load(file)
        kl_divs[file.stem] = kl_div

# Make dataframe
rows = []
for name, values in kl_divs.items():
    for layer, value in enumerate(values):
        rows.append({'name': name, 'layer': layer, 'kl_div': value.item()})
df = pd.DataFrame(rows)

# Plot the kl divs
seaborn.set_theme()
plt.figure(figsize=(10, 5))
seaborn.barplot(data=df, x='layer', y='kl_div', hue='name')