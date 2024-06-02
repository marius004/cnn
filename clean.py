import pandas as pd

df = pd.read_csv('predictions.csv')

df['image_id'] = df['image_id'].str.replace('.png', '')
df.to_csv('predictions.csv', index=False)
