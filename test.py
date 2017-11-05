import pandas as pd
df = pd.read_csv('datasets/cleaned_dataset.csv', sep=',', encoding='cp1252')
df_vw = df.loc[df['brand']=='volkswagen']
print(df_vw['model'].unique())