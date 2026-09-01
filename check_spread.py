import pandas as pd
df = pd.read_parquet('output/data/gold_silver_m10.parquet')
df['session'] = df.index.hour.map(lambda h: 'ASIA' if h < 7 else ('LONDON' if h < 12 else ('LONDON_NEW_YORK_OVERLAP' if h < 16 else 'NEW_YORK')))
for s in df['session'].unique():
    sd = df[df['session']==s]
    print(f'{s}: gold_spread mean={sd["gold_spread"].mean():.2f}, silver_spread mean={sd["silver_spread"].mean():.2f}')