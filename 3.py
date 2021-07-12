import pandas as pd

df = pd.read_csv('egauge_maxtrix_201501_work.csv', header=None)
df = df.dropna()
b =1
print(df.head())
dates = pd.DataFrame(df[1])
dates.columns = ['date']
dates_list = dates.date.unique()
print(dates_list)

