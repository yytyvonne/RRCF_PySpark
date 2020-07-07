import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
df = pd.read_csv('historical_stock_prices.csv')
df['date'] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df['year'] = df.date.apply(lambda x: x.year)
df = df[df.year.isin(range(2010,2014))]
#df = df[df['ticker']=='AIG']
df = df[df['ticker']=='GILD']
df = df.drop(['year','ticker','date'], axis = 1)
df = df.reset_index(drop=True)

plt.plot(df[['volume']])
plt.show()

mean = df[['volume']].mean()[0]
std = df[['volume']].std()[0]

rand = np.random.choice(range(len(df)),10)
print(rand)
df.loc[rand,'volume'] = mean + abs(np.random.normal(0,1,10)) * std * 10

plt.plot(df[['volume']])
plt.show()

df.to_csv('gild.csv', index = True)

df = pd.read_csv('historical_stock_prices.csv')
df['date'] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df['year'] = df.date.apply(lambda x: x.year)
df = df[df.year.isin(range(2014,2016))]
#df = df[df['ticker']=='AIG']
df = df[df['ticker']=='GILD']
df = df.drop(['year','ticker','date'], axis = 1)
df = df.reset_index(drop=True)

plt.plot(df[['volume']])
plt.show()

mean = df[['volume']].mean()[0]
std = df[['volume']].std()[0]

rand = np.random.choice(range(len(df)),8)
print(rand)
df.loc[rand,'volume'] = mean + abs(np.random.normal(0,1,8)) * std * 10

plt.plot(df[['volume']])
plt.show()

df.to_csv('gild_pred.csv', index = True)

