import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
X = pd.read_csv('gild.csv')
X = X.to_numpy()
X = X[:,1:]
x = X[:,5]
Y = pd.read_csv('gild_pred.csv')
Y = Y.to_numpy()
Y = Y[:,1:]
y = Y[:,5]
d = np.concatenate((X, Y))
d = d[:,5]
test_res = pd.read_csv('test_res_2.csv')
test_res = test_res.to_numpy()
test_res = test_res[:,1]
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:red'
ax1.set_ylabel('Data', color=color, size=14)
ax1.plot(d, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_ylim(0,500000000)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CoDisp', color=color, size=14)
ax2.plot(pd.Series(test_res).sort_index(), color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.grid('off')

plt.title('data with anomaly (red) and anomaly score (blue)', size=14)
plt.show()