import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "./records/save_records.txt"


records = pd.read_csv(path, header=None)
#print(type(records.iloc[0,0]))
#print(records.iloc[0,0])
records[0] = records[0].str.replace("Scores: [", "")
records[99] = records[99].str.replace("]", "")
#records[100] = records[100].str.replace("Losses: [", "")
#records[199] = records[199].str.replace("]", "")
records = records.astype(float)
data = []
for i in range(records.shape[0]):
    for j in range(100):
        data.append(records.iloc[i,j])

x = [i+1 for i in range(records.shape[0]*100)]
plt.plot(range(100*(records.shape[0])), data, color="#FFA500", alpha=0.7)

window_size = 100
rolling_average = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
x_rolling = x[window_size//2 : -(window_size//2) + 1]

plt.plot(x_rolling, rolling_average, label='Average line', linestyle='--', color='green')
#plt.axhline(mean, color='red')
plt.show()
'''
mean = np.mean(losses)
plt.plot(range(100*records.shape[0]), losses)
plt.axhline(mean, color='red')
plt.show()
'''
