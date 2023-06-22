import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Arquivo Csv/Pokemaos.csv", header= None)
pd.DataFrame(data)


media = np.mean(data, axis= 0)
variancia = np.var(data, axis= 0)
maximo = np.max(data, axis= 0)
minimo = np.min(data, axis= 0)

print(np.round(np.corrcoef(data.transpose()), 2))
np.savetxt("correlações.csv", np.round(np.corrcoef(data.transpose()), 2), delimiter= ",", fmt= "%.2f")

plt.hist(data[:][0])
plt.hist(data[:][1])
plt.hist(data[:][2])
plt.hist(data[:][3])
plt.hist(data[:][4])

np.savetxt("correlações2.csv", np.round(np.corrcoef(data[:23].transpose()), 2), delimiter= ",", fmt= "%.2f")
np.savetxt("correlações3.csv", np.round(np.corrcoef(data[23:].transpose()), 2), delimiter= ",", fmt= "%.2f")

