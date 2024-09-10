import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import uniform, seed

#read data
data = pd.read_csv  (r'folder_path_enc_qam_csvfile')
#data.columns=['C']
data.columns = ['enc', 'qam']
data2 = data.drop('enc', axis=1).dropna()
#print(data2)

#drop duplicate values
df=data2.drop_duplicates()
df = df.sort_values('qam').reset_index(drop=True)
print('qam_w/o duplicate\n',df)

#real and imaginary separation
arr = np.complex128(df['qam'].str.replace('i', 'j').to_numpy())
df['D1']=arr
df = df.assign(x = arr.real, y = arr.imag)
df=df.drop(['qam','D1'], axis=1)
print('co-ordinate_points\n',df)

#randomized their position
#df = {"x": [1, 9, 5, 8, 2], "y": [5, 1, 5, 2, 7]}
lower, upper = 0.5, 1.8 # allows for data to be changed by -30% to 30%
seed(100) # makes output the same every time, can change value to any number
# make randomized data
randomized_data = {}
for variable, value_lst in df.items():
    new_lst = []
    for value in value_lst:
        new_random_value = uniform(lower, upper) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
# to clarify answer in output graph
for i in range(len(df["x"])):
    points_x_vals = df["x"][i], randomized_data["x"][i]
    points_y_vals = df["y"][i], randomized_data["y"][i]
    plt.plot(points_x_vals, points_y_vals, "black")
print (randomized_data)
# actual plotting of data
plt.scatter(df["x"], df["y"], label="original data")
plt.scatter(randomized_data["x"], randomized_data["y"], label="\"randomized\" data")
plt.legend()
plt.grid()
plt.show()
