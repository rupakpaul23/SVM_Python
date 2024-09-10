import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv (r'folder_path_serialized_demodulated_csvfile')
df.columns = ['serialized', 'demodulated'] #naming the columns
print(df)

#reads demodulated symbols
serialized=df[["serialized"]].dropna()
print(serialized)

arr = np.complex128(serialized['serialized'].str.replace('i', 'j').to_numpy())
data1 = serialized.assign(x = arr.real, y = arr.imag)
data1= data1.drop('serialized', axis=1)
print('serialized\n',data1)

#reads demodulated data
demodulated=df[["demodulated"]].dropna().astype(int)
#print('demod_data\n',demod_data)
demodulated1=[]
for i in range(0, len(demodulated), 4):
    # get the current four values
    current_four = demodulated.iloc[i: (i + 4)]
    # convert the values to strings to join them together
    new_entry = ''.join(map(str, list(current_four['demodulated'])))
    demodulated1.append(new_entry)
data2 = pd.DataFrame({'demodulated': demodulated1})
data2=data2[:-1]
print(data2)
###################################################
data2.demodulated.iloc[:3071] = data2.demodulated.iloc[:3071].str.slice(stop=2) + (1 - data2.demodulated.iloc[:3071].str.get(2).astype(int)).astype(str) + data2.demodulated.iloc[:3071].str.slice(start=3)
data2.demodulated.iloc[:3071] = data2.demodulated.iloc[:3071].str.slice(stop=3) + (1 - data2.demodulated.iloc[:3071].str.get(3).astype(int)).astype(str) + data2.demodulated.iloc[:3071].str.slice(start=4)
print(data2)
data2['demodulated_int'] = data2.demodulated.apply(lambda x: (int(x, 2))) #converting to integer
data2 = data2.drop('demodulated', axis=1) #removing binary column#print(data2)
print('After n bitflip in demodulated_data\n', data2)

data=data1.join(data2['demodulated_int'])
print (data)
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

data = data.apply(min_max_normalize)
print('data\n', data)
X=data.to_numpy()
print(X)
y = np.repeat([0, 1], [3071,7168])
print('y\n',y)

#data splitting
X1, X2, y1, y2 = train_test_split(X, y, test_size = 0.20)
#print('X_training set\n', X1)
#print('y_training label\n', y1)
#print('X_testing set\n', X2)
#print('y_testing label\n', y2)

################SVC_algorithm#######################
C = 1000
gamma = 100
################SVC_algorithm#######################
linear_svc = SVC(kernel='linear', C=C, gamma=gamma).fit(X1, y1)
y_pred = linear_svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ", linear_accuracy)

# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
cm = confusion_matrix(y2, y_pred, labels=linear_svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_svc.classes_)
disp.plot()
plt.show()

rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)




