import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from itertools import product
from random import uniform, seed

###################loading_the_dataset#############
df = pd.read_csv (r'C:\Users\RupakPaul\PycharmProjects\SVM\16qam\enc_qam.csv')
df.columns = ['enc', 'qam']
print(df)

###################encoded_data_extraction##############
data1 = df.drop('qam', axis=1)
#print(data1)
enc = []
for i in range(int(len(data1) / 4)):
    # get the current four values
    current_four = data1.iloc[i * 4: (i * 4) + 4]
    # convert the values to strings to join them together
    new_entry = ''.join(map(str, list(current_four['enc'])))
    enc.append(new_entry)
data1 = pd.DataFrame({'enc': enc})
data1['enc_new'] = data1.enc.apply(lambda x: (int(x, 2)))
data1=data1.drop('enc', axis=1)
#print('enc_new\n',data1)

###################complex_data_extraction##############
data2 = df.drop('enc', axis=1).dropna()
#print(data2)
data = data1.join(data2['qam'])
data = data.sort_values(by='qam').reset_index(drop=True)
#print(data)
data['16-QAM'] = data['qam'].str.replace('i', 'j').apply(complex)
data = data.drop('qam', axis=1)
print('enc_int+16-QAM\n',data)

###############quadrants_splitting#################
# Create a new DataFrame with four columns for the four quadrants
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
data['Real'] = data['16-QAM'].apply(lambda x: x.real)
data['Imaginary'] = data['16-QAM'].apply(lambda x: x.imag)
data['Quadrant'] = ''
data.loc[(data['Real'] >= 0) & (data['Imaginary'] >= 0), 'Quadrant'] = 'I'
data.loc[(data['Real'] < 0) & (data['Imaginary'] >= 0), 'Quadrant'] = 'II'
data.loc[(data['Real'] < 0) & (data['Imaginary'] < 0), 'Quadrant'] = 'III'
data.loc[(data['Real'] >= 0) & (data['Imaginary'] < 0), 'Quadrant'] = 'IV'
# Separate DataFrames for each quadrant and their corresponding integer values
quadrant_I = data[data['Quadrant'] == 'I'].drop(columns=['Quadrant'])
quadrant_II = data[data['Quadrant'] == 'II'].drop(columns=['Quadrant'])
quadrant_III = data[data['Quadrant'] == 'III'].drop(columns=['Quadrant'])
quadrant_IV = data[data['Quadrant'] == 'IV'].drop(columns=['Quadrant'])

# Reset the indices for each quadrant DataFrame
quadrant_I.reset_index(drop=True, inplace=True)
quadrant_II.reset_index(drop=True, inplace=True)
quadrant_III.reset_index(drop=True, inplace=True)
quadrant_IV.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames for all four quadrants and their integer values
all_quadrants = pd.concat([quadrant_I, quadrant_II, quadrant_III, quadrant_IV], axis=1)

# Display the combined DataFrame
all_quadrants = all_quadrants.drop('16-QAM', axis=1)
print('All_Quadrants\n',all_quadrants)

##################################################
#random coordinate parameters
lower, upper = 1.5, 2.7
seed(10)
# SVC algorithm paramters

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

###################################################
###################################################
dframe1=all_quadrants.iloc[:,0:3].dropna()
#print(dframe1)
# randomly around their original values; (n=20%,30%,40%)
randomized_data = {}
for variable, value_list in dframe1.head(1291).items():  # change here
    new_lst = []
    for value in value_list:
        new_random_value = uniform(lower, upper) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
randomized_data = pd.DataFrame(randomized_data)
# print('Anomalies\n',randomized_data)
# replace the original value with new one in the dataset, according to their positions
dframe1.loc[randomized_data.index, 'Real'] = randomized_data['Real']
dframe1.loc[randomized_data.index, 'Imaginary'] = randomized_data['Imaginary']
#dframe1=dframe1.sample(frac=1) shuffle dataset
print('enc_new+org+anom1\n',dframe1)

X=dframe1.apply(min_max_normalize)
#print(X)
y = np.repeat([1, 0], [1291, 1292])  # change here
#print('y\n', y)

#################data splitting#######################
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20)
#print(X1)
#print(X2)

################SVC_algorithm#######################
C=1
gamma=0.1
print('1st Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=10
gamma=0.1
print('1st Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=100
gamma=0.1
print('1st Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=1000
gamma=0.1
print('1st Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)

#####################################################
######################################################
dframe2=all_quadrants.iloc[:,3:6].dropna()
#print(dframe2)
randomized_data = {}
for variable, value_list in dframe2.head(1223).items():  # change here
    new_lst = []
    for value in value_list:
        new_random_value = uniform(lower, upper) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
randomized_data = pd.DataFrame(randomized_data)
# print('Anomalies\n',randomized_data)
#replace the original value with new one in the dataset, according to their positions
dframe2.loc[randomized_data.index, 'Real'] = randomized_data['Real']
dframe2.loc[randomized_data.index, 'Imaginary'] = randomized_data['Imaginary']
#print('enc_new+org+anom2\n',dframe2)
X = dframe2.apply(min_max_normalize).to_numpy()
#print(X)
y = np.repeat([1, 0], [1223, 1224])  # change here
#print('y\n', y)

#################data splitting#######################
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20)

################SVC_algorithm#######################
C=1
gamma=0.1
print('2nd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=10
gamma=0.1
print('2nd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=100
gamma=0.1
print('2nd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=1000
gamma=0.1
print('2nd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)

###################################################
###################################################
dframe3=all_quadrants.iloc[:,6:9].dropna()
#print(dframe3)
# randomly around their original values; (n=20%,30%,40%)
randomized_data = {}
for variable, value_list in dframe3.head(1276).items():  # change here
    new_lst = []
    for value in value_list:
        new_random_value = uniform(lower, upper) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
randomized_data = pd.DataFrame(randomized_data)
# print('Anomalies\n',randomized_data)
# replace the original value with new one in the dataset, according to their positions
dframe3.loc[randomized_data.index, 'Real'] = randomized_data['Real']
dframe3.loc[randomized_data.index, 'Imaginary'] = randomized_data['Imaginary']
#print('enc_new+org+anom3\n',dframe3)
X = dframe3.apply(min_max_normalize).to_numpy()
#print(X)
y = np.repeat([1, 0], [1276, 1276])  # change here
#print('y\n', y)

#################data splitting#######################
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20)

################SVC_algorithm#######################
C=1
gamma=0.1
print('3rd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=10
gamma=0.1
print('3rd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=100
gamma=0.1
print('3rd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=1000
gamma=0.1
print('3rd Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
###################################################
###################################################
dframe4=all_quadrants.iloc[:,9:12].dropna()
#print(dframe4)
# randomly around their original values; (n=20%,30%,40%)
randomized_data = {}
for variable, value_list in dframe4.head(1328).items():  # change here
    new_lst = []
    for value in value_list:
        new_random_value = uniform(lower, upper) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
randomized_data = pd.DataFrame(randomized_data)
# print('Anomalies\n',randomized_data)
# replace the original value with new one in the dataset, according to their positions
dframe4.loc[randomized_data.index, 'Real'] = randomized_data['Real']
dframe4.loc[randomized_data.index, 'Imaginary'] = randomized_data['Imaginary']
#print('enc_new+org+anom4\n',dframe4)
X = dframe4.apply(min_max_normalize)
#print(X)
y = np.repeat([1, 0], [1328, 1329])  # change here
#print('y\n', y)

#################data splitting#######################
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20, shuffle=True)

################SVC_algorithm#######################
C=1
gamma=0.1
print('4th Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=10
gamma=0.1
print('4th Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=100
gamma=0.1
print('4th Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)
C=1000
gamma=0.1
print('4th Quadrant:')
svc = SVC(kernel='linear', C=C, degree=1, gamma='auto').fit(X1, y1)
y_pred = svc.predict(X2)
linear_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",linear_accuracy)
rbf_svc = SVC(kernel='rbf', C=C, degree=1, gamma=gamma).fit(X1, y1)
y_pred = rbf_svc.predict(X2)
rbf_accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-rbf - without feature scaling: ",rbf_accuracy)
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)

