#Erroneous Mapping for 256-QAM (Balanced Data) in Symbol Mapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

###################loading_the_dataset#############
# Define the 256-QAM constellation points w. Symbol
df = pd.read_csv (r'C:\Users\RupakPaul\PycharmProjects\SVM\256qam.csv')
#df.columns = ['enc', '256qam']
df['256qam'] = df['256qam'].str.strip('()')
#print('Symbol+I,Q\n',df)

# Separate real and imaginary parts into new columns
df['256qam'] = df['256qam'].apply(lambda x: complex(x))
df['I'] = df['256qam'].apply(lambda x: x.real)
df['Q'] = df['256qam'].apply(lambda x: x.imag)
df=df.drop(columns=['256qam'])
# Display the modified DataFrame
print('256-QAM chart\n',df)
df3 = pd.concat([df] * 255, ignore_index=True)
print('16-QAM chart w. duplicates\n',df3)

# function to generate anomalies w.r.t any specific constellation point. e.g. for 256-QAM 256*255 inaccurate data
def generate_combinations(df):
    # Create an empty DataFrame to store combinations
    new_data = {'enc': [], 'I': [], 'Q': []}

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        # Repeat the values in Column1 for each combination
        values_column1 = [row['enc']] * (len(df)-1)

        # Append the values to the new DataFrame
        new_data['enc'].extend(values_column1)
        new_data['I'].extend(df[df.index != row.name]['I'])
        new_data['Q'].extend(df[df.index != row.name]['Q'])

    # Create a new DataFrame from the generated combinations
    new_df = pd.DataFrame(new_data)
    new_df = new_df.drop_duplicates()
    return new_df

new_df = generate_combinations(df)
print('Anomalies for 256-QAM\n',new_df)
df1= pd.concat([df3, new_df], ignore_index=True).astype('int') #trained data
print('org+anom\n',df1)
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column
#df1=df1.apply(min_max_normalize) #apply normalization
y=np.repeat([0,1], [65280, 65280]) #trained label
#print('y1\n',y1)
df1['label']=y
print('org+anom+label\n',df1)

# Copy 20% random data from the original DataFrame
num_rows_to_copy = int(0.2 * len(df1))
df2 = df1.sample(n=num_rows_to_copy, random_state=42, ignore_index=True)  # Use a fixed random state for reproducibility
print('sample 20% copy from original df\n',df2)

################train_test_data_label_separation#######################
X1=df1.drop(columns=['label']).apply(min_max_normalize)
print(X1)
y1=df1['label']
X2=df2.drop(columns=['label']).apply(min_max_normalize)
print(X2)
#print('X2\n',X2)
y2=df2['label']
#print('y2\n',y2)

################SVC_algorithm#######################
C=10000
gamma=100
print('Result: ')
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

#confusion matrix
cm = confusion_matrix(y2, y_pred, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
#disp.plot()
#plt.show()
