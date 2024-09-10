#Erroneous Mapping for 64-QAM (Balanced Data) in Symbol Mapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Define the 64QAM constellation points w. Symbol (as integer)
def generate_64qam_dataframe():
    # Define the constellation points for 64-QAM
    constellation = np.array([
        [-7, 7], [-7, 5], [-7, 1], [-7, 3],
        [-5, 7], [-5, 5], [-5, 1], [-5, 3],
        [-1, 7], [-1, 5], [-1, 1], [-1, 3],
        [-3, 7], [-3, 5], [-3, 1], [-3, 3],
        [7, 7], [7, 5], [7, 1], [7, 3],
        [5, 7], [5, 5], [5, 1], [5, 3],
        [1, 7], [1, 5], [1, 1], [1, 3],
        [3, 7], [3, 5], [3, 1], [3, 3],
        [-7, -7], [-7, -5], [-7, -1], [-7, -3],
        [-5, -7], [-5, -5], [-5, -1], [-5, -3],
        [-1, -7], [-1, -5], [-1, -1], [-1, -3],
        [-3, -7], [-3, -5], [-3, -1], [-3, -3],
        [7, -7], [7, -5], [7, -1], [7, -3],
        [5, -7], [5, -5], [5, -1], [5, -3],
        [1, -7], [1, -5], [1, -1], [1, -3],
        [3, -7], [3, -5], [3, -1], [3, -3]
    ])

    # Create a DataFrame with separate columns for I and Q
    df = pd.DataFrame(constellation, columns=['I', 'Q'])
    df['enc']= range(64)
    return df
df= generate_64qam_dataframe()
df = df[['enc', 'I', 'Q']]
print('64-QAM chart\n',df)
df3 = pd.concat([df] * 63, ignore_index=True)
print('16-QAM chart w. duplicates\n',df3)

# function to generate anomalies w.r.t any specific constellation point. e.g. for 64-QAM 64*63 inaccurate data
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
#apply function to dataset
new_df = generate_combinations(df)
print('Anomalies for 64-QAM\n',new_df)
df1= pd.concat([df3, new_df], ignore_index=True)
print('org+anom\n',df1)
y=np.repeat([0,1], [4032,4032])
#print('y1\n',y1)
df1['label']=y #adding labels to dataframe
df1=df1.sample(frac=1, ignore_index=True)
print('org+anom+label\n',df1)

# Copy 20% random data from the original DataFrame
num_rows_to_copy = int(0.2 * len(df1))
df2 = df1.sample(n=num_rows_to_copy, random_state=42, ignore_index=True)  # Use a fixed random state for reproducibility
print('sample 20% copy from original df\n',df2)

################train_test_data_label_separation#######################
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column
X1=df1.drop(columns=['label']).apply(min_max_normalize) #trained data
y1=df1['label']  #trained label
X2=df2.drop(columns=['label']).apply(min_max_normalize) #test data
#print('X2\n',X2)
y2=df2['label'] #test labels
#print('y2\n',y2)

################SVC_algorithm#######################
C=10000 #Regularization parameter
gamma=10 #kernel coefficient
print('Result: ')
svc = SVC(kernel='rbf', C=C, degree=1, gamma='auto').fit(X1, y1) #kernel = linear or rbf
y_pred = svc.predict(X2)
accuracy = accuracy_score(y2, y_pred) * 100
print("SVM-Linear - without feature scaling: ",accuracy)

# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)

#confusion_matrix
#cm = confusion_matrix(y2, y_pred, labels=svc.classes_)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
#disp.plot()
#plt.show()
