#Erroneous Mapping for 16-QAM (Balanced Data) in Symbol Mapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Define the 16QAM constellation points w. Symbol (as integer)
df = pd.DataFrame({
    'enc': [0, 4, 1, 5, 12, 8, 13, 9, 3, 7, 2, 6, 15, 11, 14, 10],
    'I': [-3, -1, -3, -1, 1, 3, 1, 3, -3, -1, -3, -1, 1, 3, 1, 3],
    'Q': [3, 3, 1, 1, 3, 3, 1, 1, -1, -1, -3, -3, -1, -1, -3, -3]
})
print('16-QAM chart\n',df)
#Generate Duplicate values for balanced dataset
df3 = pd.concat([df] * 15, ignore_index=True)
print('16-QAM chart w. duplicates\n',df3)

# function to generate anomalies w.r.t any specific constellation point. e.g. for 16-QAM 16*15 inaccurate data
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
print('Anomalies for 16-QAM\n',new_df)
df1= pd.concat([df3, new_df], ignore_index=True) #trained data
print('org+anom\n',df1)
y=np.repeat([0,1], [240,240]) #trained label
#print('y1\n',y1)
df1['label']=y #adding labels to dataframe
print('org+anom+label\n',df1)

# Copy 20% random data from the original DataFrame for training data
num_rows_to_copy = int(0.2 * len(df1))
df2 = df1.sample(n=num_rows_to_copy, random_state=42, ignore_index=True)  # Use a fixed random state for reproducibility
print('sample 20% from original df\n',df2)

################train_test_data_label_separation#######################
X1=df1.drop(columns=['label']) #trained data
y1=df1['label'] #trained labels
X2=df2.drop(columns=['label']) #test data
#print('X2\n',X2)
y2=df2['label'] #test labels
#print('y2\n',y2)

################SVC_algorithm#######################
C=1 #Regularization parameter
gamma=0.01 #kernel coefficient
print('Result: ')
svc = SVC(kernel='rbf', C=C, degree=1, gamma='auto').fit(X1, y1) #kernel = linear or rbf
y_pred = svc.predict(X2)
accuracy = accuracy_score(y2, y_pred) * 100
print("SVM- without feature scaling: ",accuracy)

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
