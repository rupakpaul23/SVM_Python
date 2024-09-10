#Erroneous Mapping for 64-QAM (Imbalanced Data) in Symbol Mapper w. One class SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Define the constellation points for 64-QAM
def generate_64qam_dataframe():
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
df1= generate_64qam_dataframe()
df1 = df1[['enc', 'I', 'Q']]
print('64-QAM chart\n',df1)

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

df2 = generate_combinations(df1)
print('Anomalies for 16-QAM\n',df2)
X_train=df1.to_numpy()
X_test = np.vstack((df1, df2)) #combine original+anomolous data
print('org+anom test data\n',X_test)
y_test = np.hstack((np.ones(len(df1)), np.zeros(len(df2))))  # 1->org,0->anom as labelling
print('y_test\n',y_test)


# Fit the One-Class SVM model
clf = svm.OneClassSVM(nu=0.099, kernel="rbf", gamma=0.1)
clf.fit(X_train)
# Predictions
y_pred_test = clf.predict(X_test)
# Convert predictions to binary (1 for inliers, -1 for outliers)
y_pred_test_binary = np.where(y_pred_test == 1, 1, 0)

# Calculate accuracy
accuracy = (accuracy_score(y_test, y_pred_test_binary))*100
print("Accuracy:", accuracy)
"""
# accuracy per class calculation
accuracy_class_0 = (accuracy_score(y_test[y_test == 0], y_pred_test_binary[y_test == 0])) * 100
accuracy_class_1 = (accuracy_score(y_test[y_test == 1], y_pred_test_binary[y_test == 1])) * 100
print('accuracy per class 0:', accuracy_class_0)
print('accuracy per class 1:', accuracy_class_1)

#################################
from sklearn.metrics import f1_score
y_test[y_test == 1] = 1
y_test[y_test == 0] = -1
# calculate score
score = f1_score(y_test, y_pred_test, pos_label=-1)
print('F1 Score: %.3f' % score)
"""