#Erroneous Mapping for 16-QAM (Imbalanced Data) in Symbol Mapper w. One class SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Define the 16QAM constellation points
df1 = pd.DataFrame({
    'enc': [0, 4, 1, 5, 12, 8, 13, 9, 3, 7, 2, 6, 15, 11, 14, 10],
    'I': [-3, -1, -3, -1, 1, 3, 1, 3, -3, -1, -3, -1, 1, 3, 1, 3],
    'Q': [3, 3, 1, 1, 3, 3, 1, 1, -1, -1, -3, -3, -1, -1, -3, -3]
})
print('16-QAM chart\n',df1)

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

df2 = generate_combinations(df1)
print('Anomalies for 16-QAM\n',df2)
X_train=df1.to_numpy()
X_test = np.vstack((df1, df2)) #combine original+anomolous data
print('org+anom test data\n',X_test)
y_test = np.hstack((np.ones(len(df1)), np.zeros(len(df2))))  # 1->org,0->anom
print('y_test\n',y_test)

# Fit the One-Class SVM model
clf = svm.OneClassSVM(nu=0.005, kernel="rbf", gamma=0.1)
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
"""
