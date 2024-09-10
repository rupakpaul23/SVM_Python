#Error Propagation demonstration in Channel Encoding Block
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###################Multisimulation_setup#############
num_simulations = 1
simulation_results1 = []
simulation_results2 = []
simulation_results3 = []

#loop for multiple simulations
for _ in range(num_simulations):
    ###################loading_the_dataset#############
    df = pd.read_csv (r'folder_path_data_enc_csvfile')
    df.columns = ['data', 'enc'] #naming the columns
    print(df)

    ###################Binary_data_extraction##############
    data1 = df.drop('enc', axis=1).dropna().astype(int)
    #print('bin_data\n', data1)

    # converts every 7 bits to one symbol
    bin_data2 = []
    for i in range(0, len(data1), 7):
        current_seven = data1.iloc[i: (i + 7)]
        # convert the values to strings to join them together
        new_entry = ''.join(map(str, list(current_seven['data'])))
        bin_data2.append(new_entry)
    data1 = pd.DataFrame({'bin_data2': bin_data2})
    print(data1)
    data1['bin_data_int'] = data1.bin_data2.apply(lambda x: (int(x, 2)))
    data1=data1.drop('bin_data2', axis=1) #removing binary column and convert to str for next ops
    #print('bin_data_new\n', data1)

    ###################encoded_data_extraction##############
    data2 = df.drop('data', axis=1)
    # print(data2)
    # converts every 14 bits to one symbol
    enc_data2 = []
    for i in range(0, len(data2), 14):
        current_fourteen = data2.iloc[i: (i + 14)]
        # convert the values to strings to join them together
        new_entry = ''.join(map(str, list(current_fourteen['enc'])))
        enc_data2.append(new_entry)
    data2 = pd.DataFrame({'enc_data2': enc_data2})
    print('enc_data_new\n',data2)

    # Function to flip the last 4 bits of a binary string
    def flip_last_n_bits(binary_string): #change here
        return binary_string[:-4] + ''.join(['1' if bit == '0' else '0' for bit in binary_string[-4:]])

    # Apply the function to each row in the DataFrame, #change here
    data2.loc[:1463, 'enc_data2'] = data2.loc[:1463, 'enc_data2'].apply(flip_last_n_bits) #878 for 30% and 1463 for 50% anomalies
    #data2.loc[:1463, 'enc_data2'] = data2.loc[:1463, 'enc_data2'].apply(flip_first_n_bits)
    print(data2)
    data2['enc_data_int'] = data2.enc_data2.apply(lambda x: (int(x, 2))) #converting to integer
    data2 = data2.drop('enc_data2', axis=1) #removing binary column and convert to str for next ops
    #print('After n bitflip in enc_data1\n', data2)

    ###############Data_Normalization###########################
    def min_max_normalize(column):
        min_val = column.min()
        max_val = column.max()
        normalized_column = (column - min_val) / (max_val - min_val)
        return normalized_column
    data = data1.join(data2['enc_data_int'])
    print(data)
    data = data.apply(min_max_normalize)
    print('data\n', data)
    X=data.to_numpy() #Final Dataset
    #print(X)
    # Create a label array y
    y = np.repeat([0, 1], [1463, 1463]) #Data labeling. e.g. [878, 2048] or [1463, 1463]
    #print(y)

    ###############data_splitting#################
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20, random_state=None, shuffle=True)
    #print('X_training set\n', X1)
    #print('y_training label\n', y1)
    #print('X_testing set\n', X2)
    # print('y_testing label\n', y2)

    ################SVC_algorithm#######################
    C = 100000 #Regularization
    gamma = 1 #Kernel coefficient
    linear_svc = SVC(kernel='rbf', C=C, gamma=gamma).fit(X1, y1) #kernel='linear'
    y_pred = linear_svc.predict(X2)
    accuracy = accuracy_score(y2, y_pred) * 100
    print("SVM Prediction Accuracy: ", accuracy)

    # accuracy per class calculation
    accuracy_class_0 = (accuracy_score(y2[y2 == 0], y_pred[y2 == 0])) * 100
    accuracy_class_1 = (accuracy_score(y2[y2 == 1], y_pred[y2 == 1])) * 100
    print('accuracy per class 0:', accuracy_class_0)
    print('accuracy per class 1:', accuracy_class_1)

    # Confusion Matrix calculation
    cm = confusion_matrix(y2, y_pred, labels=linear_svc.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_svc.classes_)
    disp.plot()
    plt.show()

    # simulation append one by one
    simulation_results1.append(accuracy)
    simulation_results2.append(accuracy_class_0)
    simulation_results3.append(accuracy_class_1)

#average of all simulations
average1 = sum(simulation_results1) / num_simulations
average2 = sum(simulation_results2) / num_simulations
average3 = sum(simulation_results3) / num_simulations

# display average result
print(f"The accuracy average of the {num_simulations} simulations is: {average1}")
print(f"The accuracy per class 0 average of the {num_simulations} simulations is: {average2}")
print(f"The accuracy per class 1 average of the {num_simulations} simulations is: {average3}")


