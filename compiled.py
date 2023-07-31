import os
import shutil
import csv
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#--------------------------------------------------filter files--------------------------------------------------#

'''
to ensure mtg and xml files are the same
'''

# Source folders
folder1 = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\mtg"
folder2 = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\profile"

# Destination folder
destination_folder = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\filtered profile"
# Iterate over files in folder1
for filename1 in os.listdir(folder1):
    for filename2 in os.listdir(folder2):

        name1 = filename1.split('_')
        name2 = filename2.split('_')

        # Check if the file exists in folder2
        if name1[0] == name2[0]:

            file2 = os.path.join(folder2, filename2)

            # Copy the file to the destination folder
            shutil.copy(file2, destination_folder)

#--------------------------------------------------min rows--------------------------------------------------#

'''
to find least number of nodes in the mtg files
'''

# Folder path containing the csv files (specify directory path)
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\converted csv 2"
min_rows = np.inf

for file_name in os.listdir(folder_path):                                                   # Iterate over each file in the folder

    # Check if the file is a csv file
    if file_name.endswith(".csv"):

        # extract csv file
        csv_file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(csv_file_path)

        if len(df) < min_rows:
            min_rows = len(df)
            # print(file_name)

# print(min_rows)

#--------------------------------------------------mtg to csv conversion--------------------------------------------------#

# Folder path containing the MTG files (specify directory path)
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\mtg"

for file_name in os.listdir(folder_path):                                           # Iterate over each file in the folder

    # Check if the file is an MTG file
    if file_name.endswith(".mtg"):

        # extract MTG file
        mtg_file_path = os.path.join(folder_path, file_name)

        with open(mtg_file_path, 'r') as m1:
            lines = m1.readlines()

        data = []
        start = False
        for line in lines:
            if line.startswith('MTG :') and start == False:                         # only want the data after the line 'MTG :'
                start = True
                continue

            elif start == True:

                indentation = len(line) - len(line.lstrip())                        # calculate indentation
                elements = line.split()                                             # spilt into elements
                
                entity_code = ' '*indentation + elements[0]                         # add indentation to entity code

                if len(elements) == 1:                                              # for /P row
                    data.append([entity_code])
                
                elif len(elements) == 5:                                            # for rows without missing data
                    yy = elements[1]
                    xx = elements[2]
                    radius = elements[3]
                    zz = elements[4]
                    data.append([entity_code, yy, xx, radius, zz])

                else:                                                               # for rows with missing data
                    try:
                        ele1 = elements[1]
                        ele2 = elements[2]
                        ele3 = elements[3]
                        ele4 = -99
                        data.append([entity_code, ele1, ele2, ele3, ele4])
                    except:
                        print("MISSING DATA FOUND: ", elements)

        # export to csv (specify directory path)
        
        csv_file_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\converted csv\\" + file_name + ".csv"

        with open(csv_file_path, 'w', newline='') as csv_file:                      # Write the extracted data to the CSV file
            writer = csv.writer(csv_file)
            # writer.writerow(['Entity Code', 'XX', 'YY', 'ZZ', 'Radius'])          # Write the header row
            writer.writerows(data)                                                  # Write the data rows

#--------------------------------------------------parameter processing--------------------------------------------------#

'''
instead of to importing the csv file containing the max and min values of the parameters,
you can manually input the known min and max values of the species (similar to the beta and K values here)

'''

# Folder path containing the XML files (specify directory path)
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\filtered profile"

# csv file containing the max and min values of 'r' values (specify directory path)
df_ranges = pd.read_csv(r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\diameter growth rate\diameter_growth_rate_data.csv")
df_ranges.index = ['min', 'max']

dict_beta = {
    'min':-3.0124,
    'max':-3.6844
}

dict_K = {
    'min':-43.6656631,
    'max':58.4368109
}

data_r = []
data_beta = []
data_K = []
count1 = 1
for file_name in os.listdir(folder_path):                                           # Iterate over each file in the folder

    # if count1 == 2:
    #     break

    species = file_name.split('_')[1]

    # extract XML file
    xml_file_path = os.path.join(folder_path, file_name)

    with open(xml_file_path, 'r') as x1:
        lines = x1.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith('<diameter_growth_rate type="RGR">'):

            # Find the starting and ending indices of the value
            start_index = line.index('>') + 1
            end_index = line.index('</')

            # Extract the value of r
            value = line[start_index:end_index]

    percentage = round(float(value), 1)                                                     # first digit after decimal point -> 'r' value

    if species in df_ranges.columns:

        parameter = percentage * (df_ranges[species]['max']-df_ranges[species]['min']) + df_ranges[species]['min']

        data_r.append(parameter)

    else: data_r.append('unknown')

    # Extract value of beta
    second_digit = int(float(value) * 10) % 10                                                     # second digit after decimal point -> 'beta' value

    if species == 'SG':

        parameter_beta = second_digit * (dict_beta['max']-dict_beta['min']) + dict_beta['min']

        data_beta.append(parameter_beta)

    elif species not in df_ranges.columns: data_beta.append('unknown')

    # Extract value of K
    third_digit = int(float(value) * 100) % 10                                                     # thrid digit after decimal point -> 'K' value

    if species == 'SMy':

        parameter_K = third_digit * (dict_K['max']-dict_K['min']) + dict_K['min']

        data_K.append(parameter_K)

    elif species not in df_ranges.columns: data_K.append('unknown')

    count1 += 1

df_r = pd.DataFrame(data_r)
df_r = df_r.rename(columns={0: 'r'})  # Rename column 0 to 'r'
# print(df_r)

df_beta = pd.DataFrame(data_beta)
df_beta = df_beta.rename(columns={0: 'beta'})  # Rename column 0 to 'beta'

df_K = pd.DataFrame(data_K)
df_K = df_K.rename(columns={0: 'K'})  # Rename column 0 to 'K'

#--------------------------------------------------csv preprocessing--------------------------------------------------#

# Folder path containing the csv files (specify directory path)
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\synthesis data\converted csv"

def growth_model_type(file):
    name = file.split('_')

    if name[2] == 'SS':
        return 'linear'
    elif name[2] == 'PP':
        return 'linear'
    elif name[2] == 'HO':
        return 'linear'
    elif name[2] == 'KS':
        return 'linear'
    elif name[2] == 'AA':
        return 'linear'
    elif name[2] == 'TC':
        return 'unknown'
    elif name[2] == 'TR':
        return 'exponential'
    elif name[2] == 'SMa':
        return 'linear'
    elif name[2] == 'SG':
        return 'power_law'
    elif name[2] == 'SP':
        return 'exponential'
    elif name[2] == 'SMy':
        return 'monomolecular'
    
    else: return 'unknown'

index = 0
feature_list = []
labels = []

for file_name in os.listdir(folder_path):                                                   # Iterate over each file in the folder

    # if index == 5:                                                                        # Loop breaker for checking
    #     break

    # Check if the file is a csv file
    if file_name.endswith(".csv"):

        labels.append(growth_model_type(file_name))

        # extract csv file
        csv_file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(csv_file_path).head(min_rows)                                           # Only takes the first N nodes, where N=100 here
        # print(df)

        df_entity = df.iloc[:,0]
        # print(df_entity)

        df_diameter = df.iloc[:,3] * 2                                                      # radius * 2
        df_diameter = df_diameter.rename('diameter')
        # print(df_diameter)

        ############################## data preprocessing ##############################

        # Initialize variables
        branch = [2]                                                                       # stack data structure (LIFO)
        prev_indent = 1
        cell_ids = {}

        for row_index, cell in enumerate(df_entity):
            indent_level = len(cell) - len(cell.lstrip())                                  # Remove leading white spaces and calculate the indentation level

            cell_id = row_index + 1  # Assign cell ID

            if indent_level == prev_indent:
                cell_ids[row_index] = (cell_id, cell_id - 1)

            elif indent_level < prev_indent:
                while len(branch) > indent_level + 1:
                    branch.pop()

                cell_ids[row_index] = (cell_id, branch[-1]-1)

            elif indent_level > prev_indent:
                branch.append(cell_id)

                cell_ids[row_index] = (cell_id, cell_id - 1)

            prev_indent = indent_level

        # Convert cell_ids dictionary to DataFrame
        df_id = pd.DataFrame(cell_ids, index=['ID', 'neighbour_ID']).T

        # Merge the original DataFrame and the ID DataFrame
        df2 = pd.concat([df, df_id, df_diameter], axis=1)
        # print(df2)

        # Mapping neighbour's diameter to each row
        df3 = df2.merge(df2[['ID', 'diameter']], how='left', left_on='neighbour_ID', right_on='ID', suffixes=('', '_neighbour'))
        df3.drop(columns='ID_neighbour', inplace=True)
        # print(df3)

        # Extract the node diameters and neighbour diameters for nodes 1 to 100
        node_diameters = df3['diameter'].tolist()
        neighbour_diameters = df3['diameter_neighbour'].tolist()

        # Append the tree's flat list to the tree_data list
        feature_list.append([x for pair in zip(node_diameters, neighbour_diameters) for x in pair])

        index += 1

df_diameters = pd.DataFrame(feature_list)
column_names = []
for i in range(1, 101):
    column_names.append(f'node_{i}_diameter')
    column_names.append(f'node_{i}_parent_diameter')
df_diameters.columns = column_names

df_label = pd.DataFrame(labels)
df_label = df_label.rename(columns={0: 'growth_model_type'})

data = pd.concat([df_diameters, df_label, df_r], axis=1)

data.to_csv('dataset.csv', index=False)                                                 # export dataset to csv
# print(data)

# data.dropna(inplace=True)

unique_counts = data['growth_model_type'].value_counts()
print('number of classes in dataset:'.capitalize())
print(unique_counts)
print('\n')

#--------------------------------------------------classification (labels) task training--------------------------------------------------#

filtered_data = data[data.growth_model_type != 'unknown'].dropna()

# Separate features (X) and target variable (y)
X = filtered_data.drop(['growth_model_type', 'r'], axis=1)
y = filtered_data['growth_model_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# export features and target variables in both training and testing datasets to csv
X_train.to_csv('X_train_label.csv', index=False)
X_test.to_csv('X_test_label.csv', index=False)
y_train.to_csv('y_train_label.csv', index=False)
y_test.to_csv('y_test_label.csv', index=False)

unique_counts = y_train.value_counts()
unique_counts2 = y_test.value_counts()
print('number of classes in training dataset:'.capitalize())
print(unique_counts)
print('\n')
print('number of classes in testing dataset:'.capitalize())
print(unique_counts2)
print('\n')

classifiers = ['dt', 'rf', 'nb', 'knn']
accuracy_results = []
count2 = 1

for m in classifiers:

    ## select model
    if m == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()

    elif m == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    elif m == 'nb':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    elif m == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    ############################## model evaluation ##############################
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the evaluation results
    print("Accuracy:", accuracy)
    print('\n')
    print("Classification Report:\n", report)
    print('\n')

    ############################## export prediction results to csv ##############################
    csvname = 'model' + str(count2) + '.csv'

    df_pred = pd.DataFrame(y_pred)
    df_test = pd.DataFrame(y_test)

    df_test.reset_index(drop=True, inplace=True)
    df_test.rename(columns={'growth_model_type':'test'}, inplace=True)

    df_pred.reset_index(drop=True, inplace=True)
    df_pred.rename(columns={0:'prediction'}, inplace=True)

    pd.concat([df_test, df_pred], axis=1).to_csv(csvname, index=False)

    ############################## saving the trained models ##############################

    # export the trained models (specify directory path)
    folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\final\\"
    filename = 'model' + str(count2) + '.pkl'
    file_path = os.path.join(folder_path, filename)

    # Save the trained model to the specified file path
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

    count2 += 1

    # saving accuracy results for visualisation
    accuracy_results.append(accuracy)

    ############################# feature importance ##############################

    '''
    Feature importance is not used here due to the nature of the features used (child-parent node diameters).
    However, if other features such as diameter ratios are used, feature importance could be used to provide some insights.
    '''

    # try:
    #     importances = model.feature_importances_

    #     # Create a DataFrame to display feature importance
    #     feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

    #     # Sort the features by importance in descending order
    #     feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    #     # Print the feature importance
    #     print(feature_importance_df)
    #     print('\n')
    
    # except:
    #     continue

############################# visualisation ##############################

plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracy_results)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracies')
plt.ylim(0, 1.0)
plt.show()

#--------------------------------------------------classification (labels) task predicting--------------------------------------------------#

# Load the saved model from the folder (specify directory path)
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\final\\"

# Choose the classification model with the best accuracy
filename = 'model2.pkl'
file_path = os.path.join(folder_path, filename)

with open(file_path, 'rb') as file:
    model = pickle.load(file)

filtered_data = data[data.growth_model_type == 'unknown'].dropna()

X_test = filtered_data.drop(['growth_model_type', 'r'], axis=1)

# Use the loaded model for predictions
y_pred = model.predict(X_test)

# Print the prediction labels
print(y_pred)
print('\n')

unique_values, counts = np.unique(y_pred, return_counts=True)

# Print the unique values and their counts
label, largest = None, 0
for value, count in zip(unique_values, counts):
    if count > largest:
        label = value
        largest = count

#     print(f"{value}: {count}")
# print('\n')

print(f'Predicted growth model: {label}')
print('\n')

# Make predictions on the test set
# y_pred_proba = model.predict_proba(X_test)

# Print the probability estimates for the first 5 instances/samples (to gain insight about the prediction)
# class_labels = model.classes_
# probabilities = model.predict_proba(X_test)

# for i, prob in enumerate(probabilities[:5]):
#     print(f"Instance {i+1}:")
#     for j, class_label in enumerate(class_labels):
#         print(f"Probability of {class_label}: {prob[j]}")
#     print()

#--------------------------------------------------regression (parameters) task training--------------------------------------------------#

# amend as required to find parameter
parameter_to_predict = 'r'

'''
if trying to predict other parameters apart from 'r', 
1) drop 'r' from the dataset
2) reset indexes of filtered_data and df_beta / df_K
3) concatenante the other parameters such as 'beta' / 'K' from df_beta / df_K to the filtered_data
'''

# Load the data into a DataFrame
filtered_data = data[(data['growth_model_type'] == label) & (data[parameter_to_predict] != 'unknown')].dropna()

# # to find other parameters apart from 'r'
# filtered_data.reset_index(drop=True, inplace=True)
# df_beta.reset_index(drop=True, inplace=True)
# filtered_data = pd.concat([filtered_data, df_beta], axis=1)

# Extract the features (X) and target variable (y)
X = filtered_data.drop(['growth_model_type', parameter_to_predict], axis=1)
y = filtered_data[parameter_to_predict]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# export features and target variables in both training and testing datasets to csv
X_train.to_csv('X_train_reg.csv', index=False)
X_test.to_csv('X_test_reg.csv', index=False)
y_train.to_csv('y_train_reg.csv', index=False)
y_test.to_csv('y_test_reg.csv', index=False)

regressors = ['dt', 'rf']
accuracy_results2 = []

for m in regressors:
    
    # Create a Decision Tree/Random Forest Regression model
    # select model
    if m == 'dt':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()

    elif m == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    ############################## model evaluation ##############################
    csvname = 'model' + str(count2) + '.csv'

    df_pred = pd.DataFrame(y_pred)                                                                  # convert y_pred to df
    df_test = pd.DataFrame(y_test)                                                                  # convert y_test to df

    # rename and reset indexes
    df_test.reset_index(drop=True, inplace=True)
    df_test.rename(columns={parameter_to_predict:'test'}, inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    df_pred.rename(columns={0:'prediction'}, inplace=True)

    percentage_change = ((df_pred['prediction'] - df_test['test']) / df_test['test']) * 100         # calculate % change
    df_result = pd.concat([df_test, df_pred, percentage_change.rename('%_change')], axis=1)
    df_result.to_csv(csvname, index=False)                                                          # export result to csv

    within_10_percent = df_result[abs(df_result['%_change']) <= 10]                                 # find proportion of predicted values within 10% of true value
    accuracy2 = len(within_10_percent) / len(df_result) * 100

    print("Accuracy:", accuracy2)
    print('\n')

    # saving accuracy results for visualisation
    accuracy_results2.append(accuracy2/100)

    ############################## saving the trained models ##############################

    folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\final\\"
    filename = 'model' + str(count2) + '.pkl'
    file_path = os.path.join(folder_path, filename)

    # Save the trained model to the specified file path
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

    count2 += 1

    ############################# feature importance ##############################

    '''
    Feature importance is not used here due to the nature of the features used (child-parent node diameters).
    However, if other features such as diameter ratios are used, feature importance could be used to provide some insights.
    '''

    # importances = model.feature_importances_

    # # Create a DataFrame to display feature importance
    # feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

    # # Sort the features by importance in descending order
    # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # # Print the feature importance
    # print(feature_importance_df)

############################# visualisation ##############################

plt.figure(figsize=(8, 6))
plt.bar(regressors, accuracy_results2)
plt.xlabel('Regressor')
plt.ylabel('Accuracy')
plt.title('Regressor Accuracies')
plt.ylim(0, 1.0)
plt.show()

#--------------------------------------------------regression (parameters) task predicting--------------------------------------------------#

# Load the saved model from the file
folder_path = r"C:\Users\Seth\OneDrive - National University of Singapore\Documents\Internships\AStar\Code\final\\"

# Choose the regression model with the best accuracy
filename = 'model5.pkl'

file_path = os.path.join(folder_path, filename)

with open(file_path, 'rb') as file:
    model = pickle.load(file)

filtered_data = data[data.growth_model_type == 'unknown'].dropna()

X_test = filtered_data.drop(['growth_model_type', parameter_to_predict], axis=1)

# Use the loaded model for predictions
y_pred = model.predict(X_test)

print(y_pred)
print('\n')

# visualisation

plt.plot(y_pred, marker='o')
plt.xlabel('Tree Instance')
plt.ylabel('Predicted Value')
plt.title('Predicted Values for Unseen Trees')
plt.show()
