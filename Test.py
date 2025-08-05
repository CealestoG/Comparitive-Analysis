##### Generate Data
# import numpy as np
# import pandas as pd
# num_lines = 1000
#
# def generate_salary():
#     salary = np.abs(np.round(np.random.uniform(600000, 1200000, 1))[0])
#     # Ensure that the salary is a multiple of 100000
#     salary = (salary // 100000) * 100000
#     return salary
#
# data = {
#     'perf1': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
#     'perf2': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
#     'perf3': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
#     'salary': [generate_salary() for _ in range(num_lines)]
# }
#
# df = pd.DataFrame(data)
# df.head(800).to_csv('train_data.csv', index=True)
# df.tail(200).to_csv('test_data.csv', index=True)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import tree
from sklearn import tree
import csv

df_train = pd.read_csv('train_data.csv', nrows=500)
df_test = pd.read_csv('test_data.csv')
df_out = pd.DataFrame(columns=['perf1', 'perf2', 'perf3', 'actual_salary', 'predicted_salary'])

x_train = df_train[['perf1', 'perf2', 'perf3']]
y_train = df_train[['salary']]

x_test = df_test[['perf1', 'perf2', 'perf3']]
df_out['perf1'] = df_test[['perf1']]
df_out['perf2'] = df_test[['perf2']]
df_out['perf3'] = df_test[['perf3']]
df_out['actual_salary'] = df_test[['salary']]

dt_model = tree.DecisionTreeRegressor()
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
df_out['predicted_salary'] = np.round(dt_predictions)

df_out.to_csv('predicted_vs_actual.csv', index=True)

actual_salary = df_out['actual_salary']
predicted_salary = df_out['predicted_salary']


#### Manually calculate the salary
# Function to calculate salary based on performance
def calculate_salary(perf1, perf2, perf3):
    if perf1 < 8:
        if perf2 < 8:
            if perf3 < 8:
                return '881083'
            elif perf3 >= 8 and perf3 < 9:
                return '882763'
            elif perf3 >= 9:
                return '882763'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                return '889199'
            elif perf3 >= 8 and perf3 < 9:
                return '853811'
            elif perf3 >= 9:
                return '868424'
        elif perf2 > 9:
            if perf3 < 8:
                return '852213'
            elif perf3 >= 8 and perf3 < 9:
                return '949691'
            elif perf3 >= 9:
                return '880450'
    if perf1 >= 8 and perf1 < 9:
        if perf2 < 8:
            if perf3 < 8:
                return '871676'
            elif perf3 >= 8 and perf3 < 9:
                return '840695'
            elif perf3 >= 9:
                return '908009'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                return '904171'
            elif perf3 >= 8 and perf3 < 9:
                return '925837'
            elif perf3 >= 9:
                return '931778'
        elif perf2 > 9:
            if perf3 < 8:
                return '938998'
            elif perf3 >= 8 and perf3 < 9:
                return '954810'
            elif perf3 >= 9:
                 return '870029'
    if perf1 > 9:
        if perf2 < 8:
            if perf3 < 8:
                return '901650'
            elif perf3 >= 8 and perf3 < 9:
                return '913879'
            elif perf3 >= 9:
                 return '926167'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                 return '937857'
            elif perf3 >= 8 and perf3 < 9:
                  return '973809'
            elif perf3 >= 9:
                 return '940758'
        elif perf2 > 9:
            if perf3 < 8:
                return '936207'
            elif perf3 >= 8 and perf3 < 9:
                return '973635'
            elif perf3 >= 9:
                return '1022756'

# Read input CSV file
input_file_path = 'test_data.csv'
output_file_path = 'calculated_salary.csv'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    reader = csv.DictReader(input_file)
    fieldnames = reader.fieldnames + ['calculated_salary']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        perf1 = float(row['perf1'])
        perf2 = float(row['perf2'])
        perf3 = float(row['perf3'])

        # Calculate salary based on performance
        calculated_salary = calculate_salary(perf1, perf2, perf3)

        # Add the calculated salary to the row
        row['calculated_salary'] = calculated_salary

        # Write the updated row to the output CSV
        writer.writerow(row)

df1 = pd.read_csv('predicted_vs_actual.csv')
df2 = pd.read_csv('calculated_salary.csv')
# Specify columns to select from CSV1 and CSV2
columns_from_csv1 = ['perf1', 'perf2', 'perf3', 'actual_salary', 'predicted_salary']
columns_from_csv2 = ['calculated_salary']

# Create a new dataframe with selected columns
new_df = pd.DataFrame({
    'perf1': df1['perf1'],
    'perf2': df1['perf2'],
    'perf3': df1['perf3'],
    'actual_salary': df1['actual_salary'],
    'predicted_salary': df1['predicted_salary'],
    'calculated_salary': df2['calculated_salary'],
})

new_df.to_csv('actual_vs_predicted_vs_calculated.csv', index=False)

df3 = pd.read_csv('actual_vs_predicted_vs_calculated.csv')
calculated_salary = df3['calculated_salary']

plt.plot(actual_salary, marker='o', linestyle='-', color='g', label='Actual')
plt.plot(np.abs(actual_salary-predicted_salary), marker='x', linestyle='-', color='b', label='Predicted')
plt.plot(np.abs(actual_salary-calculated_salary), marker='x', linestyle='-', color='r', label='Calculated')
plt.plot(calculated_salary, marker='x', linestyle='-', color='b', label='Calculated')

plt.legend()
plt.grid(True)
plt.show()
