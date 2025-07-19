import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# Generating and saving train data
num_lines = 525
data = {
    'perf1': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
    'perf2': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
    'perf3': np.abs(np.round(np.random.uniform(6, 10, num_lines), 2)),
    'salary': np.abs(np.round(np.random.uniform(600000, 1200000, num_lines)))
}
df = pd.DataFrame(data)
df.to_csv('train_data.csv', index=False)

# Assuming you have test data saved in 'test_data.csv'
df_test = pd.read_csv('test_data.csv')
df_out = pd.DataFrame(columns=['actual_salary', 'predicted_salary'])

x_train = df[['perf1', 'perf2', 'perf3']]
y_train = df[['salary']]

x_test = df_test[['perf1', 'perf2', 'perf3']]

# Training the model
dt_model = tree.DecisionTreeRegressor()
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
df_out['predicted_salary'] = np.round(dt_predictions)

# Manually calculating the salary based on your provided code
def calculate_salary(perf1, perf2, perf3):
    if perf1 < 8:
        if perf2 < 8:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'
        elif perf2 > 9:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'
    if perf1 >= 8 and perf1 < 9:
        if perf2 < 8:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'
        elif perf2 > 9:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                 return 'zzzz'
    if perf1 > 9:
        if perf2 < 8:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                 return 'zzzz'
        elif perf2 >= 8 and perf2 < 9:
            if perf3 < 8:
                 return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                  return 'zzzz'
            elif perf3 >= 9:
                 return 'zzzz'
        elif perf2 > 9:
            if perf3 < 8:
                return 'xxxx'
            elif perf3 >= 8 and perf3 < 9:
                return 'zzzz'
            elif perf3 >= 9:
                return 'zzzz'

# Applying manual calculation to test data
df_out['actual_salary'] = df_test.apply(lambda row: calculate_salary(row['perf1'], row['perf2'], row['perf3']), axis=1)

df_out.to_csv('predicted_vs_actual.csv', index=False)

# Plotting
plt.plot(df_out['actual_salary'], marker='o', linestyle='-', color='g', label='Actual')
plt.plot(df_out['predicted_salary'], marker='x', linestyle='-', color='r', label='Predicted')
plt.legend()
plt.grid(True)
plt.show()
