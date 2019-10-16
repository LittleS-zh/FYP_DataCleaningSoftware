import pandas as pd
import numpy as np
import tensorflow as tf
import copy

df = pd.read_csv("testFile_withoutNaN.csv")

# find the position of columns
for column_name in df.columns:
    if df[column_name].count() != len(df):
        loc = df[column_name][df[column_name].isnull().values == True].index.tolist()
        print(column_name, loc)

position_missing_value_row = 29
position_missing_value_column = "Column C"

# copy the data frame for further use
df_temp = copy.deepcopy(df)

# delete the column which is text
column_list = df_temp.columns.values.tolist()
print(column_list)
for element in column_list:
    print(df_temp[element].dtypes)
    if df_temp[element].dtypes == "object":
        df_temp.drop(element, axis = 1, inplace = True)
        print(df_temp)

# define the norm
norm = 100

# get the number of line
number_of_line = df.shape[0]
print("total number of line: ", number_of_line)

# take out the outlier row
df_outlier_line = df_temp[position_missing_value_row:position_missing_value_row+1]
print("The selected line of outlier is: ", df_outlier_line)
df_outlier_for_training = df_outlier_line.drop([position_missing_value_column], axis = 1)
print("The outlier for training is: ", df_outlier_for_training)


# delete it
df_temp.drop(position_missing_value_row, inplace=True)
print("After taking out the outlier row, the dataframe is: ", df_temp)

# prepare for the whole block of data
X = df_temp.drop([position_missing_value_column], axis = 1)
X = np.array(X)
X = X.astype(float)
X = X/norm
print("data frame of X is: ")
print(X)

Y = df_temp[position_missing_value_column]
Y = np.array(Y)
Y = Y.astype(float)
Y = Y/norm
print("data frame of Y is: ")
print(Y)

# get the liens before 16 as training set
X_train, Y_train = X[:int((number_of_line-1)*0.8)], Y[:int((number_of_line-1)*0.8)]
print("data frame of x_train is: ")
print(X_train)

print("data frame of y_train is: ")
print(Y_train)

# get the line after 16 as testing set
X_test, Y_test = X[int((number_of_line-1)*0.2):],Y[int((number_of_line-1)*0.2):]
print("dataframe of x_test is: ")
print(X_test)

print("dataframe of y_test is: ")
print(Y_test)

# construct the models
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation=tf.nn.relu),
     tf.keras.layers.Dense(64, activation=tf.nn.relu),
     tf.keras.layers.Dense(1)
     ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

# epochs: iteration times
# batch_size: divide the data into batch and train these batch during the training process
model.fit(X_train,Y_train,epochs=10,batch_size=15)

prediction_test_set = model.predict(X_test, batch_size = 15)

print(prediction_test_set*norm)

prediction_result = model.predict(df_outlier_for_training, batch_size = 15)

print("The output is: ", prediction_result*norm)