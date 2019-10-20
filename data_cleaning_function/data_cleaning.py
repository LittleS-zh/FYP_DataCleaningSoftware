import pandas as pd
import numpy as np, numpy
import copy
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from collections import Counter


class DataCleaning(object):
    __current_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    # this attribute is for reset the operation
    __original_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    # this attribute is for revert the operation
    __list_data_frame = []
    __temp_data_frame_for_deepcopy = []

    # indicate whether there is a question in Python
    __wrong_in_python = False

    # indicate which operation that users are doing
    __detect_outlier_single_format = False
    __detect_outlier_all_attributes = False
    __check_missing_value = False

    # this attribute is for check missing value
    __missing_value_result = []

    # this attribute is for indicating the rows with outlier
    __rowWithOutlier = []

    # this attribute is for indicating the choice in "detect outlier"
    __choice_in_detect_outlier = -1
    __column_detect_name = ''

    # this attribute is for outputting the most similarity sentences
    __text_similarity = 0

    def __init__(self):
        print("BackEnd Started succesfully")

    # read datafile from csv
    def read_data_file(self, file_location,
                       delimiter_input=",", encoding_input="utf-8",
                       header_input=0):
        # 初始化
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__rowWithOutlier.clear()
        self.__missing_value_result.clear()
        self.__choice_in_detect_outlier = -1

        # read the datafiles the user uploaded
        self.__current_data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input,
                                                header=header_input)

        # change all int number to float so that it can be more beautiful when showing to users.
        print("Changing all int type to float")
        number_of_column = self.__current_data_frame.columns.size
        print(number_of_column)
        for i in range(0, number_of_column):
            print(self.__current_data_frame.iloc[:, i].dtypes)
            if self.__current_data_frame.iloc[:, i].dtypes == "int64":
                self.__current_data_frame.iloc[:, i] = self.__current_data_frame.iloc[:, i].astype("float64")
                print("int type, change it to float type")

        self.__original_data_frame = copy.deepcopy(self.__current_data_frame)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

        # for debugging
        # print(self.__current_data_frame.shape[0])
        # print(self.__current_data_frame.shape[1])

    # write datafile to an csv file
    def write_data_file(self, path):
        self.__current_data_frame.to_csv(path)

    # select rows for a data frame
    def select_rows(self, row_start_input, row_end_input):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if row_start_input <= row_end_input and row_end_input <= self.__current_data_frame.shape[
            0] and row_start_input >= 1:
            print(self.__current_data_frame.shape[0])
            self.__current_data_frame = self.__current_data_frame[row_start_input - 1:row_end_input]
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        else:
            print("invalid input")

    # select columns for a data frame, according to the start number and the end number
    def select_column_position(self, column_start_input, column_end_input):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if column_start_input <= column_end_input and column_end_input <= self.__current_data_frame.shape[
            1] and column_start_input >= 1:
            print(self.__current_data_frame.shape[1])
            self.__current_data_frame = self.__current_data_frame.iloc[:, column_start_input - 1: column_end_input]
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        else:
            print("invalid input")

    # select columns for a data frame, according to the header of column
    def select_column_heading(self, header_of_column_input):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        self.__current_data_frame = self.__current_data_frame[header_of_column_input]
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # select row and column at the same time
    def block_selection(self, column_start_input, column_end_input, header_of_column_input):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        self.__current_data_frame = self.__current_data_frame.ix[column_start_input:column_end_input,
                                    header_of_column_input]
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # create a new column according to the operation based on two previous columns
    def algorithm_operation_on_blocks(self):
        self.__current_data_frame['newColumn'] = self.__current_data_frame['Day'] * self.__current_data_frame['Return']

    # filter the rows according to a specific value
    # the algorithms need to be optimized
    def select_rows_by_conditions(self, input_column, input_condition_operator, input_condition_number):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if input_condition_operator == "==":
            self.__current_data_frame = self.__current_data_frame[
                (self.__current_data_frame[input_column] == input_condition_number)]
        elif input_condition_operator == "!=":
            self.__current_data_frame = self.__current_data_frame[
                (self.__current_data_frame[input_column] != input_condition_number)]
        elif input_condition_operator == ">":
            self.__current_data_frame = self.__current_data_frame[
                (self.__current_data_frame[input_column] > input_condition_number)]
        elif input_condition_operator == "<":
            self.__current_data_frame = self.__current_data_frame[
                (self.__current_data_frame[input_column] < input_condition_number)]

        self.__current_data_frame.index = range(len(self.__current_data_frame))
        # print(self.__current_data_frame)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    def data_reduction(self, drop_header, group_header, sum_or_mean, combine_name):
        # 初始化
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if sum_or_mean == 1:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).sum().sort_values(combine_name,
                                                ascending=False)
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

        else:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).mean().sort_values(combine_name,
                                                 ascending=False)
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # deduplicate data
    def data_de_duplication(self, input_ignore_upper_case):
        # for test only, this will be delete later
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if "isDuplicate" in list(self.__current_data_frame.columns):
            self.__current_data_frame.drop(["isDuplicate"], axis=1, inplace=True)

        if not input_ignore_upper_case:
            self.__current_data_frame["isDuplicate"] = self.__current_data_frame.duplicated()
        elif input_ignore_upper_case:
            df = self.__current_data_frame

            df_temp = copy.deepcopy(df)

            number_of_column = df_temp.columns.size

            for i in range(0, number_of_column):
                print(i)
                print(df_temp.iloc[:, i].dtypes)
                if df_temp.iloc[:, i].dtypes == "float64" or df.iloc[:, i].dtypes == "int64":
                    print("Numbers, pass")
                elif df_temp.iloc[:, i].dtypes == "object":
                    df_temp.iloc[:, i] = df_temp.iloc[:, i].str.lower()
                    print("object, change to lower case")

            df["isDuplicate"] = df_temp.duplicated()

        self.__current_data_frame = self.__current_data_frame.drop('isDuplicate', axis=1)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # calculate means of data
    def calculate_means(self, column_need_to_be_grouped_input, column_group_by):
        # 初始化
        self.__wrong_in_python = False
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        self.__current_data_frame = self.__current_data_frame[column_need_to_be_grouped_input].groupby(
            self.__current_data_frame[column_group_by])
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        return self.__current_data_frame.mean()

    # detect outlier using three sigma method
    def detect_outlier_three_sigma(self, column_input):
        self.__wrong_in_python = False
        # print(self.__current_data_frame[column_input].dtype)
        # print(self.__current_data_frame[column_input].dtypes)
        if column_input == "all_attributes":
            try:
                self.detect_outlier_all()
                return
            except:
                print("Error happen in detecting all attribute.")
                self.__wrong_in_python = True

        if self.__current_data_frame[column_input].dtypes == "float64" or self.__current_data_frame[
            column_input].dtypes == "int64":
            try:
                # 初始化用户选择
                self.__detect_outlier_single_format = True
                self.__detect_outlier_all_attributes = False
                self.__check_missing_value = False
                self.__missing_value_result.clear()

                # 初始化数组
                temp_outlier = []
                temp_outlier.clear()
                self.__rowWithOutlier.clear()
                self.__column_detect_name = column_input

                d = self.__current_data_frame[column_input]
                z_score = (d - d.mean()) / d.std()
                self.__current_data_frame['isOutlier'] = z_score.abs() > 3

                temp_outlier = self.__current_data_frame[self.__current_data_frame['isOutlier'] == True].index.tolist()
                self.__rowWithOutlier = [i + 1 for i in temp_outlier]
                self.__choice_in_detect_outlier = self.__current_data_frame.columns.get_loc(column_input)

                self.__current_data_frame = self.__current_data_frame.drop('isOutlier', axis=1)
                self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
                self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
            except:
                print("Error happen in detecting number outlier.")
                self.__wrong_in_python = True


        elif self.__current_data_frame[column_input].dtypes == "object":
            try:
                self.detect_outlier_text(column_input)
            except:
                print("Error happen in detecting text outlier.")
                self.__wrong_in_python = True

    def fill_blank(self, modification_row):
        print(self.__column_detect_name)
        print(modification_row)
        self.__wrong_in_python = False

        df = self.__current_data_frame

        # find the position of columns
        for column_name in df.columns:
            if df[column_name].count() != len(df):
                loc = df[column_name][df[column_name].isnull().values == True].index.tolist()
                print(column_name, loc)

        position_fillin_value_row = modification_row
        position_fillin_value_column = self.__column_detect_name

        # copy the data frame for further use
        df_temp = copy.deepcopy(df)

        # delete the column which is text
        column_list = df_temp.columns.values.tolist()
        print(column_list)
        for element in column_list:
            print(df_temp[element].dtypes)
            if df_temp[element].dtypes == "object":
                df_temp.drop(element, axis=1, inplace=True)
                print(df_temp)

        # define the norm
        norm = 100

        # take out the outlier row
        df_outlier_line = df_temp[position_fillin_value_row:position_fillin_value_row + 1]
        print("The selected line of outlier is: ", df_outlier_line)
        df_outlier_for_training = df_outlier_line.drop([position_fillin_value_column], axis=1)
        print("The outlier for training is: ", df_outlier_for_training)

        # delete it
        df_temp.drop(position_fillin_value_row, inplace=True)
        print("After taking out the outlier row, the dataframe is: ", df_temp)

        # delete the rows which contains nan values
        df_temp.dropna(axis=0, inplace=True)

        # get the number of line
        number_of_line = df_temp.shape[0]
        print("total number of line: ", number_of_line)

        # prepare for the whole block of data
        X = df_temp.drop([position_fillin_value_column], axis=1)
        X = np.array(X)
        X = X.astype(float)
        X = X / norm
        print("data frame of X is: ")
        print(X)

        Y = df_temp[position_fillin_value_column]
        Y = np.array(Y)
        Y = Y.astype(float)
        Y = Y / norm
        print("data frame of Y is: ")
        print(Y)

        # get the liens before 16 as training set
        X_train, Y_train = X[:int(number_of_line * 0.8)], Y[:int(number_of_line * 0.8)]
        print("data frame of x_train is: ")
        print(X_train)

        print("data frame of y_train is: ")
        print(Y_train)

        # get the line after 16 as testing set
        X_test, Y_test = X[int(number_of_line * 0.2):], Y[int(number_of_line * 0.2):]
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
        model.fit(X_train, Y_train, epochs=10, batch_size=15)

        prediction_test_set = model.predict(X_test, batch_size=15)

        print(prediction_test_set * norm)

        prediction_result = model.predict(df_outlier_for_training, batch_size=15)

        print("The output is: ", prediction_result * norm)

        self.__current_data_frame.loc[position_fillin_value_row - 1, position_fillin_value_column] = float(
            prediction_result)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    def outlier_modification(self, modification_value, modification_row):
        # todo: get users' original floating point
        self.__wrong_in_python = False
        try:
            if modification_value.isdigit():
                self.__current_data_frame.loc[modification_row - 1, self.__column_detect_name] = float(modification_value)
                self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
                self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
            else:
                self.__current_data_frame.loc[modification_row - 1, self.__column_detect_name] = modification_value
                self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
                self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        except:
            print("wrong in outlier modification")
            self.__wrong_in_python = True
        # print(self.__list_data_frame)

    def single_outlier_delete(self, modification_row):
        self.__wrong_in_python = False
        try:
            self.__current_data_frame = self.__current_data_frame.drop(modification_row - 1)
            self.__current_data_frame.index = range(len(self.__current_data_frame))

            for outlier_index in range(0, len(self.__rowWithOutlier)):
                if self.__rowWithOutlier[outlier_index] > modification_row:
                    self.__rowWithOutlier[outlier_index] -= 1
            self.__rowWithOutlier.remove(modification_row)
        except:
            print("wrong in single_outlier_delete")
            self.__wrong_in_python = True

        # print(self.__current_data_frame)

    def single_missing_value_modification(self, modification_row, modification_column):
        print(modification_column)
        print(modification_row)

    def single_missing_value_delete(self, modification_row):
        self.__wrong_in_python = False
        try:
            self.__current_data_frame = self.__current_data_frame.drop(modification_row - 1)
            self.__current_data_frame.index = range(len(self.__current_data_frame))

            for missing_value_index in range(0, len(self.__missing_value_result)):
                if self.__missing_value_result[missing_value_index] > modification_row:
                    self.__missing_value_result[missing_value_index] -= 1
            self.__missing_value_result.remove(modification_row - 1)
        except:
            print("wrong in single_outlier_delete")
            self.__wrong_in_python = True

    # todo: this function is not finished
    def detect_outlier_quantitile(self, column_input):
        d = self.__current_data_frame[column_input]
        self.__current_data_frame['isOutlier'] = d > d.quantitile(0.75)

    # delete the whole row of outlier using three sigma methods
    def deal_with_outlier(self, column_input):
        # self.detect_outlier_three_sigma(column_input)

        # this code delete the outlier, which means
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == False)]
        self.__current_data_frame = self.__current_data_frame.drop("isOutlier", axis=1)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

        self.__current_data_frame.index = range(len(self.__current_data_frame))
        # print(self.__current_data_frame)

        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

    # check missing value
    def check_missing(self):
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = True
        self.__missing_value_result.clear()

        # data_frame_column = np.array(self.__current_data_frame.columns)
        # print("data_frame_column: ", data_frame_column)
        # data_frame_array = np.array(self.__current_data_frame.isnull())
        # print("data_frame_array: ", data_frame_array)
        self.__missing_value_result = self.__current_data_frame[
            self.__current_data_frame.isnull().values == True].index.tolist()
        # print(self.__missing_value_result)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        # print(self.__current_data_frame)
        # data_frame_list = data_frame_array.tolist()
        # print("data_frame_list: ", data_frame_list)
        # data_dictionary = {'data_frame': data_frame_list, 'data_header': data_frame_column}
        # return data_dictionary

    # delete missing value
    def deal_with_missing_value(self, choice):
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        if choice == "1":
            self.__current_data_frame = self.__current_data_frame.dropna(axis=0)
        elif choice == "2":
            self.__current_data_frame = self.__current_data_frame.dropna(axis=1)
        elif choice == "3":
            self.__current_data_frame = self.__current_data_frame.fillna("missing")
        elif choice == "4":
            self.__current_data_frame = self.__current_data_frame.fillna(method='pad')
        elif choice == "5":
            self.__current_data_frame = self.__current_data_frame.fillna(method='bfill', limit=1)
        elif choice == "6":
            self.__current_data_frame = self.__current_data_frame.fillna(self.__current_data_frame.mean())

        self.__missing_value_result.clear()
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # change the data frame into a list and return to the front end for showing
    def get_frame(self, float_round):
        data_frame_column = np.array(self.__current_data_frame.columns)
        data_frame_array = np.array(self.__current_data_frame.round(float_round))
        data_frame_list = data_frame_array.tolist()
        data_frame_list.insert(0, data_frame_column)
        data_dictionary = {'data_frame': data_frame_list,
                           'data_header': data_frame_column,
                           'wrong_in_python': self.__wrong_in_python,
                           'detect_outlier_single_format': self.__detect_outlier_single_format,
                           'detect_outlier_all_attributes': self.__detect_outlier_all_attributes,
                           'data_outlier': self.__rowWithOutlier,
                           'detect_outlier_choice': self.__choice_in_detect_outlier,
                           'check_missing_value': self.__check_missing_value,
                           'missing_value_result': self.__missing_value_result,
                           'text_similarity': self.__text_similarity}
        return data_dictionary

    # reset function
    def reset_data_frame(self):
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        self.__column_detect_name = ''
        self.__current_data_frame = copy.deepcopy(self.__original_data_frame)

    # revert function
    def revert_data_frame(self):
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__choice_in_detect_outlier = -1
        self.__rowWithOutlier.clear()

        if len(self.__list_data_frame) > 1:
            self.__list_data_frame.pop()
            self.__current_data_frame = copy.deepcopy(self.__list_data_frame[-1])

    def forecast_a_value(self, target_column_input, target_column_forecast, target_row_input, target_row_output):
        # todo: parameters change here
        X = self.__current_data_frame.iloc[:, target_column_input]
        norm = 100
        X = np.array(X)
        X = X.astype(float)
        X = X / norm
        # print("dataframe of X is: ")
        # print(X)

        Y = self.__current_data_frame.iloc[:, 2:3]
        Y = np.array(Y)
        Y = Y.astype(float)
        Y = Y / norm
        # print("dataframe of Y is: ")
        # print(Y)

        # get the liens before 16 as training set
        X_train, Y_train = X[:15], Y[:15]
        # print("dataframe of x_train is: ")
        # print(X_train)

        # print("dataframe of y_train is: ")
        # print(Y_train)

        # get the line after 16 as testing set
        X_test, Y_test = X[15:], Y[15:]
        # print("dataframe of x_test is: ")
        # print(X_test)

        # print("dataframe of y_test is: ")
        # print(Y_test)

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
        model.fit(X_train, Y_train, epochs=5, batch_size=15)

        prediction = model.predict(X_test, batch_size=1)

        # print(prediction * norm)

    def detect_outlier_text(self, input_column_which_is_text):
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__detect_outlier_single_format = True
        temp_outlier = []
        temp_outlier.clear()
        self.__rowWithOutlier.clear()
        self.__column_detect_name = input_column_which_is_text

        df_text = self.__current_data_frame[input_column_which_is_text]

        array_text = df_text.values
        array_text2 = numpy.array(df_text)
        # print(array_text2.flatten())
        vectorizer = CountVectorizer()

        arranged_text = vectorizer.fit_transform(array_text2.flatten())

        transformer = TfidfTransformer()
        arranged_text_tfidf = transformer.fit_transform(arranged_text)

        # print(vectorizer.get_feature_names())
        # print(arranged_text.toarray())

        # print(arranged_text_tfidf.toarray())

        array_tfidf = arranged_text_tfidf.toarray()

        model = KMeans(n_clusters=5)
        model.fit(array_tfidf)
        predicted_label = model.predict(array_tfidf)
        # print("tfidf", predicted_label)
        count_predicted_label = Counter(predicted_label)
        count_sorted = sorted(count_predicted_label.items(), key=lambda x: x[1])
        # print("Counter_sorted", count_sorted)
        count_sorted_values = sorted(count_predicted_label.values())

        items_number = 0
        # decide get the first n element
        while items_number < len(count_sorted_values) - 1:
            if count_sorted_values[items_number] == count_sorted_values[items_number + 1]:
                items_number += 1
            else:
                break

        # print("items_number", items_number + 1)

        output_position = []

        for i in range(0, items_number + 1):
            print(count_sorted[i][0])
            target_position = numpy.argwhere(predicted_label == count_sorted[i][0])
            output_position.append(target_position[0])
        # print(output_position)
        # print(output_position[1])
        self.__current_data_frame['isOutlier'] = False
        for elements in output_position:
            for element in elements:
                self.__current_data_frame.loc[element, "isOutlier"] = True

        temp_outlier = self.__current_data_frame[self.__current_data_frame['isOutlier'] == True].index.tolist()
        self.__rowWithOutlier = [i + 1 for i in temp_outlier]
        self.__choice_in_detect_outlier = self.__current_data_frame.columns.get_loc(input_column_which_is_text)
        # print(input_column_which_is_text)
        # print(self.__choice_in_detect_outlier)
        self.__current_data_frame = self.__current_data_frame.drop('isOutlier', axis=1)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    def text_similarity(self, input_words, column_chosen):
        import jieba
        import numpy
        from gensim import corpora, models, similarities

        df_text = self.__current_data_frame[column_chosen]

        doc_test = input_words

        temp_array_text = numpy.array(df_text)

        array_text = temp_array_text.flatten()

        all_doc_list = []
        for doc in array_text:
            doc_list = [word for word in jieba.cut(doc)]
            all_doc_list.append(doc_list)

        print(all_doc_list)

        doc_test_list = [word for word in jieba.cut(doc_test)]
        print(doc_test_list)

        # make a corpus
        dictionary = corpora.Dictionary(all_doc_list)
        print(dictionary.keys())
        print(dictionary.token2id)
        corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

        doc_test_vec = dictionary.doc2bow(doc_test_list)
        print(doc_test_vec)

        # the tfidf value of each word TF: term frequency TF-IDF = TF*IDF
        tfidf = models.TfidfModel(corpus)

        # theme matrix, which can be trained
        print(tfidf[doc_test_vec])

        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim = index[tfidf[doc_test_vec]]
        print(sim)
        sorted_similarity = sorted(enumerate(sim), key=lambda item: -item[1])

        self.__text_similarity = sorted_similarity[0][0] + 1
        print(sorted(enumerate(sim), key=lambda item: -item[1]))

    def detect_outlier_all(self):
        # 初始化用户选择
        self.__detect_outlier_single_format = False
        self.__detect_outlier_all_attributes = True
        self.__detect_outlier_text = False

        temp_outlier = []
        temp_outlier.clear()
        self.__rowWithOutlier.clear()

        # to check whether there is an auto-generated columns, if has, delete it.
        if "isOutlier" in list(self.__current_data_frame.columns):
            self.__current_data_frame.drop(["isOutlier"], axis=1, inplace=True)
        if "isDuplicate" in list(self.__current_data_frame.columns):
            self.__current_data_frame.drop(["isDuplicate"], axis=1, inplace=True)

        df = self.__current_data_frame

        # check whether there is an empty value begin
        header = df.columns.values
        print(header)
        for element in header:
            # print(df[element].isnull().sum())
            if df[element].isnull().sum() != 0:
                print("this dataframe has an null value, the program will delete the rows of automatically"
                      "if you want to keep your rows, please click revert button")
        # check whether there is an empty value ends

        df_temp = copy.deepcopy(df)

        # deal with missing values
        df_temp.fillna(df_temp.mean(), inplace=True)
        # deal with missing values

        number_of_column = df_temp.columns.size
        number_of_deleted_column = 0

        for i in range(0, number_of_column):
            if df.iloc[:, i].dtypes == "float64" or df.iloc[:, i].dtypes == "int64":
                print("it is a number, pass")
            elif df.iloc[:, i].dtypes == "object":
                print("it is an object, change it into matrix and add it into the last column")
                df_text = df.iloc[:, i]

                # text vectorazation begins
                array_text = df_text.values
                array_text2 = numpy.array(df_text)
                # print(array_text2.flatten())
                vectorizer = CountVectorizer()

                arranged_text = vectorizer.fit_transform(array_text2.flatten())

                transformer = TfidfTransformer()
                arranged_text_tfidf = transformer.fit_transform(arranged_text)

                # print(vectorizer.get_feature_names())
                # print(arranged_text.toarray())

                # print(arranged_text_tfidf.toarray())

                array_tfidf = arranged_text_tfidf.toarray()
                # text categorization ends

                # drop the original column begins
                df_temp.drop(df.columns[i - number_of_deleted_column], axis=1, inplace=True)

                number_of_deleted_column += 1
                # drop the original column ends

                # add the array_tfidf to the back of the dataframe begins
                dataframe_tfidf = pd.DataFrame(array_tfidf)
                # print(dataframe_tfidf)

                new_column = dataframe_tfidf.columns.values
                # print(new_column)
                df_temp[new_column] = dataframe_tfidf
                # print(df_temp)
                # add the array_tfidf to the back of the dataframe ends

        # use unsupervised categorization algorithms to find the outlier begins
        model = KMeans(n_clusters=5)
        model.fit(df_temp)
        predicted_label = model.predict(df_temp)
        print("tfidf", predicted_label)
        count_predicted_label = Counter(predicted_label)
        count_sorted = sorted(count_predicted_label.items(), key=lambda x: x[1])
        print("Counter_sorted", count_sorted)
        count_sorted_values = sorted(count_predicted_label.values())

        items_number = 0
        # decide get the first n element
        while items_number < len(count_sorted_values) - 1:
            if count_sorted_values[items_number] == count_sorted_values[items_number + 1]:
                items_number += 1
            else:
                break

        output_position = []

        for i in range(0, items_number + 1):
            print(count_sorted[i][0])
            target_position = numpy.argwhere(predicted_label == count_sorted[i][0])
            output_position.append(target_position[0])
        self.__current_data_frame['isOutlier'] = False
        for elements in output_position:
            for element in elements:
                self.__current_data_frame.loc[element, "isOutlier"] = True


        temp_outlier = self.__current_data_frame[self.__current_data_frame['isOutlier'] == True].index.tolist()
        self.__rowWithOutlier = [i + 1 for i in temp_outlier]
        self.__choice_in_detect_outlier = self.__current_data_frame.columns.get_loc("Column A")
        self.__current_data_frame = self.__current_data_frame.drop('isOutlier', axis=1)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)