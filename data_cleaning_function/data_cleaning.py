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

    # this attribute is for indicating the rows with outlier
    __rowWithOutlier = []

    def __init__(self):
        print("class initialization")

    # read datafile from csv
    def read_data_file(self, file_location,
                       delimiter_input=",", encoding_input="utf-8",
                       header_input=0):
        self.__current_data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input,
                                                header=header_input)
        self.__original_data_frame = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__current_data_frame)


        # for debug
        print(self.__current_data_frame.shape[0])
        print(self.__current_data_frame.shape[1])

    # write datafile to an csv file
    def write_data_file(self, path):
        self.__current_data_frame.to_csv(path)

    # select rows for a data frame
    def select_rows(self, row_start_input, row_end_input):
        if row_start_input <= row_end_input and row_end_input <= self.__current_data_frame.shape[0] and row_start_input >= 1:
            print(self.__current_data_frame.shape[0])
            self.__current_data_frame = self.__current_data_frame[row_start_input-1:row_end_input]
            self.__list_data_frame.append(self.__current_data_frame)
        else:
            print("invalid input")

    # select columns for a data frame, according to the start number and the end number
    def select_column_position(self, column_start_input, column_end_input):
        if column_start_input <= column_end_input and column_end_input <= self.__current_data_frame.shape[1] and column_start_input >= 1:
            print(self.__current_data_frame.shape[1])
            self.__current_data_frame = self.__current_data_frame.iloc[:, column_start_input-1: column_end_input]
            self.__list_data_frame.append(self.__current_data_frame)
        else:
            print("invalid input")

    # select columns for a data frame, according to the header of column
    def select_column_heading(self, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame[header_of_column_input]
        self.__list_data_frame.append(self.__current_data_frame)

    # select row and column at the same time
    def block_selection(self, column_start_input, column_end_input, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame.ix[column_start_input:column_end_input,
                                    header_of_column_input]
        self.__list_data_frame.append(self.__current_data_frame)

    # create a new column according to the operation based on two previous columns
    def algorithm_operation_on_blocks(self):
        self.__current_data_frame['newColumn'] = self.__current_data_frame['Day'] * self.__current_data_frame['Return']

    # filter the rows according to a specific value
    def conditional_filter(self, column_input):
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == True)]
        self.__list_data_frame.append(self.__current_data_frame)

    def data_reduction(self, drop_header, group_header, sum_or_mean, combine_name):
        if sum_or_mean == 1:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).sum().sort_values(combine_name,
                                                ascending=False)
            self.__list_data_frame.append(self.__current_data_frame)

        else:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).mean().sort_values(combine_name,
                                                 ascending=False)
            self.__list_data_frame.append(self.__current_data_frame)

    # deduplicate data
    def data_de_duplication(self):
        self.__current_data_frame = self.__current_data_frame.drop_duplicates()
        self.__list_data_frame.append(self.__current_data_frame)

    # calculate means of data
    def calculate_means(self, column_need_to_be_grouped_input, column_group_by):
        self.__current_data_frame = self.__current_data_frame[column_need_to_be_grouped_input].groupby(
            self.__current_data_frame[column_group_by])
        self.__list_data_frame.append(self.__current_data_frame)
        return self.__current_data_frame.mean()

    # detect outlier using three sigma method
    def detect_outlier_three_sigma(self, column_input):
        d = self.__current_data_frame[column_input]
        z_score = (d - d.mean()) / d.std()
        self.__current_data_frame['isOutlier'] = z_score.abs() > 3
        __rowWithOutlier = self.__current_data_frame[self.__current_data_frame['isOutlier'] == True].index.tolist()
        print(__rowWithOutlier)

    # detect outlier using box-plot
    def detect_outlier_quantitile(self, column_input):
        d = self.__current_data_frame[column_input]
        self.__current_data_frame['isOutlier'] = d > d.quantitile(0.75)

    # delete the whole row of outlier using three sigma methods
    def deal_with_outlier(self, column_input):
        print(column_input)
        self.detect_outlier_three_sigma(column_input)
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == False)]
        self.__current_data_frame = self.__current_data_frame.drop("isOutlier",axis=1)
        self.__list_data_frame.append(self.__current_data_frame)

    # check missing value
    def check_missing(self):
        data_frame_column = np.array(self.__current_data_frame.columns)
        data_frame_array = np.array(self.__current_data_frame.isnull())
        data_frame_list = data_frame_array.tolist()
        data_dictionary = {'data_frame': data_frame_list, 'data_header': data_frame_column}
        return data_dictionary

    # delete missing value
    def deal_with_missing_value(self, choice):
        # delete the row
        print("Choice is " + choice)
        print(choice == "1")
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
        self.__list_data_frame.append(self.__current_data_frame)

    # change the data frame into a list and return to the front end for showing
    def get_frame(self, float_round):
        data_frame_column = np.array(self.__current_data_frame.columns)
        data_frame_array = np.array(self.__current_data_frame.round(float_round))
        data_frame_list = data_frame_array.tolist()
        data_frame_list.insert(0,data_frame_column)
        data_dictionary = {'data_frame': data_frame_list, 'data_header':data_frame_column, 'outlier':self.__rowWithOutlier}
        return data_dictionary

    # reset function
    def reset_data_frame(self):
        self.__current_data_frame = copy.deepcopy(self.__original_data_frame)

    # revert function
    # todo: limit the times that a user can revert
    def revert_data_frame(self):
        self.__list_data_frame.pop()
        self.__current_data_frame = copy.deepcopy(self.__list_data_frame[-1])

    def forecast_a_value(self,target_column_input,target_column_forecast,target_row_input,target_row_output):
        # todo: parameters change here
        X = self.__current_data_frame.iloc[:, target_column_input]
        norm =100
        X = np.array(X)
        X = X.astype(float)
        X = X / norm
        print("dataframe of X is: ")
        print(X)

        Y = self.__current_data_frame.iloc[:, 2:3]
        Y = np.array(Y)
        Y = Y.astype(float)
        Y = Y / norm
        print("dataframe of Y is: ")
        print(Y)

        # get the liens before 16 as training set
        X_train, Y_train = X[:15], Y[:15]
        print("dataframe of x_train is: ")
        print(X_train)

        print("dataframe of y_train is: ")
        print(Y_train)

        # get the line after 16 as testing set
        X_test, Y_test = X[15:], Y[15:]
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
        model.fit(X_train, Y_train, epochs=5, batch_size=15)

        prediction = model.predict(X_test, batch_size=1)

        print(prediction * norm)

    def find_outlier_of_text(self, input_column_which_is_text):

        #todo test this statement, this statement may have some problems
        df_text = self.__current_data_frame.iloc[:,input_column_which_is_text]

        array_text = df_text.values
        array_text2 = numpy.array(df_text)
        print(array_text2.flatten())
        vectorizer = CountVectorizer()

        arranged_text = vectorizer.fit_transform(array_text2.flatten())

        transformer = TfidfTransformer()
        arranged_text_tfidf = transformer.fit_transform(arranged_text)

        print(vectorizer.get_feature_names())
        print(arranged_text.toarray())

        print(arranged_text_tfidf.toarray())

        array_tfidf = arranged_text_tfidf.toarray()

        model = KMeans(n_clusters=5)
        model.fit(array_tfidf)
        predicted_label = model.predict(array_tfidf)
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

        print("items_number", items_number + 1)

        output_position = []

        for i in range(0, items_number + 1):
            print(count_sorted[i][0])
            target_position = numpy.argwhere(predicted_label == count_sorted[i][0])
            output_position.append(target_position[0])
        print(output_position)
        print(output_position[1])
