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

    # indicate which operation that users are doing
    __detect_outlier_numbers = False
    __detect_outlier_text = False
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
        print("class initialization")

    # read datafile from csv
    def read_data_file(self, file_location,
                       delimiter_input=",", encoding_input="utf-8",
                       header_input=0):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__rowWithOutlier.clear()
        self.__missing_value_result.clear()
        self.__choice_in_detect_outlier = -1

        self.__current_data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input,
                                                header=header_input)
        self.__original_data_frame = copy.deepcopy(self.__current_data_frame)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)


        # for debug
        print(self.__current_data_frame.shape[0])
        print(self.__current_data_frame.shape[1])

    # write datafile to an csv file
    def write_data_file(self, path):
        self.__current_data_frame.to_csv(path)

    # select rows for a data frame
    def select_rows(self, row_start_input, row_end_input):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if row_start_input <= row_end_input and row_end_input <= self.__current_data_frame.shape[0] and row_start_input >= 1:
            print(self.__current_data_frame.shape[0])
            self.__current_data_frame = self.__current_data_frame[row_start_input-1:row_end_input]
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        else:
            print("invalid input")

    # select columns for a data frame, according to the start number and the end number
    def select_column_position(self, column_start_input, column_end_input):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        if column_start_input <= column_end_input and column_end_input <= self.__current_data_frame.shape[1] and column_start_input >= 1:
            print(self.__current_data_frame.shape[1])
            self.__current_data_frame = self.__current_data_frame.iloc[:, column_start_input-1: column_end_input]
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        else:
            print("invalid input")

    # select columns for a data frame, according to the header of column
    def select_column_heading(self, header_of_column_input):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
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
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
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
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
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

        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    def data_reduction(self, drop_header, group_header, sum_or_mean, combine_name):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
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
    def data_de_duplication(self):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False
        self.__check_missing_value = False
        self.__missing_value_result.clear()
        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

        self.__current_data_frame = self.__current_data_frame.drop_duplicates()
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    # calculate means of data
    def calculate_means(self, column_need_to_be_grouped_input, column_group_by):
        # 初始化
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
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
        # 初始化用户选择
        self.__detect_outlier_numbers = True
        self.__detect_outlier_all_attributes = False
        self.__detect_outlier_text = False

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
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

    def outlier_modification(self, modification_value, modification_row):
        # todo: get users' original floating point
        if modification_value.isdigit():
            self.__current_data_frame.loc[modification_row - 1, self.__column_detect_name] = float(modification_value)
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        else:
            self.__current_data_frame.loc[modification_row - 1, self.__column_detect_name] = modification_value
            self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
            self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        print(self.__list_data_frame)

    def single_outlier_delete(self, modification_row):
        self.__current_data_frame = self.__current_data_frame.drop(modification_row-1)
        self.__current_data_frame.index = range(len(self.__current_data_frame))

        for outlier_index in range(0, len(self.__rowWithOutlier)):
            if self.__rowWithOutlier[outlier_index] > modification_row:
                self.__rowWithOutlier[outlier_index] -= 1
        self.__rowWithOutlier.remove(modification_row)

        print(self.__current_data_frame)

    # detect outlier using box-plot
    # todo: this function is not finished
    def detect_outlier_quantitile(self, column_input):
        d = self.__current_data_frame[column_input]
        self.__current_data_frame['isOutlier'] = d > d.quantitile(0.75)

    # delete the whole row of outlier using three sigma methods
    def deal_with_outlier(self, column_input):
        self.detect_outlier_three_sigma(column_input)
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == False)]
        self.__current_data_frame = self.__current_data_frame.drop("isOutlier",axis=1)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)

        self.__current_data_frame.index = range(len(self.__current_data_frame))
        print(self.__current_data_frame)

        self.__rowWithOutlier.clear()
        self.__choice_in_detect_outlier = -1

    # check missing value
    def check_missing(self):
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False

        self.__check_missing_value = True
        self.__missing_value_result.clear()

        # data_frame_column = np.array(self.__current_data_frame.columns)
        # print("data_frame_column: ", data_frame_column)
        # data_frame_array = np.array(self.__current_data_frame.isnull())
        # print("data_frame_array: ", data_frame_array)
        self.__missing_value_result = self.__current_data_frame[self.__current_data_frame.isnull().values == True].index.tolist()
        print(self.__missing_value_result)
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)
        # data_frame_list = data_frame_array.tolist()
        # print("data_frame_list: ", data_frame_list)
        # data_dictionary = {'data_frame': data_frame_list, 'data_header': data_frame_column}
        # return data_dictionary

    # delete missing value
    def deal_with_missing_value(self, choice):
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False

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
                           'detect_outlier_numbers': self.__detect_outlier_numbers,
                           'detect_outlier_text': self.__detect_outlier_text,
                           'detect_outlier_all_attributes': self.__detect_outlier_all_attributes,
                           'data_outlier': self.__rowWithOutlier,
                           'detect_outlier_choice': self.__choice_in_detect_outlier,
                           'check_missing_value': self.__check_missing_value,
                           'missing_value_result': self.__missing_value_result,
                           'text_similarity': self.__text_similarity}
        return data_dictionary

    # reset function
    def reset_data_frame(self):
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False

        self.__rowWithOutlier.clear()
        self.__column_detect_name = ''
        self.__choice_in_detect_outlier = -1
        self.__current_data_frame = copy.deepcopy(self.__original_data_frame)

    # revert function
    def revert_data_frame(self):
        self.__detect_outlier_numbers = False
        self.__detect_outlier_text = False
        self.__detect_outlier_all_attributes = False

        self.__choice_in_detect_outlier = -1
        self.__rowWithOutlier.clear()
        if len(self.__list_data_frame) > 1:
            self.__list_data_frame.pop()
            self.__current_data_frame = copy.deepcopy(self.__list_data_frame[-1])

    def forecast_a_value(self,target_column_input,target_column_forecast,target_row_input,target_row_output):
        # todo: parameters change here
        X = self.__current_data_frame.iloc[:, target_column_input]
        norm = 100
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

    def detect_outlier_text(self, input_column_which_is_text):
        temp_outlier = []
        temp_outlier.clear()
        self.__rowWithOutlier.clear()
        self.__column_detect_name = input_column_which_is_text

        df_text = self.__current_data_frame[input_column_which_is_text]

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

        print("items_number", items_number + 1)

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
        print(input_column_which_is_text)
        print(self.__choice_in_detect_outlier)
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
        self.__detect_outlier_numbers = False
        self.__detect_outlier_all_attributes = True
        self.__detect_outlier_text = False

        temp_outlier = []
        temp_outlier.clear()
        self.__rowWithOutlier.clear()

        # todo: the selection part need to be changed
        df = self.__current_data_frame

        model = KMeans(n_clusters=5)
        model.fit(df)
        predicted_label = model.predict(df)
        print(predicted_label)

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
        self.__temp_data_frame_for_deepcopy = copy.deepcopy(self.__current_data_frame)
        self.__list_data_frame.append(self.__temp_data_frame_for_deepcopy)