import pandas as pd
import numpy as np
import copy


class DataCleaning(object):
    __current_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    # this attribute is for reverting the operations
    __original_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    __data_frame_header = []

    def __init__(self):
        print("class initialization")

    # read datafile from csv
    def read_data_file(self, file_location,
                       delimiter_input=",", encoding_input="utf-8",
                       header_input=0):
        self.__current_data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input,
                                                header=header_input)
        self.__original_data_frame = copy.deepcopy(self.__current_data_frame)

    # write datafile to an csv file
    def write_data_file(self, path):
        self.__current_data_frame.to_csv(path)

    def select_rows(self, row_start_input, row_end_input):
        if row_start_input < row_end_input:
            self.__current_data_frame = self.__current_data_frame[row_start_input-1:row_end_input]
        else:
            print("invalid input")

    def select_column_position(self, column_start_input, column_end_input):
        self.__current_data_frame = self.__current_data_frame.iloc[:, column_start_input-1: column_end_input]

    def select_column_heading(self, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame[header_of_column_input]

    def block_selection(self, column_start_input, column_end_input, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame.ix[column_start_input:column_end_input,
                                    header_of_column_input]

    def algorithm_operation_on_blocks(self):
        self.__current_data_frame['newColumn'] = self.__current_data_frame['Day'] * self.__current_data_frame['Return']

    def conditional_filter(self, column_input):
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == True)]

    def data_reduction(self, drop_header, group_header, sum_or_mean, combine_name):
        if sum_or_mean == 1:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).sum().sort_values(combine_name,
                                                ascending=False)
        else:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(
                group_header).mean().sort_values(combine_name,
                                                 ascending=False)

    def data_de_duplication(self):
        self.__current_data_frame = self.__current_data_frame.drop_duplicates()

    def calculate_means(self, column_need_to_be_grouped_input, column_group_by):
        self.__current_data_frame = self.__current_data_frame[column_need_to_be_grouped_input].groupby(
            self.__current_data_frame[column_group_by])
        return self.__current_data_frame.mean()

    def detect_outlier_three_sigma(self, column_input):
        d = self.__current_data_frame[column_input]
        z_score = (d - d.mean()) / d.std()
        self.__current_data_frame['isOutlier'] = z_score.abs() > 3


    def detect_outlier_quantitile(self, column_input):
        d = self.__current_data_frame[column_input]
        self.__current_data_frame['isOutlier'] = d > d.quantitile(0.75)

    def deal_with_outlier(self, column_input):
        print(column_input)
        self.detect_outlier_three_sigma(column_input)
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame['isOutlier'] == False)]
        self.__current_data_frame = self.__current_data_frame.drop("isOutlier",axis=1)

    def check_missing(self):
        data_frame_column = np.array(self.__current_data_frame.columns)
        data_frame_array = np.array(self.__current_data_frame.isnull())
        data_frame_list = data_frame_array.tolist()
        data_dictionary = {'data_frame': data_frame_list, 'data_header': data_frame_column}
        return data_dictionary

    def deal_with_missing_value(self, choice):
        # delete the row
        print("Choice is " + choice)
        print(choice=="1")
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

    def get_frame(self,float_round):
        data_frame_column = np.array(self.__current_data_frame.columns)
        data_frame_array = np.array(self.__current_data_frame.round(float_round))
        data_frame_list = data_frame_array.tolist()
        data_frame_list.insert(0,data_frame_column)
        data_dictionary = {'data_frame': data_frame_list, 'data_header':data_frame_column}
        return data_dictionary

    def revert_data_frame(self):
        self.__current_data_frame = copy.deepcopy(self.__original_data_frame)