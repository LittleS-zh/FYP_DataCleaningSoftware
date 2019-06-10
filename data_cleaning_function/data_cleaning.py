import pandas as pd
import numpy as np


class DataCleaning(object):
    __current_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    # this attribute is for reverting the operations
    __original_data_frame = pd.read_csv("static/DataSet_Read/basic.csv")

    def __init__(self):
        print("class initialization")

    # read datafile from csv
    def read_data_file(self, file_location,
                       delimiter_input=",", encoding_input="utf-8",
                       header_input=0):
        self.__current_data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input,
                                                header=header_input)
        self.__original_data_frame = self.__current_data_frame

    # write datafile to an csv file
    def write_data_file(self, path):
        self.__current_data_frame.to_csv(path)

    # TODO: add more parameters in it

    def select_rows(self, row_start_input, row_end_input):
        self.__current_data_frame = self.__current_data_frame[row_start_input:row_end_input]

    def select_column_position(self, column_start_input, column_end_input):
        self.__current_data_frame = self.__current_data_frame.iloc[:, [column_start_input, column_end_input]]

    def select_column_heading(self, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame[header_of_column_input]

    def block_selection(self, column_start_input, column_end_input, header_of_column_input):
        self.__current_data_frame = self.__current_data_frame.ix[column_start_input:column_end_input,
                                    header_of_column_input]

    def algorithm_operation_on_blocks(self):
        self.__current_data_frame['newColumn'] = self.__current_data_frame['Day'] * self.__current_data_frame['Return']

    def conditional_filter(self, data_frame, column_input):
        self.__current_data_frame = self.__current_data_frame[(self.__current_data_frame[column_input] > 1)]

    # Max
    def data_reduction(self, drop_header, group_header, sum_or_mean, combine_name):
        if sum_or_mean == 1:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(group_header).sum().sort_values(combine_name,
                                                                                                          ascending=False)
        else:
            self.__current_data_frame = self.__current_data_frame.drop(drop_header, axis=1).groupby(group_header).mean().sort_values(combine_name,
                                                                                                           ascending=False)

    def data_de_duplication(self):
        self.__current_data_frame = self.__current_data_frame.drop_duplicates()

    def calculate_means(self, column_need_to_be_grouped_input, column_group_by):
        self.__current_data_frame = self.__current_data_frame[column_need_to_be_grouped_input].groupby(self.__current_data_frame[column_group_by])
        return self.__current_data_frame.mean()

    def detect_outlier_three_sigma(self, column_input):
        d = self.__current_data_frame[column_input]
        z_score = (d - d.mean()) / d.std()
        self.__current_data_frame['isOutlier'] = z_score.abs() > 3

    def detect_outlier_quantitile(self, column_input):
        d = self.__current_data_frame[column_input]
        self.__current_data_frame['isOutlier'] = d > d.quantitile(0.75)

    # TODO: to know how to change it to oo
    def check_missing(self, file_name, values, index, column):
        data_frame = pd.pivot_table(file_name, values=values, index=index, columns=column)
        return data_frame.isnull()

    def deal_with_missing_value(self, file_name, choice):
        # delete the row
        if choice == 1:
            return file_name.dropna(axis=0)
        elif choice == 2:
            return file_name.dropna(axis=1)
        elif choice == 3:
            instead_word = input("Please input a string to instead missing value!\n")
            return file_name.fillna(instead_word)
        elif choice == 4:
            return file_name.fillna(method='pad')
        elif choice == 5:
            return file_name.fillna(method='bfill', limit=1)
        elif choice == 6:
            return file_name.fillna(file_name.mean())

    def get_frame(self):
        data_frame_array = np.array(self.__current_data_frame)
        data_frame_list = data_frame_array.tolist()
        return data_frame_list

    def revert_data_frame(self):
        self.__current_data_frame = self.__original_data_frame

# Test Codes

# DataFrame = read_data_file("static/DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)

# DataFrame = read_data_file("DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)
# hzWeather = pd.read_csv("DataSet_Read/hz_weather.csv")

# write_data_file(DataFrame, "static/DataSet_Write/1 XAGUSD_QPR_Result.csv")
#
# rows = select_rows(DataFrame, 0, 3)
# write_data_file(rows, "DataSet_Write/1 XAGUSD_QPR_rows.csv")
#
# columns = select_column_position(DataFrame, 1, 2)
# write_data_file(columns, "DataSet_Write/1 XAGUSD_QPR_columns_position.csv")
#
# columns = select_column_heading(DataFrame, ['Day', 'Return'])
# write_data_file(columns, "DataSet_Write/1 XAGUSD_QPR_columns_heading.csv")

# Max test
# blocks = block_selection(DataFrame, 0, 3, ['Day', 'Return'])
# write_data_file(blocks, "DataSet_Write/test.csv")
#
# NewDataFrame = algorithm_operation_on_blocks(DataFrame)
# write_data_file(NewDataFrame, "DataSet_Write/1 XAGUSD_QPR_NewDataFrame.csv")
#
# FilteredData = conditional_filter(DataFrame, "Return")
# write_data_file(FilteredData, "DataSet_Write/1 XAGUSD_QPR_FiltedData.csv")

# data_frame_with_outlier = detect_outlier(DataFrame,"Return")
# write_data_file(data_frame_with_outlier, "DataSet_Write/1 XAGUSD_QPR_Outlier.csv")
#
# # detect outlier using box diagram
# data_frame_with_outlier = detect_outlier(DataFrame,"Return")
# write_data_file(data_frame_with_outlier, "DataSet_Write/1 XAGUSD_QPR_Outlier_quantile.csv")

# Max test
# Data = dataReduction(DataFrame, "Year", ['Open', 'High', 'Low'], "Year", 1, "RSI")
# writeDataFile(Data, "DataSet_Write/test2.csv")
#

# Max test
# CheckMissing = check_missing(hzWeather, "最高气温", "天气", "风向")
# write_data_file(CheckMissing, "DataSet_Write/CheckMissing.csv")
#
# df = pd.pivot_table(hzWeather, values=['最高气温'], index=['天气'], columns=['风向'])
# dealFile = deal_with_missing_value(df, 1)
# dealFile = deal_with_missing_value(df, 2)
# dealFile = deal_with_missing_value(df, 3)
# dealFile = deal_with_missing_value(df, 4)
# dealFile = deal_with_missing_value(df, 5)
# dealFile = deal_with_missing_value(df, 6)
# write_data_file(dealFile, "DataSet_Write/DealMissing.csv")
#
