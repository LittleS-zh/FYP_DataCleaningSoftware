import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# read datafile from csv
def read_data_file(
        file_location, delimiter_input,
        encoding_input, header_input):
    data_frame = pd.read_csv(file_location, delimiter=delimiter_input, encoding=encoding_input, header=header_input)
    # print(data_frame)
    return data_frame


# write datafile to an csv file
def write_data_file(data_frame, path):
    data_frame.to_csv(path)


# TODO: add more parameters in it

def select_rows(data_file, row_start_input, row_end_input):
    rows_output = data_file[row_start_input:row_end_input]
    return rows_output


def select_column_position(data_frame, column_start_input, column_end_input):
    column = data_frame.iloc[:, [column_start_input, column_end_input]]
    return column


def select_column_heading(data_frame, header_of_column_input):
    column = data_frame[header_of_column_input]
    return column


def block_selection(data_frame, column_start_input, column_end_input, header_of_column_input):
    block_data = data_frame.ix[column_start_input:column_end_input, header_of_column_input]
    return block_data


def algorithm_operation_on_blocks(data_frame):
    data_frame['newColumn'] = data_frame['Day'] * data_frame['Return']
    return data_frame


def conditional_filter(data_frame, column_input):
    filtered_data = data_frame[(data_frame[column_input] > 1)]
    return filtered_data


# Max
def data_reduction(data_frame, header, drop_header, group_header, sum_or_mean, combine_name):
    if sum_or_mean == 1:
        reductive_data = data_frame.drop(drop_header, axis=1).groupby(group_header).sum().sort_values(combine_name,
                                                                                                      ascending=False)
    else:
        reductive_data = data_frame.drop(drop_header, axis=1).groupby(group_header).mean().sort_values(combine_name,
                                                                                                       ascending=False)
    return reductive_data


def data_de_duplication(data_frame, column_input):
    deduplicated_data = data_frame.drop_duplicates(column_input)
    return deduplicated_data


def calculate_means(data_frame, column_need_to_be_grouped_input, column_group_by):
    grouped = data_frame[column_need_to_be_grouped_input].groupby(data_frame[column_group_by])
    return grouped.mean()


def detect_outlier(data_frame,column_input):
    d = data_frame[column_input]
    z_score = (d - d.mean()) / d.std()
    data_frame['isOutlier'] = z_score.abs() > 3
    return data_frame


def detect_outlier_quantile(data_frame,column_input):
    d = data_frame[column_input]
    data_frame['isOutlier'] = d > d.quantitile(0.75)
    return data_frame

# ———————————————————— Data Cleaning Function ————————————————————#
def check_missing(file_name, values, index, column):
    data_frame = pd.pivot_table(file_name, values=values, index=index, columns=column)
    return data_frame.isnull()


def deal_with_missing_value(file_name, choice):
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


# ———————————————————— END ————————————————————#

# TODO: revert operation, you need to define a global dataframe variables
# Test Codes
DataFrame = read_data_file("DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)
# hzWeather = pd.read_csv("DataSet_Read/hz_weather.csv")

write_data_file(DataFrame, "DataSet_Write/1 XAGUSD_QPR_Result.csv")

rows = select_rows(DataFrame, 0, 3)
write_data_file(rows, "DataSet_Write/1 XAGUSD_QPR_rows.csv")

columns = select_column_position(DataFrame, 1, 2)
write_data_file(columns, "DataSet_Write/1 XAGUSD_QPR_columns_position.csv")

columns = select_column_heading(DataFrame, ['Day', 'Return'])
write_data_file(columns, "DataSet_Write/1 XAGUSD_QPR_columns_heading.csv")

# Max test
# blocks = block_selection(DataFrame, 0, 3, ['Day', 'Return'])
# write_data_file(blocks, "DataSet_Write/test.csv")
#
# NewDataFrame = algorithm_operation_on_blocks(DataFrame)
# write_data_file(NewDataFrame, "DataSet_Write/1 XAGUSD_QPR_NewDataFrame.csv")
#
# FilteredData = conditional_filter(DataFrame, "Return")
# write_data_file(FilteredData, "DataSet_Write/1 XAGUSD_QPR_FiltedData.csv")

data_frame_with_outlier = detect_outlier(DataFrame,"Return")
write_data_file(data_frame_with_outlier, "DataSet_Write/1 XAGUSD_QPR_Outlier.csv")

# detect outlier using box diagram
data_frame_with_outlier = detect_outlier(DataFrame,"Return")
write_data_file(data_frame_with_outlier, "DataSet_Write/1 XAGUSD_QPR_Outlier_quantile.csv")

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