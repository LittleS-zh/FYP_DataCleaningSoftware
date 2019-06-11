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