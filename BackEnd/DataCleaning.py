import pandas as pd

#read datafile from csv
def readDataFile(FileLocation,delimiter_input,encoding_input,header_input):
    DataFrame = pd.read_csv(FileLocation,delimiter = delimiter_input,encoding = encoding_input, header = header_input)
    print(DataFrame)
    return DataFrame

#write datafile to an csv file
def writeDataFile(DataFrame, path):
    DataFrame.to_csv(path)
#TODO: add more parameters in it

def selectRows(DataFile,rowStart_input,rowEnd_input):
    rows = DataFile[rowStart_input:rowEnd_input]
    return rows

def selectColumn_position(DataFrame,columnStart_input,columnEnd_input):
    column = DataFrame.iloc[:,[columnStart_input,columnEnd_input]]
    return column

def selectColumn_heading(DataFrame,headerOfColumn_input):
    column = DataFrame[headerOfColumn_input]
    return column

# Max
def blockSelection(DataFrame, columnStart_input, columnEnd_input, headerOfColumn_input):
    BlockData = DataFrame.ix[columnStart_input:columnEnd_input, headerOfColumn_input]
    return BlockData
#

def operationOnBlocks(DataFrame):
    DataFrame['newColumn'] = DataFrame['Day'] * DataFrame['Return']
    return DataFrame

def FilterAccordingToCondiction(DataFrame,Column_input):
    FiltedData = DataFrame[(DataFrame[Column_input] > 1)]
    return FiltedData

#Max
def dataReduction(DataFrame, Header, DropHeader, GroupHeader, SumOrMean, CombineName):
    ReductionData = DataFrame.set_index(Header).sortlevel(0)
    if SumOrMean == 1:
        ReductiveData = DataFrame.drop(DropHeader, axis=1).groupby(GroupHeader).sum().sort_values(CombineName, ascending=False)
    else:
        ReductiveData = DataFrame.drop(DropHeader, axis=1).groupby(GroupHeader).mean().sort_values(CombineName, ascending=False)
    return ReductiveData

def dataDeduplication(DataFrame,Column_input):
    deduplicatedData = DataFrame.drop_duplicates(Column_input)
    return deduplicatedData

def CalculateMeans(DataFrame,columnNeedtoBeGrouped_input,columnGroupBy):
    grouped = DataFrame[columnNeedtoBeGrouped_input].groupby(DataFrame[columnGroupBy])
    return grouped.mean()

#———————————————————— Data Cleaning Function ————————————————————#
def checkMissing(fileName, values, index, columns):
    df = pd.pivot_table(fileName, values=values, index=index, columns=columns)
    return df.isnull()

def deal_with_missing_value(fileName, choice):
    # delete the row
    if choice == 1:
        return fileName.dropna(axis=0)
    elif choice == 2:
        return fileName.dropna(axis=1)
    elif choice == 3:
        insteadWord = input("Please input a string to instead missing value!\n")
        return fileName.fillna(insteadWord)
    elif choice == 4:
        return fileName.fillna(method='pad')
    elif choice == 5:
        return fileName.fillna(method='bfill', limit=1)
    elif choice == 6:
        return fileName.fillna(fileName.mean())

#———————————————————— END ————————————————————#


#TODO: revert operation, you need to define a global dataframe variables
#Test Codes
DataFrame = readDataFile("DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)
hzWeather = pd.read_csv("DataSet_Read/hz_weather.csv")

writeDataFile(DataFrame, "DataSet_Write/1 XAGUSD_QPR_Result.csv")

rows = selectRows(DataFrame,0,3)
writeDataFile(rows, "DataSet_Write/1 XAGUSD_QPR_rows.csv")

columns = selectColumn_position(DataFrame,1,2)
writeDataFile(columns, "DataSet_Write/1 XAGUSD_QPR_columns_position.csv")

columns = selectColumn_heading(DataFrame,['Day','Return'])
writeDataFile(columns, "DataSet_Write/1 XAGUSD_QPR_columns_heading.csv")

#Max test
blocks = blockSelection(DataFrame, 0, 3, ['Day', 'Return'])
writeDataFile(blocks, "DataSet_Write/test.csv")
#

NewDataFrame = operationOnBlocks(DataFrame)
writeDataFile(NewDataFrame, "DataSet_Write/1 XAGUSD_QPR_NewDataFrame.csv")

FiltedData = FilterAccordingToCondiction(DataFrame,"Return")
writeDataFile(FiltedData, "DataSet_Write/1 XAGUSD_QPR_FiltedData.csv")

#Max test
#Data = dataReduction(DataFrame, "Year", ['Open', 'High', 'Low'], "Year", 1, "RSI")
#writeDataFile(Data, "DataSet_Write/test2.csv")
#

#Max test
CheckMissing = checkMissing(hzWeather, "最高气温", "天气", "风向")
writeDataFile(CheckMissing, "DataSet_Write/CheckMissing.csv")

df = pd.pivot_table(hzWeather, values=['最高气温'], index=['天气'], columns=['风向'])
#dealFile = deal_with_missing_value(df, 1)
#dealFile = deal_with_missing_value(df, 2)
#dealFile = deal_with_missing_value(df, 3)
#dealFile = deal_with_missing_value(df, 4)
#dealFile = deal_with_missing_value(df, 5)
dealFile = deal_with_missing_value(df, 6)
writeDataFile(dealFile, "DataSet_Write/DealMissing.csv")
#

