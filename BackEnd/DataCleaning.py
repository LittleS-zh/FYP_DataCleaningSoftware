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

def operationOnBlocks(DataFrame):
    DataFrame['newColumn'] = DataFrame['Day'] * DataFrame['Return']
    return DataFrame

def FilterAccordingToCondiction(DataFrame,Column_input):
    FiltedData = DataFrame[(DataFrame[Column_input] > 1)]
    return FiltedData

#Test Codes
DataFrame = readDataFile("DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)

writeDataFile(DataFrame, "DataSet_Write/1 XAGUSD_QPR_Result.csv")

rows = selectRows(DataFrame,0,3)
writeDataFile(rows, "DataSet_Write/1 XAGUSD_QPR_rows.csv")

columns = selectColumn_position(DataFrame,1,2)
writeDataFile(columns, "DataSet_Write/1 XAGUSD_QPR_columns_position.csv")

columns = selectColumn_heading(DataFrame,['Day','Return'])
writeDataFile(columns, "DataSet_Write/1 XAGUSD_QPR_columns_heading.csv")

NewDataFrame = operationOnBlocks(DataFrame)
writeDataFile(NewDataFrame, "DataSet_Write/1 XAGUSD_QPR_NewDataFrame.csv")

FiltedData = FilterAccordingToCondiction(DataFrame,"Return")
writeDataFile(FiltedData, "DataSet_Write/1 XAGUSD_QPR_FiltedData.csv")

writeDataFile(DataFrame.describe(), "DataSet_Write/1 XAGUSD_QPR_Describe.csv")

