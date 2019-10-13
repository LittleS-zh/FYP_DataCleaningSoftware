import pandas as pd
import numpy

df = pd.read_csv("testFile_WithoutNaN.csv")

number_of_column = df.columns.size

# problems: if user use the column named "column1isduplicate", the program will get an error

for i in range(0, number_of_column):
    print(i)
    print(df.iloc[:, i].dtypes)
    if df.iloc[:, i].dtypes == "float64" or df.iloc[:, i].dtypes == "int64":
        df["column" + str(i+1) + "isDuplicate"] = df.iloc[:, i].duplicated()
        print("Numbers")
    elif df.iloc[:, i].dtypes == "object":
        print("object")

print(df)