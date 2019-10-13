import pandas as pd
import copy

df = pd.read_csv("testFile_WithoutNaN.csv")

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

print(df_temp)
print(df)