import pandas as pd

df = pd.read_csv("testFile_WithoutNaN.csv")

number_of_column = df.columns.size

for i in range(0, number_of_column):
    print(df.iloc[:, i].dtypes)