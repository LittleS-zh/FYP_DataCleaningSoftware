import pandas as pd

df = pd.read_csv("testFile_WithoutNaN.csv")

loc = "df = df[df['Column A'] == 2]"
exec(loc)

print(df)