import pandas as pd
import numpy as np

# wrong method
df = pd.DataFrame(np.arange(12).reshape(3,4),
           columns=['A', 'B', 'C', 'D'])
x=[1,2]
df.drop(index=[1,2], axis=1, inplace=True) #axis=1，试图指定列，然并卵
print(df)

# good method
df = pd.DataFrame(np.arange(12).reshape(3,4),
           columns=['A', 'B', 'C', 'D'])
x=[1,2]
df.drop(df.columns[x], axis=1, inplace=True)
print(df)