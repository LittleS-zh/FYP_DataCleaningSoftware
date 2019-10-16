import pandas as pd
import copy
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from collections import Counter

df = pd.read_csv("testFile_WithoutNaN.csv")

# check whether there is an empty value begin
if True == True:
    print("this dataframe has an null value, the program will delete the rows of automatically"
          "if you want to keep your rows, please click revert button")
# check whether there is an empty value ends

df_temp = copy.deepcopy(df)

# deal with missing values
df_temp.fillna(df_temp.mean(), inplace= True)
# deal with missing values

number_of_column = df_temp.columns.size
number_of_deleted_column = 0

for i in range(0, number_of_column):
    if df.iloc[:, i].dtypes == "float64" or df.iloc[:, i].dtypes == "int64":
        print("it is a number, pass")
    elif df.iloc[:, i].dtypes == "object":
        print("it is an object, change it into matrix and add it into the last column")
        df_text = df.iloc[:, i]

        # text vectorazation begins
        array_text = df_text.values
        array_text2 = numpy.array(df_text)
        # print(array_text2.flatten())
        vectorizer = CountVectorizer()

        arranged_text = vectorizer.fit_transform(array_text2.flatten())

        transformer = TfidfTransformer()
        arranged_text_tfidf = transformer.fit_transform(arranged_text)

        # print(vectorizer.get_feature_names())
        # print(arranged_text.toarray())

        # print(arranged_text_tfidf.toarray())

        array_tfidf = arranged_text_tfidf.toarray()
        # text categorization ends

        # drop the original column begins
        df_temp.drop(df.columns[i - number_of_deleted_column], axis=1, inplace=True)

        number_of_deleted_column += 1
        # drop the original column ends

        # add the array_tfidf to the back of the dataframe begins
        dataframe_tfidf = pd.DataFrame(array_tfidf)
        # print(dataframe_tfidf)

        new_column = dataframe_tfidf.columns.values
        # print(new_column)
        df_temp[new_column] = dataframe_tfidf
        # print(df_temp)
        # add the array_tfidf to the back of the dataframe ends


# use unsupervised categorization algorithms to find the outlier begins
model = KMeans(n_clusters=5)
model.fit(df_temp)
predicted_label = model.predict(df_temp)
print("tfidf",predicted_label)
count_predicted_label = Counter(predicted_label)
count_sorted = sorted(count_predicted_label.items(), key=lambda x:x[1])
print("Counter_sorted", count_sorted)
count_sorted_values = sorted(count_predicted_label.values())

items_number = 0
# decide get the first n element
while items_number < len(count_sorted_values) - 1:
    if count_sorted_values[items_number] == count_sorted_values[items_number+1]:
        items_number += 1
    else:
        break

print("items_number", items_number+1)

output_position = []

for i in range (0,items_number+1):
    print(count_sorted[i][0])
    target_position = numpy.argwhere(predicted_label == count_sorted[i][0])
    output_position.append(target_position[0])
print(output_position)
for elements in output_position:
    for element in elements:
        print("Outlier element",element)
# use unsupervised categorization algorithms to find the outlier ends