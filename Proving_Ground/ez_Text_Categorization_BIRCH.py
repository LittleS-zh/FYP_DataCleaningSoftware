import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import birch
from collections import Counter

df_text = pandas.read_csv("tfidf_test.csv")

df_text = df_text['Sentences']

array_text = df_text.values
array_text2 = numpy.array(df_text)
print(array_text2.flatten())
vectorizer = CountVectorizer()

arranged_text = vectorizer.fit_transform(array_text2.flatten())

transformer = TfidfTransformer()
arranged_text_tfidf = transformer.fit_transform(arranged_text)

print(vectorizer.get_feature_names())
print(arranged_text.toarray())

print(arranged_text_tfidf.toarray())

array_tfidf = arranged_text_tfidf.toarray()

model = birch.Birch(n_clusters=5)
model.fit(array_tfidf)

predicted_label = model.predict(array_tfidf)
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


# model.fit(arranged_text)
# predicted_label = model.predict(arranged_text)
# print("normal",predicted_label)