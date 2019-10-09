import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from collections import Counter

df = pandas.read_csv("testFile_withoutNaN.csv")
df = df.iloc[:,0:4]

model = KMeans(n_clusters=5)
model.fit(df)
predicted_label = model.predict(df)
print(predicted_label)

count_predicted_label = Counter(predicted_label)
count_sorted = sorted(count_predicted_label.items(), key=lambda x:x[1])
print("Counter_sorted",count_sorted)
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