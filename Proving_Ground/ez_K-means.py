from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris_df = datasets.load_iris()

# print(iris_df)

model = KMeans(n_clusters=3)

model.fit(iris_df.data)

predicted_label = model.predict(iris_df.data)

print(predicted_label)