import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df_text = pandas.read_csv("tfidf_test.csv")

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

