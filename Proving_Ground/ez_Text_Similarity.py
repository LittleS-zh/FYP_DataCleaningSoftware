import jieba
import pandas
import numpy
from gensim import corpora,models,similarities

df_text = pandas.read_csv("tfidf_test.csv")

df_text = df_text['Sentences']

doc_test="I love Shanghai"

temp_array_text = numpy.array(df_text)
array_text = temp_array_text.flatten()

all_doc_list = []
for doc in array_text:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

print(all_doc_list)

doc_test_list = [word for word in jieba.cut(doc_test)]
print(doc_test_list)

# make a corpus
dictionary = corpora.Dictionary(all_doc_list)
print(dictionary.keys())
print(dictionary.token2id)
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

doc_test_vec = dictionary.doc2bow(doc_test_list)
print(doc_test_vec)

# the tfidf value of each word TF: term frequency TF-IDF = TF*IDF
tfidf = models.TfidfModel(corpus)

# theme matrix, which can be trained
print(tfidf[doc_test_vec])

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]
print(sim)

print(sorted(enumerate(sim), key=lambda item: -item[1]))

