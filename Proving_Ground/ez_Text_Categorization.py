import jieba
import gensim
from gensim import corpora,models,similarities

doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc8 = "欸哟我的鸡巴"
doc9 = "怪物猎人"
doc_test="我喜欢上海的小吃"

all_doc = []
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)
all_doc.append(doc5)
all_doc.append(doc6)
all_doc.append(doc7)
all_doc.append(doc8)
all_doc.append(doc9)

all_doc_list = []
for doc in all_doc:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

print(all_doc_list)

doc_test_list = [word for word in jieba.cut(doc_test)]
print("the doc test list is: ")
print(doc_test_list)

# make a corpus
dictionary = corpora.Dictionary(all_doc_list)
print("the dictionary keys are: ")
print(dictionary.keys())
print("the relationship between token and id are: ")
print(dictionary.token2id)
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]
print("The corpus is: ")
print(corpus)


doc_test_vec = dictionary.doc2bow(doc_test_list)
print("the test document vector is: ")
print(doc_test_vec)

# the tfidf value of each word TF: term frequency TF-IDF = TF*IDF
tfidf = models.TfidfModel(corpus)
print("the information of tfidf model is:")
print(tfidf)

# theme matrix, which can be trained
print("the tfidf information of all the document is: ")
print(tfidf[doc_test_vec])