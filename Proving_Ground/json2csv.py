import pandas as pd

list_question = []
list_answer = []
list_answer_starts = []
list_is_impossible = []
list_context = []
list_title = []

df = pd.read_json("train-v2.0.json",orient='records')
df.drop(['version'], axis = 1, inplace = True)

for elements in df['data']:
    for element in elements['paragraphs']:
        for i in element['qas']:
            list_question.append(i['question'])

for elements in df['data']:
    for element in elements['paragraphs']:
        for i in element['qas']:
            list_is_impossible.append(i['is_impossible'])

for elements in df['data']:
    for element in elements['paragraphs']:
        for i in element['qas']:
            list_context.append(element['context'])
            list_title.append(elements['title'])
            if i['is_impossible'] == True:
                list_answer.append("None")
                list_answer_starts.append(0)
            else:
                list_answer.append(i['answers'][0]['text'])
                list_answer_starts.append(i['answers'][0]['answer_start'])

df_combine = pd.DataFrame(list_question,columns=['questions'])
df_combine_temp = pd.DataFrame(list_is_impossible, columns = ['is_impossible'])
df_combine['is_impossible'] = df_combine_temp['is_impossible']
df_combine_temp = pd.DataFrame(list_answer, columns = ['answers'])
df_combine['answers'] = df_combine_temp['answers']
df_combine_temp = pd.DataFrame(list_answer_starts, columns = ['starts'])
df_combine['starts'] = df_combine_temp['starts']
df_combine_temp = pd.DataFrame(list_context, columns = ['context'])
df_combine['context'] = df_combine_temp['context']
df_combine_temp = pd.DataFrame(list_title, columns = ['title'])
df_combine['title'] = df_combine_temp['title']

print(df_combine)

df_combine.to_csv("SQuAD2.0.csv",encoding = 'utf-8')