import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import itertools

okt = Okt()

with open('./dataset/stopwords.word.txt', encoding='utf-8') as sw:
    stopwords_raw = sw.readlines()

stopwords = []
for i in stopwords_raw:
    stopwords.append(i.replace('\n', ''))

removable = ['$', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i in removable:
    stopwords.remove(i)

#def data_tokenizer(list_data):
#    token_nstops = []
#        i = 0
#    for text, ques, answ in list_data:
#        text_token = okt.morphs(text)
#        st_rm_text = [word_tx for word_tx in text_token if word_tx not in stopwords]
#        ques_token = okt.morphs(ques)
#        st_rm_ques = [word_qu for word_qu in ques_token if word_qu not in stopwords]
#        answ_token = okt.morphs(str(answ))
#        st_rm_answ = [word_an for word_an in answ_token if word_an not in stopwords]
#        token_nstops.append([text_token, ques_token, answ_token])
#
#        print(i)
#        i = i+1
#    return token_nstops

tokenized_data = data_tokenizer(filtered_data)
# df_t_d = pd.DataFrame(tokenized_data, columns = ['text', 'ques', 'answ'])
# df_t_d.to_excel('tokenized_data.xlsx', encoding='utf-8-sig')
# df_t_d.to_csv('tokenized_data.csv', encoding='utf-8-sig', sep='\t')
# df_t_d = pd.read_excel('tokenized_data.xlsx', index_col = 0)
# text_list = df_t_d['text'].values
# text_ques = df_t_d['ques'].values
# ext_answ = df_t_d['answ'].values


df_filtered = pd.DataFrame(filtered_data, columns = ['text', 'ques','answ'])
df_filtered.isnull().sum()
df_filtered = df_filtered[~(df_filtered['answ'].isnull())]



text_list = []
ques_list = []
answ_list = []

#불용어 처리
for text, ques, answ in filtered_data_nn:
    text_split = re.split('\s', text)
    text = [w_te for w_te in text_split if w_te not in stopwords]
    ques_split = re.split('\s', ques)
    ques = [w_te for w_te in ques_split if w_te not in stopwords]
    answ_split = re.split('\s', answ)
    answ = [w_te for w_te in answ_split if w_te not in stopwords]

    text_list.append(text)
    ques_list.append(ques)
    answ_list.append(answ)

# text, question, answer의 길이를 추출
t_len = []
q_len = []
a_len = []
for i in range(len(text_list)):
    t_l = len(text_list[i])
    q_l = len(ques_list[i])
    a_l = len(answ_list[i])
    t_len.append(t_l)
    q_len.append(q_l)
    a_len.append(a_l)

df_len = pd.DataFrame()
df_len['text'] = t_len
df_len['ques'] = q_len
df_len['answ'] = a_len
df_len.describe()

#일부 데이터만 추출
df_len[(df_len['text'] <= 600) & (df_len['ques'] <= 6) & (df_len['answ'] <=30)]
len(df_len[(df_len['text'] <= 500) & (df_len['ques'] <= 10) & (df_len['answ'] <=30)])
data_index = list(df_len[(df_len['text'] <= 500) & (df_len['ques'] <= 10) & (df_len['answ'] <=30)].index)

df_ppd = pd.DataFrame({'text' : text_list, 'ques' : ques_list, 'answ' : answ_list})
df_ppd = df_ppd.iloc[data_index, :].reset_index()
df_ppd.reset_index(drop = True, inplace = True)

all_data = []
for ppd_i in range(len(df_ppd)):
    all_data.append(list(df_ppd.iloc[ppd_i, :]))

vocab = set()
for text, ques, answ in all_data:
    vocab = vocab.union(set(text))
    vocab = vocab.union(set(ques))
    vocab = vocab.union(set(answ))

vocab_len = len(vocab) + 1
max_text_len = max([len(word) for word in df_ppd['text']])
max_ques_len = max([len(word) for word in df_ppd['ques']])
max_answ_len = max([len(word) for word in df_ppd['answ']])

tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)
vocab_len = len(tokenizer.word_index) + 1

train_text_seq = tokenizer.texts_to_sequences(df_ppd['text'])

def vectorize_stories(data, word_index=tokenizer.word_index, max_text_len=max_text_len, max_ques_len=max_ques_len):
    X = []
    Xq = []
    Y = []
    for story, question, answer in data:
        x = []
        for word_s in story:
            try:
                x.append(word_index[word_s])
            except KeyError:
                continue

        xq = []
        for word_q in question:
            try:
                xq.append(word_index[word_q])
            except KeyError:
                continue

        y = np.zeros(len(word_index) + 1)
        for word_a in answer:
            y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_text_len), pad_sequences(Xq, maxlen=max_ques_len), pad_sequences(Y, maxlen=max_answ_len))
input_train, question_train, answer_train = vectorize_stories(df_ppd.values.tolist())