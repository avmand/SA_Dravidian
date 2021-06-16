
"""

Citation:
@inproceedings{ bitsnlp2021dravidian,
 title={Sentiment Analysis of Dravidian Code Mixed Data},
 author={Mandalam, Asrita Venkata and Sharma, Yashvardhan},
 booktitle={ 15th Conference of the European Chapter of the Association for Computational Linguistics 2021 (EACL 2021) },
 year={2021}
}  

"""

import numpy as np
import pandas as pd
import re
from keras.layers import LSTM, Dense, Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from keras.models import Model
from keras.layers import BatchNormalization, Activation


def token(sentence):
    tokens = []
    for t in re.findall("[a-zA-Z]+",sentence):
         if len(t)>=1:
             tokens.append(t)
    return tokens

def emoji_ch (words):
    if words == '😄' :
      words='good'
    elif words == '😆' :
      words='good'
    elif words == '😊' :
      words='good'
    elif words == '😃' :
      words='good'	
    elif words == '🤬' :
      words='bad'
    elif words == '😏' :
      words='good'	
    elif words == '😍' :
      words='good'	
    elif words == '😘' :
      words='good'
    elif words == '😚' :
      words='good'	
    elif words == '😳' :
      words='flushed'	
    elif words == '😌' :
      words='bad'
    elif words == '😆' :
      words='good'	
    elif words == '😂' :
      words='good'	
    elif words == '😎' :
      words='good'		
    elif words == '🤯' :
      words='blow_mind'	
    elif words == '✌️' :
      words='vhand'			
    elif words == '🤘' :
      words='metal'
    elif words == '🤦' :
      words='bad'				
    elif words == '🤩':
      words='good'
    elif words == '🥰':
      words='good'
    elif words == '🤔':
      words='thinking'
    elif words == '🤣':
      words='good'
    elif words == '🤗':
      words = 'good'
    else :
      return None
    return words+" "


def tamilchar(ch):
    if ch == u'\u0B85' or ch== u'அ':
        ch="A"
    elif ch == u'\u0B86' or ch== u'ஆ':
        ch="AA"
    elif ch == u'\u0B87' or ch== u'இ':
        ch="I"
    elif ch == u'\u0B88' or ch== u'ஈ':
        ch="II"
    elif ch == u'\u0B89' or ch== u'உ':
        ch="U"
    elif ch == u'\u0B8A' or ch== u'ஊ':
        ch="UU"
    elif ch == u'\u0B8E' or ch== u'எ':
        ch="E"
    elif ch == u'\u0B8F' or ch== u'ஏ':
        ch="EE"
    elif ch == u'\u0B90' or ch== u'ஐ':
        ch="AI"
    elif ch == u'\u0B92' or ch== u'ஒ':
        ch="O"
    elif ch == u'\u0B93' or ch== u'ஓ':
        ch="OO"
    elif ch == u'\u0B94' or ch== u'ஔ':
        ch="AU"
    elif ch ==u'\u0B95'or ch==u'க':
        ch="KA"
    elif ch ==u'\u0B99'or ch==u'ங':
        ch="NGA"
    elif ch ==u'\u0B9A'or ch==u'ச':
        ch="SA"
    elif ch ==u'\u0B9C' or ch==u'ஜ':
        ch="JA"
    elif ch ==u'\u0B9E'or ch==u'ஞ':
        ch=u"NYA"
    elif ch ==u'\u0B9F'or ch==u'ட':
        ch="DA"
    elif ch ==u'\u0BA3'or ch==u'ண':
        ch="NNA"
    elif ch ==u'\u0BA4'or ch==u'த':
        ch="THA"
    elif ch ==u'\u0BA8'or ch==u'ந':
        ch="NA"
    elif ch ==u'\u0BA9' or ch==u'ன':
        ch="NA"
    elif ch ==u'\u0BAA'or ch==u'ப':
        ch="PA"
    elif ch ==u'\u0BAE'or ch==u'ம':
        ch="MA"
    elif ch ==u'\u0BAF' or ch==u'ய':
        ch="YA"
    elif ch ==u'\u0BB0' or ch==u'ர':
        ch="RA"
    elif ch ==u'\u0BB1'or ch==u'ற':
        ch="RRA"
    elif ch ==u'\u0BB2' or ch==u'ல':
        ch="LA"
    elif ch ==u'\u0BB3'or ch==u'ள':
        ch="LLA"
    elif ch ==u'\u0BB4'or ch ==u'ழ':
        ch=="LLLA"
    elif ch ==u'\u0BB5'or ch==u'வ':
        ch="VA"
    elif ch ==u'\u0BB6'or ch==u'ஶ':
        ch="SHA"
    elif ch ==u'\u0BB7'or ch==u'ஷ':
        ch="SSA"
    elif ch ==u'\u0BB8'or ch==u'ஸ':
        ch="SA"
    elif ch ==u'\u0BB9'or ch==u'ஹ':
        ch="HA"
    elif ch ==u'\u0BBE'or ch==u'$ா':
        ch="AA"
    elif ch ==u'\u0BBF'or ch==u'$ி':
        ch="I"
    elif ch ==u'\u0BC0'or ch==u'$ீ':
        ch="II"
    elif ch ==u'\u0BC1'or ch==u'$ு':
        ch="U"
    elif ch ==u'\u0BC2'or ch==u'ூ$':
        ch="UU"
    elif ch ==u'\u0BC6'or ch==u'$ெ':
        ch="E"
    elif ch ==u'\u0BC7'or ch==u'$ே':
        ch="EE"
    elif ch ==u'\u0BC8'or ch==u'$ை':
        ch="AI"
    elif ch ==u'\u0BCA'or ch==u'$ொ':
        ch="O"
    elif ch ==u'\u0BCB'or ch==u'$ோ':
        ch="OO"
    elif ch ==u'\u0BCC'or ch==u'$ௌ':
        ch="AU"
    elif ch ==u'\u0BCD'or ch==u'$்':
        ch="PULLI"
    elif ch ==u'\u0BD0'or ch==u'ௐ':
        ch="OM"
    elif ch ==u'\u0BD7'or ch==u'$ௗ':
        ch="AU"       
    else:
        return None
    return ch

def preprocess(word):
  word=re.sub(r'([a-z])\1+', r'\1',word)
  return word

def tamilword(word):
    fla=0
    string=""
    for ch in word:
        if(tamilchar(ch) is not None):
            ch=tamilchar(ch)
            fla=1
        if(emoji_ch(ch) is not None):
            ch=emoji_ch(ch)
            fla=1
        string=string+str(ch)
    if(fla==1):
        return string
    else:
        return None


def malchar(ch):
    if ch == u'അ':
        ch="A"
    elif ch == u'ആ':
        ch="AA"
    elif ch == u'ഇ':
        ch="I"
    elif ch == u'ഈ':
        ch="II"
    elif ch == u'ഉ':
        ch="U"
    elif ch == u'ഊ':
        ch="UU"
    elif ch == u'എ':
        ch="E"
    elif ch == u'ഏ':
        ch="EE"
    elif ch == u'ഐ':
        ch="AI"
    elif ch == u'ഒ':
        ch="O"
    elif ch == u'ഓ':
        ch="OO"
    elif ch == u'ഔ':
        ch="AU"
    elif ch ==u'ക'or ch==u'ഖ':
        ch="KA"
    elif ch ==u'ഗ'or ch==u'ഘ':
        ch="GA"
    elif ch ==u'ങ':
        ch="NGA"
    elif ch ==u'ച'or ch==u'ഛ':
        ch="CH" 
    elif ch ==u'ച'or ch==u'ഛ':
        ch="CH" 
    elif ch ==u'ശ'or ch==u'ഷ':
        ch="SH"
    elif ch ==u'സ':
        ch="S"
    elif ch ==u'ജ' or ch==u'ഝ':
        ch="JA"
    elif ch ==u'ഡ'or ch==u'ദ':
        ch="DA"
    elif ch ==u'ഥ'or ch==u'ത':
        ch="THA"
    elif ch ==u'ണ'or ch==u'ന':
        ch="NA"
    elif ch ==u'പ':
        ch="PA"
    elif ch ==u'മ':
        ch="MA"
    elif ch ==u'യ':
        ch="YA"
    elif ch ==u'ര':
        ch="RA"
    elif ch ==u'ള':
        ch="LA"
    elif ch ==u'വ':
        ch="VA"
    elif ch ==u'ഹ':
        ch="HA"
    elif ch ==u'ഭ'or ch==u'ബ':
        ch="BA"
    elif ch ==u'ഢ'or ch==u'ധ':
        ch="DH"
    elif ch ==u'ഠ':
        ch="DT"
    elif ch ==u'ട':
        ch="T"
    elif ch ==u'ഞ':
        ch="NJ"
    elif ch ==u'ഫ':
        ch="PH"
    elif ch ==u'ഴ':
        ch="ZH"
    elif ch ==u'റ':
        ch="RA"    
    else:
        return None
    return ch


def malword(word):
    fla=0
    string=""
    for ch in word:
        if(malchar(ch) is not None):
            ch=malchar(ch)
            fla=1
        string=string+str(ch)
    if(fla==1):
        return string
    else:
        return None



colnames=['text','label']
df = pd.read_csv(r'/content/drive/My Drive/Dravidian/EACL/tamil_train.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])

df_val = pd.read_csv(r'/content/drive/My Drive/Dravidian/EACL/tamil_dev.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])

colnames=['index','text','label']
df_test = pd.read_csv(r'/content/drive/My Drive/Dravidian/EACL/tamil_test.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])

# colnames=['text','label']
# df = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_train.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
#                       usecols=['text','label'], na_values=" NaN",skiprows=[0])

# df_val = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_dev.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
#                       usecols=['text','label'], na_values=" NaN",skiprows=[0])

# colnames=['index','text','label']
# df_test = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_test.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
#                       usecols=['text','label'], na_values=" NaN",skiprows=[0])

labels = ['Mixed_feelings ','Negative ','Positive ','not-Tamil ','unknown_state '] 

for index, line in df.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = 0
  elif line['label'].lower() == labels[1].lower():
        line['label'] = 1
  elif line['label'].lower() == labels[2].lower():
        line['label'] = 2
  elif line['label'].lower() == labels[3].lower():
        line['label'] = 3
  elif line['label'].lower() == labels[4].lower():
        line['label'] = 4
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      
      checker=tamilword(i[word])
      if(checker is not None):
          i[word]=checker
      i[word]=preprocess(i[word])
  line[0]=' '.join(i)

for index, line in df_val.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = 0
  elif line['label'].lower() == labels[1].lower():
        line['label'] = 1
  elif line['label'].lower() == labels[2].lower():
        line['label'] = 2
  elif line['label'].lower() == labels[3].lower():
        line['label'] = 3
  elif line['label'].lower() == labels[4].lower():
        line['label'] = 4
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      
      checker=tamilword(i[word])
      if(checker is not None):
          i[word]=checker
      i[word]=preprocess(i[word])
  line[0]=' '.join(i)

labels = ['Mixed_feelings','Negative','Positive','not-Tamil','unknown_state'] 
for index, line in df_test.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = 0
  elif line['label'].lower() == labels[1].lower():
        line['label'] = 1
  elif line['label'].lower() == labels[2].lower():
        line['label'] = 2
  elif line['label'].lower() == labels[3].lower():
        line['label'] = 3
  elif line['label'].lower() == labels[4].lower():
        line['label'] = 4
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      i[word]=preprocess(i[word])
      
      checker=tamilword(i[word])
      if(checker is not None):
          i[word]=checker
      i[word]=preprocess(i[word])
          
          
  line[0]=' '.join(i)

print(df.head())
print(df_val.head())
print(df_test.head())

x_train = df['text']
y_train = df['label']
x_dev = df_val['text']
y_dev = df_val['label']
x_test = df_test['text']
y_test = df_test['label']

from collections import Counter
unique = Counter()
df['text'].str.lower().str.split().apply(unique.update)
avl=[]
for a in df['text']:
  avl.append(len(a))


import math
cbowsize=100
sgsize=100
totalsize=cbowsize+sgsize
maxlen=math.ceil(sum(avl)/len(avl))
tokmaxwords=len(unique)
batch_size=128


def LabelComments(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

trainTotal = x_train
trainTotalAll = LabelComments(trainTotal, 'all')

cores = multiprocessing.cpu_count()
CBOW_Model = Word2Vec(sg=0, size=cbowsize, negative=10, window=15, min_count=2, workers=cores, alpha=0.1, min_alpha=0.001)
CBOW_Model.build_vocab([x.words for x in tqdm(trainTotalAll)])
for epoch in range(10):
    CBOW_Model.train(utils.shuffle([x.words for x in tqdm(trainTotalAll)]), total_examples=len(trainTotalAll), epochs=1)

from gensim.models import FastText
FASTTEXT_Model = FastText([x.words for x in tqdm(trainTotalAll)], size=sgsize, window=15, workers=4,sg=1,negative=10,iter=10)

embeddings_index = {}
for w in CBOW_Model.wv.vocab.keys():
  if(w in FASTTEXT_Model.wv.vocab.keys()):
    embeddings_index[w] = np.append(CBOW_Model.wv[w],FASTTEXT_Model.wv[w])


tokenizer = Tokenizer(num_words=tokmaxwords)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
x_train_sequences = pad_sequences(sequences, maxlen=maxlen)

sequences_val = tokenizer.texts_to_sequences(x_dev)
x_val_sequences= pad_sequences(sequences_val, maxlen=maxlen)

sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_sequences = pad_sequences(sequences_test, maxlen=maxlen)

num_words = tokmaxwords
embedding_matrix = np.zeros((num_words, totalsize))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def LSTM_model(max_sequence_length, max_words_fn, embedding_dim, numberOfClasses):
    embedding_layer = Embedding(max_words_fn,embedding_dim,input_length=max_sequence_length,trainable=False,weights=[embedding_matrix])
    sequence_input = Input(shape=(max_sequence_length,), dtype='float32')
    embedded_sequences = embedding_layer(sequence_input)
    layer1 = LSTM(64, return_sequences=True, dropout = 0.3, recurrent_dropout=0.2)(embedded_sequences)
    layer2 = LSTM(32, dropout = 0.3, recurrent_dropout =0.2)(layer1)
    layer3 = Dense(numberOfClasses)(layer2)
    layer4 = BatchNormalization()(layer3)
    preds = Activation('softmax')(layer4)
    model = Model(sequence_input, preds)
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.summary()
    return model

max_words = tokmaxwords
maxlen = maxlen
embedding_dims = totalsize
model = LSTM_model(maxlen, max_words, embedding_dims, 5)
num_epochs = 500

y_train2 = np.asarray(y_train).astype('float32').reshape((-1,1))
y_val2 = np.asarray(y_dev).astype('float32').reshape((-1,1))

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train_sequences, y_train2, batch_size=batch_size,
                  callbacks=[es_callback], epochs=num_epochs, validation_data=(x_val_sequences, y_val2), shuffle=True,verbose=1)

from keras.utils import np_utils
from sklearn.metrics import classification_report
y_test3 = np_utils.to_categorical(y_test, 5)
predictions = model.predict(x_test_sequences)
prediction = (predictions>0.20) 
print(classification_report(y_test3, prediction))
