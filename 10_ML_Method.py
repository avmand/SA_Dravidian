
"""

Citation:
@inproceedings{ bitsnlp2021dravidian,
 title={Sentiment Analysis of Dravidian Code Mixed Data},
 author={Mandalam, Asrita Venkata and Sharma, Yashvardhan},
 booktitle={ 15th Conference of the European Chapter of the Association for Computational Linguistics 2021 (EACL 2021) },
 year={2021}
}  

"""


import pandas as pd
import re
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

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
  word=re.sub(r'([a-z])\1+$', r'\1',word)
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

labels = ['Mixed_feelings ','Negative ','Positive ','not-Tamil ','unknown_state '] 
train_csv='/content/drive/My Drive/Dravidian/EACL/tamil_train.tsv'



colnames=['tweet','sentiment']
dataset_csv = pd.read_csv(train_csv,names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=colnames, na_values=" NaN",encoding="ISO-8859-1")

dataset_csv = dataset_csv.dropna()

dataset=dataset_csv
dataset=dataset[1:]

for index, line in dataset.iterrows():
  if line['sentiment'].lower() == labels[0].lower():
        line['sentiment'] = "0"
  elif line['sentiment'].lower() == labels[1].lower():
        line['sentiment'] = "1"
  elif line['sentiment'].lower() == labels[2].lower():
        line['sentiment'] = "2"
  elif line['sentiment'].lower() == labels[3].lower():
        line['sentiment'] = "3"
  elif line['sentiment'].lower() == labels[4].lower():
        line['sentiment'] = "4"
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



x_train = dataset['tweet']
y_train = dataset['sentiment']

test_csv='/content/drive/My Drive/Dravidian/EACL/tamil_dev.tsv'

labels = ['Mixed_feelings','Negative','Positive','not-Tamil','unknown_state'] 
colnames=['index','tweet','sentiment']
dataset_csv = pd.read_csv(r'/content/drive/My Drive/Dravidian/EACL/tamil_test.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['tweet','sentiment'], na_values=" NaN",skiprows=[0])

dataset_csv = dataset_csv.dropna()

dataset=dataset_csv
dataset=dataset[1:]

for index, line in dataset.iterrows():
  if line['sentiment'].lower() == labels[0].lower():
        line['sentiment'] = "0"
  elif line['sentiment'].lower() == labels[1].lower():
        line['sentiment'] = "1"
  elif line['sentiment'].lower() == labels[2].lower():
        line['sentiment'] = "2"
  elif line['sentiment'].lower() == labels[3].lower():
        line['sentiment'] = "3"
  elif line['sentiment'].lower() == labels[4].lower():
        line['sentiment'] = "4"
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


x_test = dataset['tweet']
y_test = dataset['sentiment']

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
df = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_train.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])


df_val = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_dev.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])

colnames=['index','text','label']
df_test = pd.read_csv(r'/content/drive/MyDrive/Dravidian/EACL/malayalam_test.tsv',names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                      usecols=['text','label'], na_values=" NaN",skiprows=[0])

labels = ['Mixed_feelings ','Negative ','Positive ','not-malayalam ','unknown_state '] 

for index, line in df.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = "0"
  elif line['label'].lower() == labels[1].lower():
        line['label'] = "1"
  elif line['label'].lower() == labels[2].lower():
        line['label'] = "2"
  elif line['label'].lower() == labels[3].lower():
        line['label'] = "3"
  elif line['label'].lower() == labels[4].lower():
        line['label'] = "4"
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      
      checker=malword(i[word])
      if(checker is not None):
          i[word]=checker
      i[word]=preprocess(i[word])
          
          
  line[0]=' '.join(i)

for index, line in df_val.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = "0"
  elif line['label'].lower() == labels[1].lower():
        line['label'] = "1"
  elif line['label'].lower() == labels[2].lower():
        line['label'] = "2"
  elif line['label'].lower() == labels[3].lower():
        line['label'] = "3"
  elif line['label'].lower() == labels[4].lower():
        line['label'] = "4"
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      
      checker=malword(i[word])
      if(checker is not None):
          i[word]=checker
      i[word]=preprocess(i[word])
          
          
  line[0]=' '.join(i)


labels = ['Mixed_feelings','Negative','Positive','not-malayalam','unknown_state'] 
for index, line in df_test.iterrows():
  if line['label'].lower() == labels[0].lower():
        line['label'] = "0"
  elif line['label'].lower() == labels[1].lower():
        line['label'] = "1"
  elif line['label'].lower() == labels[2].lower():
        line['label'] = "2"
  elif line['label'].lower() == labels[3].lower():
        line['label'] = "3"
  elif line['label'].lower() == labels[4].lower():
        line['label'] = "4"
  i=line[0].split()
  for word in range(len(i)):
      for ch in range(len(i[word])):
          checker=emoji_ch(i[word][ch])
          if(checker is not None):
              i[word]+=checker
      i[word]=preprocess(i[word])
      
      checker=malword(i[word])
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

vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1,2))
X_train = vectorizer.fit_transform(x_train).toarray()
X_test = vectorizer.transform(x_test).toarray()

print(y_train, y_test)

att = [5,12] #temp
for i in range(len(att)):
    names = ["SVM Linear", "Logistic Regression" ]
    classifiers = [
        LinearSVC(random_state=0, tol=1e-6, multi_class= 'crammer_singer',dual = True, intercept_scaling = att[i]) ,
        LogisticRegression(max_iter=500, solver = 'newton-cg',C = att[i], multi_class = 'ovr')
    ]

    models = zip(names, classifiers)
    results = []
    names = []
    print("att: ",att[i])
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(name)
        print("Acc: ", accuracy_score(y_test, predictions) , f1_score(y_test, predictions, average="weighted"))
        print(classification_report(y_test, predictions))

