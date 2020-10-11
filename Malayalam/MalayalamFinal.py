import numpy as np
import re
import keras
from copy import deepcopy
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import pandas as pd
from keras.utils import np_utils
from keras.layers import BatchNormalization
import csv
from matplotlib import pyplot

###############################################################################

"""
Preprocessing functions
"""


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
      words='relieved'
    elif words == '😆' :
      words='satisfied'	
    elif words == '😂' :
      words='good'	
    elif words == '😎' :
      words='sunglasses'		
    elif words == '🤯' :
      words='blow_mind'	
    elif words == '✌️' :
      words='vhand'			
    elif words == '🤘' :
      words='metal'
    elif words == '🤦' :
      words='facepalm'				
    elif words == '🤩':
      words='good'
    elif words == '🥰':
      words='good'
    elif words == '🤔':
      words='thinking'
    elif words == '🤣':
      words='good'
    else :
      return None
    return words


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
#        print(string)
        return string
    else:
        return None
    
    
"""
End of Preprocessing Functions
"""
###############################################################################



def parse(filename):
    
    f=open(filename,'r',encoding='utf-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    lines=lines[1:]
    labels = ['Mixed_feelings ','Negative ','Positive ','not-malayalam ','unknown_state '] 
    X_train = []
    Y_train = []
    for line in lines:
        line = line.split('\t')
        i=line[0].split()
        for word in range(len(i)):
            for ch in range(len(i[word])):
                checker=emoji_ch(i[word][ch])
                if(checker is not None):
                    i[word]+=checker
            
            checker=malword(i[word])
            if(checker is not None):
                i[word]=checker
                
                
        line[0]=' '.join(i)
        
        tokenized_lines = token(line[0])
        
        char_list = []
        for words in tokenized_lines:
        			for char in words:
        				char_list.append(char)
        			char_list.append(' ')
                    
        X_train.append(char_list)
        		
        if line[1].lower() == labels[0].lower():
        			Y_train.append(0)
        if line[1].lower() == labels[1].lower():
        			Y_train.append(1)
        if line[1].lower() == labels[2].lower():
        			Y_train.append(2)
        if line[1].lower() == labels[3].lower():
        			Y_train.append(3)
        if line[1].lower() == labels[4].lower():
        			Y_train.append(4)
    	
    Y_train = np.asarray(Y_train)
    assert(len(X_train) == Y_train.shape[0])
    return [X_train,Y_train]



def parsetest(filename):
    
    f=open(filename,'r',encoding='utf-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')
#    print(len(lines))
#    print(lines[:5])
    lines=lines[1:]
    X_test = []
#    print(len(lines))
#    print(lines[:5])
    for line in lines:
        line = line.split('\t')
        i=line[1].split()
        for word in range(len(i)):
            for ch in range(len(i[word])):
                checker=emoji_ch(i[word][ch])
                if(checker is not None):
                    i[word]+=checker
            
            checker=malword(i[word])
            if(checker is not None):
                i[word]=checker
        line[0]=' '.join(i)
        tokenized_lines = token(line[0])
        
        char_list = []
        for words in tokenized_lines:
        			for char in words:
        				char_list.append(char)
        			char_list.append(' ')
                    
        X_test.append(char_list)
    	
    return [X_test]



def convert_char2num(mapping_n2c,mapping_c2n,trainwords):
	allchars = []
	errors = 0
	for line in trainwords:
		try:
			allchars = set(allchars+line)
			allchars = list(allchars)
		except:
			errors += 1
	charno = 0
	for char in allchars:
		charnum[char] = charno
		numchar[charno] = char
		charno += 1
	assert(len(allchars)==charno) 
	X_train = []
	for line in trainwords:
		char_list=[]
		for letter in line:
			char_list.append(charnum[letter])
		X_train.append(char_list)
	X_train = sequence.pad_sequences(X_train[:], maxlen=maximumlength)
	return [X_train,numchar,charnum,charno]



def GetModelLSTM(X_train,y_train, X_dev,y_dev,max_features):
    
    y_train = np_utils.to_categorical(y_train, 5)
    y_dev = np_utils.to_categorical(y_dev, 5)
    
    
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maximumlength))
    model.add(Convolution1D(filters=128,kernel_size=5,activation='relu'))
    model.add(MaxPooling1D())
    model.add(LSTM(128, dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, return_sequences=False))
    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
			  	optimizer='adamax',
			  	metrics=['accuracy'])
	
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

    history=model.fit(X_train, y_train, 
    			  batch_size=128, 
    			  shuffle=True,
                  verbose=2,
    			  epochs=30,
                  callbacks=[es_callback],
    			  validation_data=(X_dev, y_dev))
    
    
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation acc')
    pyplot.ylabel('acc')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    return model


def evaluate_model(X_test,model):
    
    predictions = model.predict(X_test)
    prediction = (predictions > 0.2) 
#    print(len(prediction))
    colnames=['id','text']
    dataset = pd.read_csv(inputdatasetfilenametest,names=colnames, delimiter='\t', error_bad_lines=False, header=None,
                          usecols=['id','text'], na_values=" NaN",encoding='utf-8')
    result=[]
    result.append('label')
    for x in prediction:
        if(x[0]==True):
            result.append('Mixed_feelings')
        elif(x[1]==True):
            result.append('Negative')
        elif(x[2]==True):
            result.append('Positive')
        elif(x[3]==True):
            result.append('not-malayalam')
        elif(x[4]==True):
            result.append('unknown_state')
#    print(len(result))
    dataset['label']=result
    with open('result_callbacks.tsv', 'wt',encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['id','text','label'])
        for row in range(1,(len(dataset))):
            tsv_writer.writerow([dataset['id'][row], dataset['text'][row],dataset['label'][row]])



inputdatasetfilename = 'malayalam_train.tsv'
inputdatasetfilenamedev = 'malayalam_dev.tsv'
inputdatasetfilenametest = 'malayalam_test.tsv'
max_features=0

charnum = {}
numchar = {}
maximumlength = 200

parsed = parse(inputdatasetfilename)
X_train = parsed[0]
y_train = parsed[1]

parsed = parse(inputdatasetfilenamedev)
X_dev = parsed[0]
y_dev = parsed[1]

parsed = parsetest(inputdatasetfilenametest)
X_test = parsed[0]

joined=X_train+X_dev+X_test

parsed = convert_char2num(numchar,charnum,joined)
numchar = parsed[1]
charnum = parsed[2]
max_features = parsed[3]
joined_train = np.asarray(parsed[0])
X_train=joined_train[:(len(X_train))]
X_dev=joined_train[(len(X_train)):(len(X_train)+len(X_dev))]
X_test=joined_train[(len(X_train)+len(X_dev)):]
y_train = np.asarray(y_train).flatten()
y_dev = np.asarray(y_dev).flatten()


model = GetModelLSTM(deepcopy(X_train),deepcopy(y_train),
            deepcopy(X_dev),deepcopy(y_dev),max_features)

evaluate_model(X_test,model)
    	
