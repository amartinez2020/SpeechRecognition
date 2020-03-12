# import keras
import librosa
import os
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import librosa.display
from ast import literal_eval
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from sklearn import preprocessing #import sequence
from scipy.sparse import csr_matrix
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_df(file_path,columns,df_path):
    df = pd.DataFrame(columns=columns)
    index = 0
    for subdir,dirs,files in os.walk(file_path):
        for each_dir in dirs:
            # print(each_dir)
            dir_path = os.path.join(file_path,each_dir)
            for files in os.walk(dir_path):
                #print(files[2])
                for each_wav in files[2]:
                    # print(os.path.join(dir_path,each_wav))
                    label = each_wav[-6:-4]
                    # print(label)
                    df.loc[index] = [os.path.join(dir_path,each_wav),None,label,None]
                    index += 1
    df_path = os.path.join(df_path)
    df.to_csv(df_path)
    return(df_path)

def extract_mfcc(df_path,vowels_to_classify,ratio=0.9,maxlen=12):
    df = pd.read_csv(df_path)
    df = df.astype(object)
    mfccs = []
    label_encoder = LabelEncoder()
    # print(list(vowels_to_classify.keys()))
    vowels_encoded = label_encoder.fit_transform(list(vowels_to_classify.keys()))
    onehot_encoder = OneHotEncoder(sparse=False)
    vowels_encoded = vowels_encoded.reshape(len(vowels_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(vowels_encoded)

    # print(vowels_encoded)
    for index, row in df.iterrows():
        y,sr = librosa.load(df.loc[index,'audio file'],sr=None)
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)

        # data = np.array([[1, 2, 3], [5, 6, 7]])

        # mfcc = np.average(mfcc, axis=0)

        #print(len(mfcc))
        mfccs.append(mfcc)
        df.loc[index,'mfcc'] = mfcc
        #print(onehot_encoded[vowels_to_classify[df.loc[index,'vowel label']]])
        df.loc[index,'encoded label'] = onehot_encoded[vowels_to_classify[df.loc[index,'vowel label']]]
        # sleep(10)


    #df.apply(np.random.shuffle, axis=0)
    # length_of_samples = len(mfccs[0])
    # length_of_vectors = len(mfccs[0][0])

    x = mfccs
    y = df['encoded label']


    #
    # encoderX = preprocessing.OneHotEncoder()
    # X = encoderX.fit_transform(x)
    # encoderY = preprocessing.LabelEncoder()
    # y = encoderY.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1-ratio)
    npx = np.random.random((len(X_train), len(X_train[0]), maxlen))
    npy = np.random.random((len(y_train), 12))
    npxt = np.random.random((len(X_test), len(X_test[0]), maxlen))
    npyt = np.random.random((len(y_test), 12))
    # print(y_train)
    #within each row
    for sample in range(len(npx)):
        #within each sample
        X_train[sample] = sequence.pad_sequences(X_train[sample], maxlen=maxlen)

        for mfcc in range(len(npx[sample])):

            #X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
            npx[sample][mfcc] = np.array(X_train[sample][mfcc])
            # np.array(X_train[sample][mfcc])
    # print(len(npy))
    for sample in range(len(npy)):
        #within each sample
            #convert to one hot encoding

            npy[sample] = np.array(y_train)[sample]
    for sample in range(len(npxt)):
        #within each sample
        X_test[sample] = sequence.pad_sequences(X_test[sample], maxlen=maxlen)

        for mfcc in range(len(npxt[sample])):

            #X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
            npxt[sample][mfcc] = np.array(X_test[sample][mfcc])
            # np.array(X_train[sample][mfcc])
    for sample in range(len(npyt)):
        #within each sample
            #convert to one hot encoding
            npyt[sample] = np.array(y_test)[sample]




    # print(type(X_train))
    df.to_csv(df_path,index=False)
    return(npx, npxt, npy, npyt)


        # sleep(10)
        #plot_mels(mfcc)


def plot_mels(mfcc):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()



def train(df_path, X_train, X_test, y_train, y_test,top_words,params,prediction_path,vowels_to_classify,units=100,epochs=2):
    df = pd.read_csv(os.path.join(df_path))
    df = df.astype(object)

    # create the model


    model = Sequential()
    #model.add(Embedding(top_words, length_of_vectors, input_length=length_of_samples))
    model.add(LSTM(units,input_shape=(params['data dim'],params['timesteps'])))
    # model.add(Dropout(0.2))
    # model.add(LSTM(params['batch size']))
    model.add(Dense(params['num classes'], activation='sigmoid'))
    # keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("++++++++")
    print(type(X_train))
    print("++++++++")
    # print(len(X_train))

    # print(len(y_train))
    print(model.summary())

    model.fit(X_train, y_train, epochs=epochs, batch_size=params['batch size'],verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    #predict
    y,sr = librosa.load(prediction_path,sr=16000)
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)
    mfcc = sequence.pad_sequences(mfcc, maxlen=12)
    mfcc = np.array(mfcc).reshape((1,np.array(mfcc).shape[0], np.array(mfcc).shape[1]))
    prediction = model.predict(mfcc)
    # print(vowels_to_classify)
    print(prediction)
    idx = np.argmax(prediction)
    print(f'vowel is: {list(vowels_to_classify.keys())[idx]}')
