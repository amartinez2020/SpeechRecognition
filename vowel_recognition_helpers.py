# imports
import librosa
import os
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#load dataframe with training samples and labels
def load_df(file_path,columns,df_path):
    df = pd.DataFrame(columns=columns)
    index = 0
    #extract wav file path and insert in df
    for subdir,dirs,files in os.walk(file_path):
        for each_dir in dirs:
            dir_path = os.path.join(file_path,each_dir)
            for files in os.walk(dir_path):
                for each_wav in files[2]:
                    label = each_wav[-6:-4]
                    df.loc[index] = [os.path.join(dir_path,each_wav),None,label,None]
                    index += 1
    #return df path and save to csv
    df_path = os.path.join(df_path)
    df.to_csv(df_path)
    return(df_path)

#extract mfccs from each wav file
def extract_mfcc(df_path,vowels_to_classify,ratio=0.9,maxlen=12):
    df = pd.read_csv(df_path)
    df = df.astype(object)
    #load all mfccs into array for later processing
    mfccs = []

    #encode training labels as one hot encoded vectors
    label_encoder = LabelEncoder()
    vowels_encoded = label_encoder.fit_transform(list(vowels_to_classify.keys()))
    onehot_encoder = OneHotEncoder(sparse=False)
    vowels_encoded = vowels_encoded.reshape(len(vowels_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(vowels_encoded)

    #extract mfccs from each wav file in df
    for index, row in df.iterrows():
        y,sr = librosa.load(df.loc[index,'audio file'],sr=None)
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)

        #plotting function commented out
        #plot_mels(mfcc)

        #load all mfccs into array for later processing
        mfccs.append(mfcc)

        #save individual mfccs and encodings in df
        df.loc[index,'mfcc'] = mfcc
        df.loc[index,'encoded label'] = onehot_encoded[vowels_to_classify[df.loc[index,'vowel label']]]

    #separate all training samples and labels
    #training samples: mfccs
    #training labels: encodings
    x = mfccs
    y = df['encoded label']

    #split up train and val sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1-ratio)

    #format train and val tests
    #applying padded sequences to encodings
    #convert both samples and encodings to high dimensional numpy arrays
    npx = np.random.random((len(X_train), len(X_train[0]), maxlen))
    npy = np.random.random((len(y_train), 12))
    npxt = np.random.random((len(X_test), len(X_test[0]), maxlen))
    npyt = np.random.random((len(y_test), 12))
    for sample in range(len(npx)):
        X_train[sample] = sequence.pad_sequences(X_train[sample], maxlen=maxlen)
        for mfcc in range(len(npx[sample])):
            npx[sample][mfcc] = np.array(X_train[sample][mfcc])
    for sample in range(len(npy)):
            npy[sample] = np.array(y_train)[sample]
    for sample in range(len(npxt)):
        X_test[sample] = sequence.pad_sequences(X_test[sample], maxlen=maxlen)
        for mfcc in range(len(npxt[sample])):
            npxt[sample][mfcc] = np.array(X_test[sample][mfcc])
    for sample in range(len(npyt)):
            npyt[sample] = np.array(y_test)[sample]

    #save df and return train and val sets
    df.to_csv(df_path,index=False)
    return(npx, npxt, npy, npyt)



#plot mfccs
def plot_mels(mfcc):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()


#train network
def train(X_train, X_test, y_train, y_test,params,vowels_to_classify,units=100,epochs=15):

    #create lstm and dense layer
    model = Sequential()
    model.add(LSTM(units,input_shape=(params['data dim'],params['timesteps'])))
    model.add(Dense(params['num classes'], activation='sigmoid'))

    #compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #display model summary
    print(model.summary())

    #fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=params['batch size'],verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
