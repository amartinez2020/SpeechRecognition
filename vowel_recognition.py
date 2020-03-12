import os
from vowel_recognition_helpers import *
import numpy as np
# data_dim = 16
# timesteps = 8
# num_classes = 10
batch_size = 32

df_path = "audio_files/df5.csv"
prediction_path = "Alexae.wav"
# os.chdir("..")
cwd = os.getcwd()
file_path = os.path.join(cwd,'audio_files')

columns = ["audio file","mfcc","vowel label","encoded label"]
df_path = load_df(file_path,columns,df_path)
# extract_mfcc(df_path)
# smooth_features(file_path)
print("extacting mfccs and splitting train test sets\n")
vowels_to_classify = {"ae":0, "ah":1, "aw":2, "eh":3, "er":4, "ei":5, "ih":6, "iy":7, "oa":8, "oo":9, "uh":10, "uw":11}

X_train, X_test, y_train, y_test = extract_mfcc(df_path,vowels_to_classify)
top_words = 5000
print("training")
print(len(X_train[0]))
print(len(X_train[0][0]))
# print(len(X_train[0])
# print(X_train[0][0])
params = {'data dim':len(X_train[0]),'timesteps':len(X_train[0][0]),'num classes':len(vowels_to_classify.keys()),'batch size':32,'input dim':len(X_train)}
train(df_path, X_train, X_test, y_train, y_test, top_words, params,prediction_path,vowels_to_classify,epochs=10)
# print(X_test)
# print(y_test)
