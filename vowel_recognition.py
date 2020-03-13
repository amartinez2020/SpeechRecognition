import os
from vowel_recognition_helpers import *
import numpy as np

#batch size
batch_size = 32

#indicate column names and vowels to classify
columns = ["audio file","mfcc","vowel label","encoded label"]
vowels_to_classify = {"ae":0, "ah":1, "aw":2, "eh":3, "er":4, "ei":5, "ih":6, "iy":7, "oa":8, "oo":9, "uh":10, "uw":11}

#paths
df_path = "audio_files/df.csv"
cwd = os.getcwd()
file_path = os.path.join(cwd,'audio_files')

#load df
print("LOADING DATAFRAME...\n")
df_path = load_df(file_path,columns,df_path)

#extract mfccs and split train test sets
print("EXTRACTING MFCCS AND SPLITTING TRAIN AND TEST SETS...\n")
X_train, X_test, y_train, y_test = extract_mfcc(df_path,vowels_to_classify)

#create training parameters
params = {'data dim':len(X_train[0]),'timesteps':len(X_train[0][0]),'num classes':len(vowels_to_classify.keys()),'batch size':32,'input dim':len(X_train)}

#train network
print("TRAINING NETWORK...")
train(X_train, X_test, y_train, y_test, params,vowels_to_classify)
