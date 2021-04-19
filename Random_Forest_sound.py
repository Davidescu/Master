import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa.display
import numpy, scipy, matplotlib.pyplot as plt, sklearn, pandas, librosa, urllib, IPython.display, os.path
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import load_model
import csv
import pandas as pd
import os
import soundfile as sf
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds
def load_sound_files_ret_fs(file_paths):
    fs=[]
    for fp in file_paths:
        x,sr=librosa.load(fp)
        fs.append(sr)
    return fs
def plot_waves(sound_names,raw_sounds):
    i = 1
    
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
       
        librosa.display.waveplot(np.array(f),sr=22050)
       # plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        specgram(np.array(f), Fs=22050)
        #plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_mfcc(sound_names,raw_sounds):
    i=1
    featuresFileFullPath=os.path.join(r'F:\MasterCI\train.csv')
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        mffc=24
        mffc_b=librosa.feature.mfcc(np.array(f),sr=22050,n_mfcc=mffc)
        
        np.savetxt(featuresFileFullPath,mffc_b,fmt='%1.3f',delimiter=",")
        i+=1
        librosa.display.specshow(mffc_b, sr=22050, x_axis='time')
        
     
    plt.suptitle("Figure 3: MFFC",x=0.5, y=0.915,fontsize=18)
    plt.show()
def get_features(file_name):

    if file_name: 
        X, sample_rate = sf.read(file_name, dtype='float32')

    
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled

def extract_features():

    
    sub_dirs = os.listdir('dataset')
    sub_dirs.sort()
    features_list = []
    for label, sub_dir in enumerate(sub_dirs):  
        for file_name in glob.glob(os.path.join('dataset',sub_dir,"*.ogg")):
            print("Extracting file ", file_name)
            try:
                mfccs = get_features(file_name)
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,label])

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label'])
    print(features_df.head())    
    return features_df
def get_numpy_array(features_df):

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    return X,yy,le

def train(model,X_train, X_test, y_train, y_test,model_file):    
    
   
    model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    print(model.summary())
    print("training for 100 epochs with batch size 32")
   
    model.fit(X_train,y_train,batch_size= 32, epochs = 100, validation_data=(X_test,y_test))
    print("Saving model to disk")
    model.save(model_file)
def compute(X_test,y_test,model_file):

    
    loaded_model = load_model(model_file)
    score = loaded_model.evaluate(X_test,y_test)
    return score[0],score[1]*100
	
def predict(filename,le,model_file):

    model = load_model(model_file)
    prediction_feature =get_features(filename)
    if model_file == "trained_rf.h5":
        prediction_feature = np.array([prediction_feature])
  
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("Predicted class",predicted_class[0])
    predicted_proba_vector = model.predict_proba([prediction_feature])

    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )



def get_train_test(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    return  X_train, X_test, y_train, y_test

path=r'F:\MasterCI\ESC-10\001 - Dog bark'
files=[]
sound_names=[]
for r,d,f in os.walk(path):
    for file in f:
        if '.ogg' in file:
            files.append(os.path.join(r,file))
            sound_names.append(os.path.join(r,file))
            
sound_file_paths = [r'F:\MasterCI\ESC-10\001 - Dog bark\1-30226-A.ogg',r'F:\MasterCI\ESC-10\001 - Dog bark\1-32318-A.ogg',r'F:\MasterCI\ESC-10\001 - Dog bark\1-110389-A.ogg',r'F:\MasterCI\ESC-10\001 - Dog bark\2-118964-A.ogg']

sound_names = ["Dogs","Dogs","Dogs","Dogs","Dogs"]

raw_sounds = load_sound_files(sound_file_paths)

plot_waves(sound_names,raw_sounds)
plot_specgram(sound_names,raw_sounds)

#Mel-Frequency Cepstral Coefficients
plot_mfcc(sound_names,raw_sounds)

print("Extracting features..")
features_df =extract_features()
X, y, le = get_numpy_array(features_df)

# split into training and testing data
X_train, X_test, y_train, y_test = get_train_test(X,y)
num_labels = y.shape[1]
  
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
  

#Random Forest
forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
forest.fit(X_train, y_train)

print("Training..")
nn.train(model,X_train, X_test, y_train, y_test,"trained_rf.h5")


test_loss, test_accuracy =compute(X_test,y_test,"trained_rf.h5")
print("Test loss",test_loss)
print("Test accuracy",test_accuracy)

#Predictia
predict("dataset/001 - Dog bark/1-30226-A.ogg",le,"trained_rf.h5")