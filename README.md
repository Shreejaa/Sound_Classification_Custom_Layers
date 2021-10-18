# Sounds Classification and Detection for Rain forest Species Using Custom CNN And Combinations Of Spectrograms

---

**Shreejaa Talla**<br>
*Department of Computer Science	
Georgia State University*<br>
Georgia, USA<br>
stalla1@student.gsu.edu	

---

[Research Paper]()

## INDEX
1. [Instructions](#instructions)
2. [Import Required Libraries](#import)
3. [Data Preprocessing](#load)
    1. [Load Data](#load)
    2. [Data Understanding](#under)
    3. [Signals and Spectrograms](#ss)
    4. [Assign Sound Constants](#constants)
    5. [Audio Augmentation](#audio)
    6. [Data Model for training](#dmtrain)
    7. [Extract Features](#features)
    8. [Extract Labels](#labels)
4. [Classification Models](#model)
    1. [Main Model](#main)
    2. [Model1 MFCC and LOG_MEL](#model)
    3. [Model2 STFT and LOG_MEL](#model2)
    4. [Model3 STFT and MFCC](#model3)
5. [Testing Models](#model1p)
    1. [Model1 predictions](#model1p)
    2. [Model2 predictions](#model2p)
    3. [Model3 predictions](#model3p)
6. [Comparision of Models](#compare)
7. [Conclusion](#conclusion)
---

<a id="instructions"></a>
## INSTRUCTIONS

Instruction to run the code and the execution time for each module:
- Please run all the load data part first and then move on to assigning constants as some of the cells are later arranged based on order of understanding in a jupiter notebook.
- For Data Preprocessing, extraction of features might take up to 11-12 hours for each spectrogram.
- For Training model, the estimated time to train each model is 20-24 hours.
- For prediction, the estimated time is 3-4 hours for each model.
- For comparision of models, the estimated time is 2 mins on total.

<a id="import"></a>
## IMPORT REQUIRED LIBRARIES


```python
import os
import math, glob
import re
import json

from tqdm import tqdm

import librosa
import librosa.display as ld 

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from IPython.display import Audio

from skimage.transform import resize
import multiprocessing


import soundfile as sf
from audiomentations import TimeStretch,TimeMask,FrequencyMask, Compose

import tensorflow as tf 
from tensorflow import keras 
from keras import regularizers
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')
```

<a id="load"></a>
## DATA PREPROCESSING 
### LOAD DATA

- Here, train_tp (true positive) values and train_fp (false positive) values are loaded from csv files and are concatenated to a single table based on is_tp value as 0 or 1 based on whether it is true positive or false positive values.


```python
train_fp = pd.read_csv("train_fp.csv")
train_fp["is_tp"] = 0
train_fp = train_fp[:25]
train_fp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recording_id</th>
      <th>species_id</th>
      <th>songtype_id</th>
      <th>t_min</th>
      <th>f_min</th>
      <th>t_max</th>
      <th>f_max</th>
      <th>is_tp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00204008d</td>
      <td>21</td>
      <td>1</td>
      <td>13.8400</td>
      <td>3281.2500</td>
      <td>14.9333</td>
      <td>4125.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00204008d</td>
      <td>8</td>
      <td>1</td>
      <td>24.4960</td>
      <td>3750.0000</td>
      <td>28.6187</td>
      <td>5531.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00204008d</td>
      <td>4</td>
      <td>1</td>
      <td>15.0027</td>
      <td>2343.7500</td>
      <td>16.8587</td>
      <td>4218.7500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>003b04435</td>
      <td>22</td>
      <td>1</td>
      <td>43.2533</td>
      <td>10687.5000</td>
      <td>44.8587</td>
      <td>13687.5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>003b04435</td>
      <td>23</td>
      <td>1</td>
      <td>9.1254</td>
      <td>7235.1562</td>
      <td>15.2091</td>
      <td>11283.3984</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_tp = pd.read_csv("train_tp.csv")
train_tp["is_tp"] = 1
train_tp = train_tp[:25]
train_tp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recording_id</th>
      <th>species_id</th>
      <th>songtype_id</th>
      <th>t_min</th>
      <th>f_min</th>
      <th>t_max</th>
      <th>f_max</th>
      <th>is_tp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>003bec244</td>
      <td>14</td>
      <td>1</td>
      <td>44.5440</td>
      <td>2531.250</td>
      <td>45.1307</td>
      <td>5531.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>006ab765f</td>
      <td>23</td>
      <td>1</td>
      <td>39.9615</td>
      <td>7235.160</td>
      <td>46.0452</td>
      <td>11283.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>007f87ba2</td>
      <td>12</td>
      <td>1</td>
      <td>39.1360</td>
      <td>562.500</td>
      <td>42.2720</td>
      <td>3281.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0099c367b</td>
      <td>17</td>
      <td>4</td>
      <td>51.4206</td>
      <td>1464.260</td>
      <td>55.1996</td>
      <td>4565.04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009b760e6</td>
      <td>10</td>
      <td>1</td>
      <td>50.0854</td>
      <td>947.461</td>
      <td>52.5293</td>
      <td>10852.70</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = pd.concat([train_fp,train_tp]).reset_index()
del train["index"]
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recording_id</th>
      <th>species_id</th>
      <th>songtype_id</th>
      <th>t_min</th>
      <th>f_min</th>
      <th>t_max</th>
      <th>f_max</th>
      <th>is_tp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00204008d</td>
      <td>21</td>
      <td>1</td>
      <td>13.8400</td>
      <td>3281.2500</td>
      <td>14.9333</td>
      <td>4125.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00204008d</td>
      <td>8</td>
      <td>1</td>
      <td>24.4960</td>
      <td>3750.0000</td>
      <td>28.6187</td>
      <td>5531.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00204008d</td>
      <td>4</td>
      <td>1</td>
      <td>15.0027</td>
      <td>2343.7500</td>
      <td>16.8587</td>
      <td>4218.7500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>003b04435</td>
      <td>22</td>
      <td>1</td>
      <td>43.2533</td>
      <td>10687.5000</td>
      <td>44.8587</td>
      <td>13687.5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>003b04435</td>
      <td>23</td>
      <td>1</td>
      <td>9.1254</td>
      <td>7235.1562</td>
      <td>15.2091</td>
      <td>11283.3984</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<a id="under"></a>
## DATA UNDERSTANDING

- The range of maximum and minimum frequencies through the data.


```python
g = sns.catplot(y="f_max", x="species_id", hue = "songtype_id",
                     kind = "strip", data=train, height=7,aspect=3)
ax = sns.catplot(x="species_id", y="f_min",hue = "songtype_id",
                     kind = "strip", data=train, height=7,aspect=3)
```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    


- The range of maximum and minimum times through the data.


```python
g = sns.catplot(y="t_max", x="species_id", hue = "songtype_id",
                     kind = "strip", data=train, height=7,aspect=3)
ax = sns.catplot(x="species_id", y="t_min",hue = "songtype_id",
                     kind = "strip", data=train, height=7,aspect=3)
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    


### LABEL DISTRIBUTION 
- Number of labels and its subsequent data is vizualized in below graph.


```python
labels = pd.DataFrame(train.groupby(["songtype_id","species_id"])["recording_id"].count().reset_index())
max_y = labels["recording_id"].max()+5
min_y = labels["recording_id"].min()-5
g = sns.catplot(x="species_id", y="recording_id", hue="songtype_id", kind="bar", data=labels, height=5, aspect = 3)
g.set(ylim=(min_y, max_y)) 
```




    <seaborn.axisgrid.FacetGrid at 0x228e56670d0>




    
![png](output_13_1.png)
    


### CATEGORICAL AND NUMERICAL DATA DISTRIBUTION


```python
import numpy as np

col = np.array(train.columns)
cat_col = []
num_col = []
for c in col:
    cname = train[c]
    if((cname.dtype == 'int64' or cname.dtype == 'float64') and c != "songtype_id" and c!="species_id"):
        num_col.append(c)
    elif(c != "songtype_id" and c!="species_id"):
        cat_col.append(c)
print("Numerical data colums name: ")
print(num_col)
print("Categorical data columns name without target: ")
print(cat_col)
```

    Numerical data colums name: 
    ['t_min', 'f_min', 't_max', 'f_max', 'is_tp']
    Categorical data columns name without target: 
    ['recording_id']
    

### CORRELATION MATRIX FOR MIN AND MAX FREQUENCIES AND TIME


```python
num = sns.heatmap(train[num_col].corr(),annot=True)
```


    
![png](output_17_0.png)
    


<a id="ss"></a>
## SIGNAL AND SPECTROGRAM

In this each step of signal processing to feature extraction is visualised.<br>
**STEP 1 Signal pre-processing**<br>
The signal is trimmed based on minimum and maximum time.


```python
recording_id = train["recording_id"][1]
species_id = train["species_id"][1]
songtype_id = train["songtype_id"][1]
start = math.floor((train["t_min"][1]-0.75)*42000)
end = math.floor((train["t_max"][1]+1.75)*42000)
file_path = "train/"+recording_id+'.flac'
Audio(file_path)
signal, sr = librosa.load(file_path,sr=SR)
plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
ld.waveplot(signal,sr)
plt.ylabel('Magnitude')
plt.title('Original Signal')

signal = signal[start:end]
print(signal)
plt.figure(figsize=(16,8))
plt.subplot(2,1,2)
ld.waveplot(signal,sr)
plt.ylabel('Magnitude')
plt.title('Processed Signal')
```

    [ 0.00236117 -0.01068376  0.00997555 ...  0.00770767  0.00947417
      0.00507605]
    




    Text(0.5, 1.0, 'Processed Signal')




    
![png](output_19_2.png)
    



    
![png](output_19_3.png)
    


**STEP2 AUDIO AUGMENTATION**<br>
The above signal is sent as an input to the following code and the time stretching methods are applied on the signal, result is shown below.


```python
recording_id = train["recording_id"][1]
species_id = train["species_id"][1]
songtype_id = train["songtype_id"][1]
file_path = "train/"+recording_id+'.flac'
signal, sr = librosa.load(file_path,sr=SR)
augmented_signal = AUDIO_AUGMENTATION(samples=signal, sample_rate=SR)


plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
ld.waveplot(signal,sr)
plt.ylabel('Magnitude')
plt.title('Original Audio')


plt.figure(figsize=(16,8))
plt.subplot(2,1,2)
ld.waveplot(augmented_signal,sr)
plt.ylabel('Magnitude')
plt.title('Augmented Audio')
```




    Text(0.5, 1.0, 'Augmented Audio')




    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    


**STEP3 PLOTTING MFCC AND LOG_MEL SPECTROGRAMS**<br>
Extracting the signal features based on the output of above step.


```python
def plot_mfcc_log_scaled_mel(df, fft=2048, n_hops = 512, sr=SR):
    i=0
    plt.subplots(4,3,figsize=(16,12))
    for _,r in df.iterrows():
        recording_id = r["recording_id"]
        start = int(r["t_min"]*0.75)
        end = int(r["t_max"]*1.25)
        sid = r["species_id"]
        stid = r["songtype_id"]
        
        fpath = os.path.join('train/'+f'{recording_id}.flac')
        signal, sr = librosa.load(fpath, sr=sr)
        signal = signal[sr*start:sr*end]
        
        plt.subplot(4,3,i+1)
        ld.waveplot(signal,sr)
        plt.title(f'Species Id {sid} SongType Id {stid}')
        plt.ylabel('Magnitude')
        
        
        plt.subplot(4,3,i+2)
        mfcc = librosa.feature.mfcc(signal, n_fft = fft, hop_length = n_hops, n_mfcc = 30)
        img = ld.specshow(mfcc,sr=sr,hop_length=n_hops)
        plt.title('MFCC Spectogram')
        plt.colorbar(img)
       
        
        plt.subplot(4,3,i+3)
        ps = librosa.feature.melspectrogram(y=signal, sr=sr)
        ps_db= librosa.power_to_db(ps, ref=np.max)
        ld.specshow(ps_db, x_axis='time', y_axis='mel')
        plt.title('Log-scaled mel spectogram')
        
        i += 3
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)     
    plt.show()
   
```


```python
plot_mfcc_log_scaled_mel(train[:4])
```


    
![png](output_24_0.png)
    


**STEP3 PLOTTING STFT AND LOG_MEL SPECTROGRAMS**<br>
Extracting the signal features based on the output of audio augmentations step.


```python
fft = 2048
n_hops = 512
def plot_stft_log_scaled_mel(df, fft=fft, hop_length = n_hops, sr=SR):
    i=0
    plt.subplots(4,3,figsize=(16,12))
    for _,r in df.iterrows():
        recording_id = r["recording_id"]
        start = int(r["t_min"]*0.75)
        end = int(r["t_max"]*1.25)
        sid = r["species_id"]
        stid = r["songtype_id"]
        
        fpath = os.path.join('train/'+f'{recording_id}.flac')
        signal, sr = librosa.load(fpath, sr=sr)
        signal = signal[sr*start:sr*end]
        
        plt.subplot(4,3,i+1)
        ld.waveplot(signal,sr)
        plt.title(f'Species Id {sid} SongType Id {stid}')
        plt.ylabel('Magnitude')
        
        
        plt.subplot(4,3,i+2)
        stft=librosa.core.stft(signal,hop_length=n_hops,n_fft=fft)
        spectrogram=librosa.amplitude_to_db(np.abs(stft),ref=np.max)
        img=ld.specshow(spectrogram,sr=sr,hop_length=n_hops,x_axis='time',y_axis='log')
        plt.title('SHORT TIME FOURIER TRANSFORM(STFT)')
        plt.colorbar(img)
       
        
        plt.subplot(4,3,i+3)
        ps = librosa.feature.melspectrogram(y=signal, sr=sr)
        ps_db= librosa.power_to_db(ps, ref=np.max)
        ld.specshow(ps_db, x_axis='time', y_axis='mel')
        plt.title('Log-scaled mel spectogram')
        
        i += 3
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)     
    plt.show()
```


```python
plot_stft_log_scaled_mel(train[4:8])
```


    
![png](output_27_0.png)
    


**STEP3 PLOTTING STFT AND MFCC SPECTROGRAMS**<br>
Extracting the signal features based on the output of audio augmentations step.


```python
def plot_stft_mfcc(df, fft=fft, hop_length = n_hops, sr=SR):
    i=0
    plt.subplots(4,3,figsize=(16,12))
    for _,r in df.iterrows():
        recording_id = r["recording_id"]
        start = int(r["t_min"]*0.75)
        end = int(r["t_max"]*1.25)
        sid = r["species_id"]
        stid = r["songtype_id"]
        
        fpath = os.path.join('train/'+f'{recording_id}.flac')
        signal, sr = librosa.load(fpath, sr=sr)
        signal = signal[sr*start:sr*end]
        
        plt.subplot(4,3,i+1)
        ld.waveplot(signal,sr)
        plt.title(f'Species Id {sid} SongType Id {stid}')
        plt.ylabel('Magnitude')
        
        
        plt.subplot(4,3,i+2)
        stft=librosa.core.stft(signal,hop_length=n_hops,n_fft=fft)
        spectrogram=librosa.amplitude_to_db(np.abs(stft),ref=np.max)
        img=ld.specshow(spectrogram,sr=sr,hop_length=n_hops,x_axis='time',y_axis='log')
        plt.title('SHORT TIME FOURIER TRANSFORM(STFT)')
        plt.colorbar(img)
       
        
        plt.subplot(4,3,i+3)
        mfcc = librosa.feature.mfcc(signal, n_fft = fft, hop_length = n_hops, n_mfcc = 30)
        img = ld.specshow(mfcc,sr=sr,hop_length=n_hops)
        plt.title('MFCC Spectogram')
        plt.colorbar(img)
        
        i += 3
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)     
    plt.show()
```


```python
plot_stft_mfcc(train[8:12])
```


    
![png](output_30_0.png)
    


<a id="constants"></a>
### IMPORTANT SOUND CONSTANTS 


```python
F_MAX = math.floor(train["f_max"].max()*1.25)
F_MIN = math.ceil(train["f_min"].min()*0.75)
SR = math.floor(F_MAX * 2.75)
T_MAX = math.floor(train["t_max"].max())
T_MIN = math.floor(train["t_min"].min())
N_HOPS = 512
N_FFT = 2048
SHAPE=(256,512)

print("MAXIMUM FREQUENCY OF SIGNAL----",F_MAX)
print("MINIMUM FREQUENCY OF SIGNAL----",F_MIN)
print("MAXIMUM TIME OF SIGNAL----",T_MAX)
print("MINIMUM TIME OF SIGNAL----",T_MIN)
print("SAMPLE RATE----",SR)
print("NUMBER OF HOPS----",N_HOPS)
print("NUMBER OF FAST FOUIER TRANSFORMATIONS----",N_FFT)
```

    MAXIMUM FREQUENCY OF SIGNAL---- 17109
    MINIMUM FREQUENCY OF SIGNAL---- 71
    MAXIMUM TIME OF SIGNAL---- 58
    MINIMUM TIME OF SIGNAL---- 0
    SAMPLE RATE---- 47049
    NUMBER OF HOPS---- 512
    NUMBER OF FAST FOUIER TRANSFORMATIONS---- 2048
    

<a id="audio"></a>
### AUDIO AUGMENTATION

Here, The audio is augmented by time stretching method, below is the sample audio file augmentention.


```python
AUDIO_AUGMENTATION = Compose([
    TimeMask(min_band_part=0.005, max_band_part=0.10, p=0.5),
    FrequencyMask(min_frequency_band=0.005, max_frequency_band=0.10, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])
```

<a id="dmtrain"></a>
### CUSTOM DATA MODEL FOR TRAINING 


```python
DATA = {
    "spectogram" : [],
    "label" : []
}
TRAIN_PATH = 'train/'
TEST_PATH = 'test/'
```

<a id="labels"></a>
### EXTRACTION OF LABELS

Here, the labels extracted are the combination of song type and species type. On total we have about 23 different types of sounds to classify.


```python
SPECIES_SONG_DICT = {}
i = 0 
for species,songType in zip(train["species_id"], train["songtype_id"]):
    if (species,songType) not in SPECIES_SONG_DICT.keys():
        SPECIES_SONG_DICT[(species,songType)] = i
        i+=1
ss_df = pd.DataFrame.from_dict(SPECIES_SONG_DICT,orient='index').reset_index()
n_classes = len(SPECIES_SONG_DICT)
print("SPECIES_ID","SONGTYPE_ID"," |Label_Class")
print("-"*40)
for d in SPECIES_SONG_DICT:
    print(f'{d}\t\t\t|\t{SPECIES_SONG_DICT.get(d)}')
print("Number of Labels",n_classes)
```

    SPECIES_ID SONGTYPE_ID  |Label_Class
    ----------------------------------------
    (21, 1)			|	0
    (8, 1)			|	1
    (4, 1)			|	2
    (22, 1)			|	3
    (23, 1)			|	4
    (10, 1)			|	5
    (2, 1)			|	6
    (1, 1)			|	7
    (11, 1)			|	8
    (19, 1)			|	9
    (20, 1)			|	10
    (17, 1)			|	11
    (9, 1)			|	12
    (15, 1)			|	13
    (16, 4)			|	14
    (6, 1)			|	15
    (14, 1)			|	16
    (12, 1)			|	17
    (17, 4)			|	18
    (0, 1)			|	19
    (18, 1)			|	20
    (7, 1)			|	21
    (13, 1)			|	22
    Number of Labels 23
    

<a id="features"></a>
### EXTRACT FEATURES FOR TRAINING
- Here, different types of spectrograms such as SFTF, MFCC, and LOG MEL are extracted after the sound files are sliced by minimum and maximum time based on the values obtained from train table and then signal is generated. 
- This signal is sent as an input to audio augmentation function and the augmented signal is generated.
- This augmented signal is then sent as an input to different spectrogram based on the stype of the function.
- The output obtained from the below funtion is a dictonary of each spectrogram and its label.


```python
def extract_features(stype,sets,df = train, path = TRAIN_PATH, sr = SR, 
                    data= DATA, audio_augment = True, 
                    ldict = SPECIES_SONG_DICT, tmin = T_MIN, tmax = T_MAX,
                    n_hops = N_HOPS, n_fft = N_FFT, shape=SHAPE):
    
    stype = stype
    specto, labels = [], []
    title = ''
    for i, r in df.iterrows():
            sigt = []
            spectogram = []
            rid = r["recording_id"]
            label = ldict.get((r["species_id"], r["songtype_id"]))
            fpath = os.path.join(path+f'{rid}.flac')
            start, end = r["t_min"], r["t_max"]
            
            buf = np.random.uniform(2,3)
            
            if start<(tmin+3):
                start = tmin
            else:
                start = math.floor(start-buf)*sr
            
            if end>(tmax-3):
                end = tmax
            else:
                end = math.ceil(end+buf)*sr
                
            signal, sr = librosa.load(fpath, sr=sr)
            
            if(len(signal[start:end]) !=0):
                signal = signal[start:end]
            
            if audio_augment == True:
                signal = AUDIO_AUGMENTATION(signal, sample_rate=sr)
            
            if stype == 'stft':
                title = 'STFTs FEATURES EXTRACTED......'
                sigt = librosa.core.stft(signal,hop_length=n_hops,n_fft=n_fft)
                spectogram=librosa.amplitude_to_db(np.abs(sigt),ref=np.max)
            elif stype == 'log-mel':
                title = 'LOG_SCALED MEL SPECTOGRAM FEATURES EXTRACTED......' 
                sigt = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=shape[0], power = 1.5)
                spectogram= librosa.power_to_db(sigt, ref=np.max)
            elif stype == 'mfcc':
                title = 'MEL FREQUENCY CEPSTRAL COEFFICIENT FEATURES EXTRACTED.....'
                spectogram = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length = n_hops, n_mfcc = 30)
            secpto_db = resize(spectogram,shape)
            secpto_db = np.stack((secpto_db,secpto_db,secpto_db))
            specto.append(secpto_db)
            labels.append(label)
     
    data['spectogram'] = specto
    data['label'] = labels
    np.save(stype+'_'+sets+'.npy', data) 
    print(title)
    

```


```python
LOG_MEL = 'log-mel'
MFCC ='mfcc'
STFT = 'stft'
```

<a id="model"></a>
# CLASSFICATION MODELS
## MODEL1 - LOG-MEL AND MFCC

- Here the train data is split into two equal parts and each part is sent to extract features.
- Train1 extracts the MFCC features and send these features to "mfcc_train1.npy" file.
- Train2 extracts the LOG_MEL features and sends it to a file 'log-mel_train2.npy'.


```python
train1, train2 = np.split(train, [int(.5*len(train))])
```


```python
extract_features(LOG_MEL,'train2',train2)
```

    LOG_SCALED MEL SPECTOGRAM FEATURES EXTRACTED......
    


```python
extract_features(MFCC,'train1',train1)
```

    MEL FREQUENCY CEPSTRAL COEFFICIENT FEATURES EXTRACTED.....
    

#### Load feature to data model for training
- Then these files are called and the data is stored in data model "model1_train" 


```python
log_mel2 = np.load('log-mel_train2.npy',allow_pickle='TRUE').item()
mfcc1 = np.load('mfcc_train1.npy',allow_pickle='TRUE').item()

model1_train = {
    'spectogram':[],
    'label':[]
}


model1_train = log_mel2['spectogram']+mfcc1['spectogram']
model1_tlabel = log_mel2['label']+mfcc1['label']
```

#### Labels are binarized
Here the labels are encoded based on binary values.


```python
X = np.array(model1_train)
Y = model1_tlabel

Y = keras.utils.to_categorical(Y,num_classes = n_classes)
print(Y)
```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    


```python
shape = X.shape
print(shape)
```

    (50, 3, 256, 512)
    

### RESHAPING
The features are reshaped from (,3,256,512) to (,256,512,3)


```python
X = MinMaxScaler().fit_transform(X.reshape(-1,X.shape[-1])).reshape(shape)
X = X.reshape(X.shape[0],X.shape[2],X.shape[3],X.shape[1])
print(X.shape)
```

    (50, 256, 512, 3)
    

<a id="main"></a>
# MAIN MODEL FOR ALL
- All the three models used this basic model with different spectograms


```python
def build_sc_model(n_hops=N_HOPS,n_mel=256,DROPOUTout_rate=0.8):
    inp=layers.Input(shape=(X.shape[1:]))
    
    x=layers.Conv2D(32, strides=1,kernel_size=(7,7),padding='same')(inp)    
    x=layers.MaxPooling2D(pool_size=(2,2))(x)
    
        
    for i in range(len(N_FILTERS)):
        x=layers.Conv2D(N_FILTERS[i], strides=STRIDES[i],kernel_size=KERNAL_SIZE[i],padding='same')(x)     
        x=layers.MaxPooling2D(pool_size=POOL_SIZE[i])(x)
        x=layers.LeakyReLU(alpha=0.2)(x)
        x=layers.BatchNormalization()(x)
        
    x=layers.Flatten()(x)
    
    for i in range(len(DENSE_LAYERS)):
        x=layers.BatchNormalization()(x)
        x=layers.Dense(DENSE_LAYERS[i],kernel_regularizer=l2)(x)
        x=layers.Dropout(rate=DROPOUT[i])(x)
    
    x=layers.BatchNormalization()(x)
    output=layers.Dense(23,activation='softmax')(x)
    
    model=keras.Model(inputs=inp,outputs=output)
    
    return model
```

#### Required constants for model


```python
N_FILTERS=[128,256,512,1024]        
KERNAL_SIZE=[(5,5),(3,3),(3,3),(3,3)]  
STRIDES=[1,1,1,1]                  
POOL_SIZE=[(2,2),(2,2),(2,2),(2,2)]    
DENSE_LAYERS=[256,128,64]              
DROPOUT=[0.8,0.7,0.5]   
```

## Plot a graph for each model accuracy and loss


```python
def plot_model(model,specto1,specto2):
    his=pd.DataFrame(model.history)
    plt.subplots(1,2,figsize=(16,8))
    plt.subplot(1,2,1)
    plt.plot(range(len(his)),his['val_loss'],color='r',label='validation')
    plt.plot(range(len(his)),his['loss'],color='g',label='train')
    plt.xlabel("Number of Epoches")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(range(len(his)),his['val_accuracy'],color='r',label='validation')
    plt.plot(range(len(his)),his['accuracy']*2,color='g',label='train')
    plt.xlabel("Number of Epoches")
    plt.ylabel("Accuracy")
    plt.title('Accuracy')
    plt.suptitle(specto1+" and "+specto2+" Spectogram")
    plt.show()                
```

## MODEL SUMMARY 


```python
model1 = build_sc_model()
model1.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
               ,loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary()
```

    Model: "model_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_14 (InputLayer)        [(None, 256, 512, 3)]     0         
    _________________________________________________________________
    conv2d_58 (Conv2D)           (None, 256, 512, 32)      4736      
    _________________________________________________________________
    max_pooling2d_49 (MaxPooling (None, 128, 256, 32)      0         
    _________________________________________________________________
    conv2d_59 (Conv2D)           (None, 128, 256, 128)     102528    
    _________________________________________________________________
    max_pooling2d_50 (MaxPooling (None, 64, 128, 128)      0         
    _________________________________________________________________
    leaky_re_lu_40 (LeakyReLU)   (None, 64, 128, 128)      0         
    _________________________________________________________________
    batch_normalization_84 (Batc (None, 64, 128, 128)      512       
    _________________________________________________________________
    conv2d_60 (Conv2D)           (None, 64, 128, 256)      295168    
    _________________________________________________________________
    max_pooling2d_51 (MaxPooling (None, 32, 64, 256)       0         
    _________________________________________________________________
    leaky_re_lu_41 (LeakyReLU)   (None, 32, 64, 256)       0         
    _________________________________________________________________
    batch_normalization_85 (Batc (None, 32, 64, 256)       1024      
    _________________________________________________________________
    conv2d_61 (Conv2D)           (None, 32, 64, 512)       1180160   
    _________________________________________________________________
    max_pooling2d_52 (MaxPooling (None, 16, 32, 512)       0         
    _________________________________________________________________
    leaky_re_lu_42 (LeakyReLU)   (None, 16, 32, 512)       0         
    _________________________________________________________________
    batch_normalization_86 (Batc (None, 16, 32, 512)       2048      
    _________________________________________________________________
    conv2d_62 (Conv2D)           (None, 16, 32, 1024)      4719616   
    _________________________________________________________________
    max_pooling2d_53 (MaxPooling (None, 8, 16, 1024)       0         
    _________________________________________________________________
    leaky_re_lu_43 (LeakyReLU)   (None, 8, 16, 1024)       0         
    _________________________________________________________________
    batch_normalization_87 (Batc (None, 8, 16, 1024)       4096      
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 131072)            0         
    _________________________________________________________________
    batch_normalization_88 (Batc (None, 131072)            524288    
    _________________________________________________________________
    dense_36 (Dense)             (None, 256)               33554688  
    _________________________________________________________________
    dropout_35 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    batch_normalization_89 (Batc (None, 256)               1024      
    _________________________________________________________________
    dense_37 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    dropout_36 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    batch_normalization_90 (Batc (None, 128)               512       
    _________________________________________________________________
    dense_38 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    dropout_37 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    batch_normalization_91 (Batc (None, 64)                256       
    _________________________________________________________________
    dense_39 (Dense)             (None, 23)                1495      
    =================================================================
    Total params: 40,433,303
    Trainable params: 40,166,423
    Non-trainable params: 266,880
    _________________________________________________________________
    

### TRAINING MODEL 1


```python
Epochs=300
batch_size=32
filepath = "CNNModel1.hdf5"


train_model = model1.fit(X,Y,validation_split = 0.05,verbose=1, epochs = Epochs, batch_size = batch_size, shuffle = True)
```

    Epoch 1/300
    2/2 [==============================] - 25s 9s/step - loss: 5.6267 - accuracy: 0.0284 - val_loss: 5863.2129 - val_accuracy: 0.0000e+00
    Epoch 2/300
    2/2 [==============================] - 26s 8s/step - loss: 8.8803 - accuracy: 0.0880 - val_loss: 680.0760 - val_accuracy: 0.0000e+00
    Epoch 3/300
    2/2 [==============================] - 24s 8s/step - loss: 9.7171 - accuracy: 0.0000e+00 - val_loss: 4970.0371 - val_accuracy: 0.0000e+00
    Epoch 4/300
    2/2 [==============================] - 26s 11s/step - loss: 10.7053 - accuracy: 0.0388 - val_loss: 3049.1462 - val_accuracy: 0.0000e+00
    Epoch 5/300
    2/2 [==============================] - 24s 8s/step - loss: 11.8136 - accuracy: 0.0284 - val_loss: 1358.0457 - val_accuracy: 0.0000e+00
    Epoch 6/300
    2/2 [==============================] - 26s 9s/step - loss: 12.0726 - accuracy: 0.0492 - val_loss: 3966.1328 - val_accuracy: 0.0000e+00
    Epoch 7/300
    2/2 [==============================] - 26s 10s/step - loss: 12.6006 - accuracy: 0.0672 - val_loss: 3072.2432 - val_accuracy: 0.0000e+00
    Epoch 8/300
    2/2 [==============================] - 25s 8s/step - loss: 12.9343 - accuracy: 0.0918 - val_loss: 2692.2847 - val_accuracy: 0.0000e+00
    Epoch 9/300
    2/2 [==============================] - 25s 9s/step - loss: 12.9553 - accuracy: 0.0984 - val_loss: 4933.0513 - val_accuracy: 0.0000e+00
    Epoch 10/300
    2/2 [==============================] - 26s 9s/step - loss: 13.2525 - accuracy: 0.0738 - val_loss: 5258.9810 - val_accuracy: 0.0000e+00
    Epoch 11/300
    2/2 [==============================] - 25s 8s/step - loss: 13.4432 - accuracy: 0.1126 - val_loss: 4638.7339 - val_accuracy: 0.0000e+00
    Epoch 12/300
    2/2 [==============================] - 26s 10s/step - loss: 13.1731 - accuracy: 0.1230 - val_loss: 4300.3145 - val_accuracy: 0.0000e+00
    Epoch 13/300
    2/2 [==============================] - 26s 8s/step - loss: 13.0004 - accuracy: 0.1268 - val_loss: 4014.3582 - val_accuracy: 0.0000e+00
    Epoch 14/300
    2/2 [==============================] - 26s 10s/step - loss: 12.5675 - accuracy: 0.0776 - val_loss: 3711.8416 - val_accuracy: 0.0000e+00
    Epoch 15/300
    2/2 [==============================] - 26s 10s/step - loss: 12.2279 - accuracy: 0.1372 - val_loss: 3014.6160 - val_accuracy: 0.0000e+00
    Epoch 16/300
    2/2 [==============================] - 23s 8s/step - loss: 11.5312 - accuracy: 0.2498 - val_loss: 2726.1921 - val_accuracy: 0.0000e+00
    Epoch 17/300
    2/2 [==============================] - 27s 10s/step - loss: 11.7799 - accuracy: 0.0880 - val_loss: 2211.2173 - val_accuracy: 0.0000e+00
    Epoch 18/300
    2/2 [==============================] - 28s 10s/step - loss: 11.1990 - accuracy: 0.1230 - val_loss: 2466.6387 - val_accuracy: 0.0000e+00
    Epoch 19/300
    2/2 [==============================] - 24s 8s/step - loss: 10.8030 - accuracy: 0.1230 - val_loss: 1364.9497 - val_accuracy: 0.0000e+00
    Epoch 20/300
    2/2 [==============================] - 26s 10s/step - loss: 10.4184 - accuracy: 0.1372 - val_loss: 1262.7975 - val_accuracy: 0.0000e+00
    Epoch 21/300
    2/2 [==============================] - 23s 8s/step - loss: 10.0285 - accuracy: 0.1372 - val_loss: 1409.9264 - val_accuracy: 0.0000e+00
    Epoch 22/300
    2/2 [==============================] - 26s 8s/step - loss: 9.7299 - accuracy: 0.2394 - val_loss: 1411.3877 - val_accuracy: 0.0000e+00
    Epoch 23/300
    2/2 [==============================] - 27s 10s/step - loss: 9.2933 - accuracy: 0.1514 - val_loss: 1315.9652 - val_accuracy: 0.0000e+00
    Epoch 24/300
    2/2 [==============================] - 24s 8s/step - loss: 9.0816 - accuracy: 0.1126 - val_loss: 1400.5204 - val_accuracy: 0.0000e+00
    Epoch 25/300
    2/2 [==============================] - 26s 9s/step - loss: 8.7817 - accuracy: 0.0738 - val_loss: 988.4872 - val_accuracy: 0.0000e+00
    Epoch 26/300
    2/2 [==============================] - 25s 8s/step - loss: 8.3296 - accuracy: 0.1410 - val_loss: 873.5324 - val_accuracy: 0.0000e+00
    Epoch 27/300
    2/2 [==============================] - 28s 10s/step - loss: 7.9365 - accuracy: 0.1022 - val_loss: 751.0593 - val_accuracy: 0.0000e+00
    Epoch 28/300
    2/2 [==============================] - 25s 10s/step - loss: 7.8720 - accuracy: 0.1059 - val_loss: 615.8642 - val_accuracy: 0.0000e+00
    Epoch 29/300
    2/2 [==============================] - 26s 10s/step - loss: 7.6566 - accuracy: 0.1126 - val_loss: 565.8629 - val_accuracy: 0.0000e+00
    Epoch 30/300
    2/2 [==============================] - 26s 10s/step - loss: 7.3601 - accuracy: 0.1656 - val_loss: 482.8877 - val_accuracy: 0.0000e+00
    Epoch 31/300
    2/2 [==============================] - 23s 8s/step - loss: 7.0136 - accuracy: 0.1864 - val_loss: 403.6393 - val_accuracy: 0.0000e+00
    Epoch 32/300
    2/2 [==============================] - 26s 8s/step - loss: 6.8956 - accuracy: 0.1656 - val_loss: 362.2289 - val_accuracy: 0.0000e+00
    Epoch 33/300
    2/2 [==============================] - 27s 9s/step - loss: 6.6503 - accuracy: 0.2148 - val_loss: 344.2694 - val_accuracy: 0.0000e+00
    Epoch 34/300
    2/2 [==============================] - 23s 8s/step - loss: 6.5394 - accuracy: 0.1268 - val_loss: 284.9359 - val_accuracy: 0.0000e+00
    Epoch 35/300
    2/2 [==============================] - 27s 10s/step - loss: 6.4423 - accuracy: 0.0918 - val_loss: 266.2366 - val_accuracy: 0.0000e+00
    Epoch 36/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2968 - accuracy: 0.1126 - val_loss: 342.2191 - val_accuracy: 0.0000e+00
    Epoch 37/300
    2/2 [==============================] - 26s 10s/step - loss: 6.1188 - accuracy: 0.0672 - val_loss: 320.4778 - val_accuracy: 0.0000e+00
    Epoch 38/300
    2/2 [==============================] - 26s 10s/step - loss: 5.8496 - accuracy: 0.1722 - val_loss: 252.4936 - val_accuracy: 0.0000e+00
    Epoch 39/300
    2/2 [==============================] - 24s 8s/step - loss: 5.9768 - accuracy: 0.1022 - val_loss: 223.7649 - val_accuracy: 0.0000e+00
    Epoch 40/300
    2/2 [==============================] - 27s 10s/step - loss: 5.3821 - accuracy: 0.1514 - val_loss: 184.4040 - val_accuracy: 0.0000e+00
    Epoch 41/300
    2/2 [==============================] - 25s 10s/step - loss: 5.5514 - accuracy: 0.1410 - val_loss: 155.9400 - val_accuracy: 0.0000e+00
    Epoch 42/300
    2/2 [==============================] - 24s 8s/step - loss: 5.6786 - accuracy: 0.1126 - val_loss: 178.8707 - val_accuracy: 0.0000e+00
    Epoch 43/300
    2/2 [==============================] - 24s 8s/step - loss: 5.5779 - accuracy: 0.1268 - val_loss: 148.8793 - val_accuracy: 0.0000e+00
    Epoch 44/300
    2/2 [==============================] - 25s 8s/step - loss: 5.3131 - accuracy: 0.1372 - val_loss: 158.1053 - val_accuracy: 0.0000e+00
    Epoch 45/300
    2/2 [==============================] - 27s 10s/step - loss: 5.2202 - accuracy: 0.1372 - val_loss: 111.9623 - val_accuracy: 0.0000e+00
    Epoch 46/300
    2/2 [==============================] - 24s 8s/step - loss: 5.3323 - accuracy: 0.0880 - val_loss: 186.8592 - val_accuracy: 0.0000e+00
    Epoch 47/300
    2/2 [==============================] - 26s 10s/step - loss: 5.3116 - accuracy: 0.1410 - val_loss: 157.7860 - val_accuracy: 0.0000e+00
    Epoch 48/300
    2/2 [==============================] - 25s 10s/step - loss: 5.0409 - accuracy: 0.1656 - val_loss: 139.1844 - val_accuracy: 0.0000e+00
    Epoch 49/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9916 - accuracy: 0.1514 - val_loss: 114.9856 - val_accuracy: 0.0000e+00
    Epoch 50/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9021 - accuracy: 0.1410 - val_loss: 97.2634 - val_accuracy: 0.0000e+00
    Epoch 51/300
    2/2 [==============================] - 26s 10s/step - loss: 4.8029 - accuracy: 0.1372 - val_loss: 113.0320 - val_accuracy: 0.0000e+00
    Epoch 52/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7074 - accuracy: 0.1372 - val_loss: 72.3059 - val_accuracy: 0.0000e+00
    Epoch 53/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6813 - accuracy: 0.0880 - val_loss: 2059.6238 - val_accuracy: 0.0000e+00
    Epoch 54/300
    2/2 [==============================] - 26s 10s/step - loss: 5.7121 - accuracy: 0.1268 - val_loss: 1370.2120 - val_accuracy: 0.0000e+00
    Epoch 55/300
    2/2 [==============================] - 25s 10s/step - loss: 5.3068 - accuracy: 0.1656 - val_loss: 1164.0161 - val_accuracy: 0.0000e+00
    Epoch 56/300
    2/2 [==============================] - 26s 10s/step - loss: 5.2120 - accuracy: 0.1693 - val_loss: 1150.0562 - val_accuracy: 0.0000e+00
    Epoch 57/300
    2/2 [==============================] - 26s 10s/step - loss: 4.7759 - accuracy: 0.3273 - val_loss: 990.8237 - val_accuracy: 0.0000e+00
    Epoch 58/300
    2/2 [==============================] - 26s 10s/step - loss: 5.1599 - accuracy: 0.1164 - val_loss: 983.3712 - val_accuracy: 0.0000e+00
    Epoch 59/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9356 - accuracy: 0.1230 - val_loss: 803.9631 - val_accuracy: 0.0000e+00
    Epoch 60/300
    2/2 [==============================] - 27s 9s/step - loss: 4.9065 - accuracy: 0.1022 - val_loss: 687.5767 - val_accuracy: 0.0000e+00
    Epoch 61/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9283 - accuracy: 0.0880 - val_loss: 672.5291 - val_accuracy: 0.0000e+00
    Epoch 62/300
    2/2 [==============================] - 24s 9s/step - loss: 4.6769 - accuracy: 0.0880 - val_loss: 539.7722 - val_accuracy: 0.0000e+00
    Epoch 63/300
    2/2 [==============================] - 26s 11s/step - loss: 4.7094 - accuracy: 0.2043 - val_loss: 456.5360 - val_accuracy: 0.0000e+00
    Epoch 64/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6171 - accuracy: 0.1164 - val_loss: 355.9485 - val_accuracy: 0.0000e+00
    Epoch 65/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7083 - accuracy: 0.1514 - val_loss: 359.4804 - val_accuracy: 0.0000e+00
    Epoch 66/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5695 - accuracy: 0.2148 - val_loss: 285.7278 - val_accuracy: 0.0000e+00
    Epoch 67/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6748 - accuracy: 0.1835 - val_loss: 255.3992 - val_accuracy: 0.0000e+00
    Epoch 68/300
    2/2 [==============================] - 27s 10s/step - loss: 4.4340 - accuracy: 0.1797 - val_loss: 269.1421 - val_accuracy: 0.0000e+00
    Epoch 69/300
    2/2 [==============================] - 25s 9s/step - loss: 4.2882 - accuracy: 0.1722 - val_loss: 215.8890 - val_accuracy: 0.0000e+00
    Epoch 70/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0215 - accuracy: 0.2214 - val_loss: 218.3090 - val_accuracy: 0.0000e+00
    Epoch 71/300
    2/2 [==============================] - 25s 8s/step - loss: 4.1829 - accuracy: 0.2356 - val_loss: 237.9075 - val_accuracy: 0.0000e+00
    Epoch 72/300
    2/2 [==============================] - 27s 10s/step - loss: 4.3534 - accuracy: 0.0776 - val_loss: 244.8477 - val_accuracy: 0.0000e+00
    Epoch 73/300
    2/2 [==============================] - 26s 10s/step - loss: 4.4945 - accuracy: 0.1305 - val_loss: 163.8647 - val_accuracy: 0.0000e+00
    Epoch 74/300
    2/2 [==============================] - 23s 8s/step - loss: 4.4701 - accuracy: 0.1722 - val_loss: 188.6324 - val_accuracy: 0.0000e+00
    Epoch 75/300
    2/2 [==============================] - 27s 10s/step - loss: 4.6208 - accuracy: 0.1059 - val_loss: 160.7884 - val_accuracy: 0.0000e+00
    Epoch 76/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2631 - accuracy: 0.1760 - val_loss: 153.7625 - val_accuracy: 0.0000e+00
    Epoch 77/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0979 - accuracy: 0.1760 - val_loss: 114.9965 - val_accuracy: 0.0000e+00
    Epoch 78/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2648 - accuracy: 0.2110 - val_loss: 88.0609 - val_accuracy: 0.0000e+00
    Epoch 79/300
    2/2 [==============================] - 27s 11s/step - loss: 4.0616 - accuracy: 0.2356 - val_loss: 87.4697 - val_accuracy: 0.0000e+00
    Epoch 80/300
    2/2 [==============================] - 26s 8s/step - loss: 4.5725 - accuracy: 0.1372 - val_loss: 97.1313 - val_accuracy: 0.0000e+00
    Epoch 81/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2472 - accuracy: 0.1022 - val_loss: 87.2097 - val_accuracy: 0.0000e+00
    Epoch 82/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2429 - accuracy: 0.1864 - val_loss: 97.7812 - val_accuracy: 0.0000e+00
    Epoch 83/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3204 - accuracy: 0.1797 - val_loss: 78.6683 - val_accuracy: 0.0000e+00
    Epoch 84/300
    2/2 [==============================] - 25s 10s/step - loss: 4.1879 - accuracy: 0.1722 - val_loss: 66.5883 - val_accuracy: 0.0000e+00
    Epoch 85/300
    2/2 [==============================] - 26s 10s/step - loss: 4.1636 - accuracy: 0.0738 - val_loss: 65.3908 - val_accuracy: 0.0000e+00
    Epoch 86/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0839 - accuracy: 0.2110 - val_loss: 63.9045 - val_accuracy: 0.0000e+00
    Epoch 87/300
    2/2 [==============================] - 26s 10s/step - loss: 4.1612 - accuracy: 0.1410 - val_loss: 54.1803 - val_accuracy: 0.0000e+00
    Epoch 88/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9618 - accuracy: 0.2848 - val_loss: 45.4602 - val_accuracy: 0.0000e+00
    Epoch 89/300
    2/2 [==============================] - 28s 10s/step - loss: 4.0694 - accuracy: 0.2043 - val_loss: 46.7014 - val_accuracy: 0.0000e+00
    Epoch 90/300
    2/2 [==============================] - 26s 10s/step - loss: 4.5490 - accuracy: 0.1722 - val_loss: 42.3941 - val_accuracy: 0.0000e+00
    Epoch 91/300
    2/2 [==============================] - 23s 8s/step - loss: 4.2649 - accuracy: 0.1797 - val_loss: 32.7163 - val_accuracy: 0.0000e+00
    Epoch 92/300
    2/2 [==============================] - 29s 11s/step - loss: 4.2081 - accuracy: 0.1126 - val_loss: 20.1534 - val_accuracy: 0.0000e+00
    Epoch 93/300
    2/2 [==============================] - 24s 8s/step - loss: 4.1908 - accuracy: 0.1618 - val_loss: 30.0121 - val_accuracy: 0.0000e+00
    Epoch 94/300
    2/2 [==============================] - 27s 9s/step - loss: 4.1493 - accuracy: 0.1022 - val_loss: 25.7044 - val_accuracy: 0.0000e+00
    Epoch 95/300
    2/2 [==============================] - 25s 8s/step - loss: 4.0308 - accuracy: 0.2602 - val_loss: 20.6479 - val_accuracy: 0.0000e+00
    Epoch 96/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9947 - accuracy: 0.1902 - val_loss: 23.0586 - val_accuracy: 0.0000e+00
    Epoch 97/300
    2/2 [==============================] - 23s 8s/step - loss: 4.1721 - accuracy: 0.1618 - val_loss: 15.6605 - val_accuracy: 0.0000e+00
    Epoch 98/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9826 - accuracy: 0.2602 - val_loss: 26.3012 - val_accuracy: 0.0000e+00
    Epoch 99/300
    2/2 [==============================] - 24s 8s/step - loss: 4.1340 - accuracy: 0.1618 - val_loss: 15.1758 - val_accuracy: 0.0000e+00
    Epoch 100/300
    2/2 [==============================] - 23s 8s/step - loss: 4.3105 - accuracy: 0.1372 - val_loss: 10.3437 - val_accuracy: 0.0000e+00
    Epoch 101/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9244 - accuracy: 0.1551 - val_loss: 14.4215 - val_accuracy: 0.0000e+00
    Epoch 102/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0422 - accuracy: 0.1410 - val_loss: 16.8127 - val_accuracy: 0.0000e+00
    Epoch 103/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9514 - accuracy: 0.0738 - val_loss: 9.2532 - val_accuracy: 0.0000e+00
    Epoch 104/300
    2/2 [==============================] - 25s 8s/step - loss: 4.4674 - accuracy: 0.1230 - val_loss: 9.9772 - val_accuracy: 0.0000e+00
    Epoch 105/300
    2/2 [==============================] - 23s 8s/step - loss: 4.4212 - accuracy: 0.1693 - val_loss: 8.8890 - val_accuracy: 0.0000e+00
    Epoch 106/300
    2/2 [==============================] - 22s 8s/step - loss: 4.1488 - accuracy: 0.2252 - val_loss: 9.7617 - val_accuracy: 0.0000e+00
    Epoch 107/300
    2/2 [==============================] - 23s 8s/step - loss: 4.1675 - accuracy: 0.2043 - val_loss: 12.4018 - val_accuracy: 0.0000e+00
    Epoch 108/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9399 - accuracy: 0.1618 - val_loss: 7.5577 - val_accuracy: 0.0000e+00
    Epoch 109/300
    2/2 [==============================] - 23s 8s/step - loss: 3.8119 - accuracy: 0.2677 - val_loss: 6.2599 - val_accuracy: 0.0000e+00
    Epoch 110/300
    2/2 [==============================] - 23s 8s/step - loss: 3.6788 - accuracy: 0.2781 - val_loss: 6.3680 - val_accuracy: 0.0000e+00
    Epoch 111/300
    2/2 [==============================] - 23s 8s/step - loss: 3.5431 - accuracy: 0.1902 - val_loss: 6.3301 - val_accuracy: 0.0000e+00
    Epoch 112/300
    2/2 [==============================] - 23s 8s/step - loss: 3.8644 - accuracy: 0.1864 - val_loss: 6.4717 - val_accuracy: 0.0000e+00
    Epoch 113/300
    2/2 [==============================] - 22s 8s/step - loss: 3.6439 - accuracy: 0.2848 - val_loss: 6.4354 - val_accuracy: 0.0000e+00
    Epoch 114/300
    2/2 [==============================] - 23s 8s/step - loss: 3.6171 - accuracy: 0.2602 - val_loss: 8.1942 - val_accuracy: 0.0000e+00
    Epoch 115/300
    2/2 [==============================] - 23s 8s/step - loss: 3.7722 - accuracy: 0.1514 - val_loss: 7.8259 - val_accuracy: 0.0000e+00
    Epoch 116/300
    2/2 [==============================] - 23s 8s/step - loss: 4.2988 - accuracy: 0.2744 - val_loss: 7.6835 - val_accuracy: 0.0000e+00
    Epoch 117/300
    2/2 [==============================] - 23s 8s/step - loss: 4.3028 - accuracy: 0.2848 - val_loss: 6.7389 - val_accuracy: 0.0000e+00
    Epoch 118/300
    2/2 [==============================] - 25s 8s/step - loss: 4.2587 - accuracy: 0.1618 - val_loss: 6.5815 - val_accuracy: 0.0000e+00
    Epoch 119/300
    2/2 [==============================] - 23s 8s/step - loss: 4.1439 - accuracy: 0.1864 - val_loss: 5.4562 - val_accuracy: 0.0000e+00
    Epoch 120/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0341 - accuracy: 0.1864 - val_loss: 8.5335 - val_accuracy: 0.0000e+00
    Epoch 121/300
    2/2 [==============================] - 25s 8s/step - loss: 3.8761 - accuracy: 0.2185 - val_loss: 6.6001 - val_accuracy: 0.0000e+00
    Epoch 122/300
    2/2 [==============================] - 25s 8s/step - loss: 3.7296 - accuracy: 0.2289 - val_loss: 6.0787 - val_accuracy: 0.0000e+00
    Epoch 123/300
    2/2 [==============================] - 23s 8s/step - loss: 3.6829 - accuracy: 0.1656 - val_loss: 5.4807 - val_accuracy: 0.0000e+00
    Epoch 124/300
    2/2 [==============================] - 23s 7s/step - loss: 3.6828 - accuracy: 0.1164 - val_loss: 5.7035 - val_accuracy: 0.0000e+00
    Epoch 125/300
    2/2 [==============================] - 23s 8s/step - loss: 3.7202 - accuracy: 0.2356 - val_loss: 5.9039 - val_accuracy: 0.0000e+00
    Epoch 126/300
    2/2 [==============================] - 23s 7s/step - loss: 3.9210 - accuracy: 0.1797 - val_loss: 6.6141 - val_accuracy: 0.0000e+00
    Epoch 127/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0421 - accuracy: 0.1902 - val_loss: 7.0137 - val_accuracy: 0.0000e+00
    Epoch 128/300
    2/2 [==============================] - 23s 8s/step - loss: 3.7693 - accuracy: 0.3027 - val_loss: 5.9030 - val_accuracy: 0.0000e+00
    Epoch 129/300
    2/2 [==============================] - 23s 8s/step - loss: 3.7467 - accuracy: 0.1864 - val_loss: 6.7413 - val_accuracy: 0.0000e+00
    Epoch 130/300
    2/2 [==============================] - 23s 8s/step - loss: 3.5428 - accuracy: 0.2677 - val_loss: 6.5165 - val_accuracy: 0.0000e+00
    Epoch 131/300
    2/2 [==============================] - 23s 8s/step - loss: 3.4420 - accuracy: 0.2744 - val_loss: 6.2020 - val_accuracy: 0.0000e+00
    Epoch 132/300
    2/2 [==============================] - 23s 7s/step - loss: 3.5529 - accuracy: 0.1126 - val_loss: 76.6623 - val_accuracy: 0.0000e+00
    Epoch 133/300
    2/2 [==============================] - 23s 8s/step - loss: 5.6492 - accuracy: 0.0880 - val_loss: 37.5061 - val_accuracy: 0.0000e+00
    Epoch 134/300
    2/2 [==============================] - 23s 8s/step - loss: 5.1452 - accuracy: 0.2535 - val_loss: 35.2524 - val_accuracy: 0.0000e+00
    Epoch 135/300
    2/2 [==============================] - 24s 8s/step - loss: 5.3395 - accuracy: 0.1835 - val_loss: 34.4759 - val_accuracy: 0.0000e+00
    Epoch 136/300
    2/2 [==============================] - 23s 8s/step - loss: 5.4716 - accuracy: 0.1514 - val_loss: 20.8753 - val_accuracy: 0.0000e+00
    Epoch 137/300
    2/2 [==============================] - 26s 9s/step - loss: 5.4494 - accuracy: 0.1722 - val_loss: 13.1540 - val_accuracy: 0.0000e+00
    Epoch 138/300
    2/2 [==============================] - 26s 8s/step - loss: 5.4757 - accuracy: 0.1126 - val_loss: 14.7231 - val_accuracy: 0.0000e+00
    Epoch 139/300
    2/2 [==============================] - 27s 9s/step - loss: 5.4560 - accuracy: 0.2535 - val_loss: 10.2860 - val_accuracy: 0.0000e+00
    Epoch 140/300
    2/2 [==============================] - 25s 9s/step - loss: 5.5406 - accuracy: 0.0918 - val_loss: 11.0255 - val_accuracy: 0.0000e+00
    Epoch 141/300
    2/2 [==============================] - 25s 8s/step - loss: 5.3706 - accuracy: 0.1551 - val_loss: 13.3006 - val_accuracy: 0.0000e+00
    Epoch 142/300
    2/2 [==============================] - 25s 8s/step - loss: 5.5204 - accuracy: 0.2498 - val_loss: 10.4282 - val_accuracy: 0.0000e+00
    Epoch 143/300
    2/2 [==============================] - 27s 10s/step - loss: 5.4642 - accuracy: 0.1410 - val_loss: 10.6198 - val_accuracy: 0.0000e+00
    Epoch 144/300
    2/2 [==============================] - 25s 9s/step - loss: 5.1142 - accuracy: 0.2923 - val_loss: 8.4789 - val_accuracy: 0.3333
    Epoch 145/300
    2/2 [==============================] - 25s 7s/step - loss: 5.3008 - accuracy: 0.1589 - val_loss: 7.9615 - val_accuracy: 0.0000e+00
    Epoch 146/300
    2/2 [==============================] - 27s 10s/step - loss: 5.2423 - accuracy: 0.2602 - val_loss: 13.9474 - val_accuracy: 0.0000e+00
    Epoch 147/300
    2/2 [==============================] - 25s 8s/step - loss: 5.3915 - accuracy: 0.2394 - val_loss: 14.3836 - val_accuracy: 0.0000e+00
    Epoch 148/300
    2/2 [==============================] - 26s 9s/step - loss: 5.5062 - accuracy: 0.1902 - val_loss: 15.3134 - val_accuracy: 0.0000e+00
    Epoch 149/300
    2/2 [==============================] - 25s 9s/step - loss: 5.3965 - accuracy: 0.2148 - val_loss: 14.4942 - val_accuracy: 0.0000e+00
    Epoch 150/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9804 - accuracy: 0.3132 - val_loss: 8.1950 - val_accuracy: 0.0000e+00
    Epoch 151/300
    2/2 [==============================] - 25s 8s/step - loss: 5.2051 - accuracy: 0.1372 - val_loss: 9.3913 - val_accuracy: 0.0000e+00
    Epoch 152/300
    2/2 [==============================] - 27s 10s/step - loss: 4.9330 - accuracy: 0.1372 - val_loss: 8.2541 - val_accuracy: 0.0000e+00
    Epoch 153/300
    2/2 [==============================] - 27s 9s/step - loss: 5.2763 - accuracy: 0.1760 - val_loss: 8.4219 - val_accuracy: 0.0000e+00
    Epoch 154/300
    2/2 [==============================] - 26s 8s/step - loss: 5.2371 - accuracy: 0.2460 - val_loss: 8.9853 - val_accuracy: 0.0000e+00
    Epoch 155/300
    2/2 [==============================] - 27s 9s/step - loss: 5.1156 - accuracy: 0.1656 - val_loss: 7.4650 - val_accuracy: 0.0000e+00
    Epoch 156/300
    2/2 [==============================] - 29s 10s/step - loss: 4.8849 - accuracy: 0.2289 - val_loss: 8.3616 - val_accuracy: 0.0000e+00
    Epoch 157/300
    2/2 [==============================] - 27s 8s/step - loss: 4.5321 - accuracy: 0.2043 - val_loss: 10.9582 - val_accuracy: 0.0000e+00
    Epoch 158/300
    2/2 [==============================] - 27s 9s/step - loss: 4.5704 - accuracy: 0.1968 - val_loss: 9.6086 - val_accuracy: 0.0000e+00
    Epoch 159/300
    2/2 [==============================] - 27s 8s/step - loss: 4.8106 - accuracy: 0.2043 - val_loss: 8.2845 - val_accuracy: 0.0000e+00
    Epoch 160/300
    2/2 [==============================] - 28s 8s/step - loss: 4.6886 - accuracy: 0.2185 - val_loss: 7.5728 - val_accuracy: 0.0000e+00
    Epoch 161/300
    2/2 [==============================] - 26s 9s/step - loss: 4.6256 - accuracy: 0.2110 - val_loss: 7.6224 - val_accuracy: 0.0000e+00
    Epoch 162/300
    2/2 [==============================] - 28s 9s/step - loss: 4.4406 - accuracy: 0.1864 - val_loss: 8.2786 - val_accuracy: 0.0000e+00
    Epoch 163/300
    2/2 [==============================] - 26s 8s/step - loss: 4.3229 - accuracy: 0.3027 - val_loss: 8.7381 - val_accuracy: 0.0000e+00
    Epoch 164/300
    2/2 [==============================] - 28s 9s/step - loss: 4.4329 - accuracy: 0.1864 - val_loss: 12.0011 - val_accuracy: 0.0000e+00
    Epoch 165/300
    2/2 [==============================] - 29s 9s/step - loss: 5.2400 - accuracy: 0.2043 - val_loss: 10.9203 - val_accuracy: 0.0000e+00
    Epoch 166/300
    2/2 [==============================] - 25s 8s/step - loss: 5.0945 - accuracy: 0.1656 - val_loss: 8.8009 - val_accuracy: 0.0000e+00
    Epoch 167/300
    2/2 [==============================] - 27s 9s/step - loss: 4.7072 - accuracy: 0.1551 - val_loss: 7.9843 - val_accuracy: 0.0000e+00
    Epoch 168/300
    2/2 [==============================] - 27s 10s/step - loss: 4.7883 - accuracy: 0.2394 - val_loss: 7.2032 - val_accuracy: 0.0000e+00
    Epoch 169/300
    2/2 [==============================] - 28s 9s/step - loss: 4.5698 - accuracy: 0.2148 - val_loss: 7.2698 - val_accuracy: 0.0000e+00
    Epoch 170/300
    2/2 [==============================] - 25s 8s/step - loss: 5.0257 - accuracy: 0.1514 - val_loss: 8.7323 - val_accuracy: 0.0000e+00
    Epoch 171/300
    2/2 [==============================] - 27s 10s/step - loss: 4.9246 - accuracy: 0.2006 - val_loss: 13.1051 - val_accuracy: 0.0000e+00
    Epoch 172/300
    2/2 [==============================] - 26s 9s/step - loss: 7.5964 - accuracy: 0.1372 - val_loss: 12.0810 - val_accuracy: 0.0000e+00
    Epoch 173/300
    2/2 [==============================] - 25s 8s/step - loss: 7.0029 - accuracy: 0.1760 - val_loss: 10.1249 - val_accuracy: 0.0000e+00
    Epoch 174/300
    2/2 [==============================] - 27s 9s/step - loss: 6.4923 - accuracy: 0.2148 - val_loss: 10.3707 - val_accuracy: 0.0000e+00
    Epoch 175/300
    2/2 [==============================] - 26s 8s/step - loss: 6.1168 - accuracy: 0.1514 - val_loss: 12.0108 - val_accuracy: 0.0000e+00
    Epoch 176/300
    2/2 [==============================] - 27s 9s/step - loss: 5.8218 - accuracy: 0.2252 - val_loss: 9.5655 - val_accuracy: 0.0000e+00
    Epoch 177/300
    2/2 [==============================] - 27s 10s/step - loss: 5.7089 - accuracy: 0.2185 - val_loss: 8.2584 - val_accuracy: 0.0000e+00
    Epoch 178/300
    2/2 [==============================] - 28s 9s/step - loss: 5.5764 - accuracy: 0.1760 - val_loss: 8.2701 - val_accuracy: 0.0000e+00
    Epoch 179/300
    2/2 [==============================] - 27s 8s/step - loss: 5.3229 - accuracy: 0.2252 - val_loss: 8.3388 - val_accuracy: 0.0000e+00
    Epoch 180/300
    2/2 [==============================] - 27s 9s/step - loss: 4.7995 - accuracy: 0.2006 - val_loss: 8.5588 - val_accuracy: 0.0000e+00
    Epoch 181/300
    2/2 [==============================] - 27s 9s/step - loss: 4.5866 - accuracy: 0.2252 - val_loss: 171.3251 - val_accuracy: 0.0000e+00
    Epoch 182/300
    2/2 [==============================] - 27s 8s/step - loss: 6.5870 - accuracy: 0.0738 - val_loss: 57.3646 - val_accuracy: 0.0000e+00
    Epoch 183/300
    2/2 [==============================] - 28s 9s/step - loss: 7.0749 - accuracy: 0.0000e+00 - val_loss: 32.2713 - val_accuracy: 0.0000e+00
    Epoch 184/300
    2/2 [==============================] - 27s 9s/step - loss: 6.7051 - accuracy: 0.1372 - val_loss: 22.3782 - val_accuracy: 0.0000e+00
    Epoch 185/300
    2/2 [==============================] - 28s 9s/step - loss: 6.6870 - accuracy: 0.1902 - val_loss: 14.2800 - val_accuracy: 0.0000e+00
    Epoch 186/300
    2/2 [==============================] - 27s 9s/step - loss: 6.4544 - accuracy: 0.3027 - val_loss: 11.9447 - val_accuracy: 0.0000e+00
    Epoch 187/300
    2/2 [==============================] - 29s 10s/step - loss: 6.1477 - accuracy: 0.0880 - val_loss: 12.0927 - val_accuracy: 0.0000e+00
    Epoch 188/300
    2/2 [==============================] - 28s 8s/step - loss: 5.8455 - accuracy: 0.2640 - val_loss: 10.8589 - val_accuracy: 0.0000e+00
    Epoch 189/300
    2/2 [==============================] - 30s 10s/step - loss: 5.6364 - accuracy: 0.2289 - val_loss: 39.5677 - val_accuracy: 0.3333
    Epoch 190/300
    2/2 [==============================] - 27s 8s/step - loss: 6.9994 - accuracy: 0.1022 - val_loss: 36.7721 - val_accuracy: 0.3333
    Epoch 191/300
    2/2 [==============================] - 29s 10s/step - loss: 7.3988 - accuracy: 0.1372 - val_loss: 15.2556 - val_accuracy: 0.0000e+00
    Epoch 192/300
    2/2 [==============================] - 29s 9s/step - loss: 6.8539 - accuracy: 0.1410 - val_loss: 17.4604 - val_accuracy: 0.0000e+00
    Epoch 193/300
    2/2 [==============================] - 27s 9s/step - loss: 6.4927 - accuracy: 0.1618 - val_loss: 12.1346 - val_accuracy: 0.0000e+00
    Epoch 194/300
    2/2 [==============================] - 30s 10s/step - loss: 6.3661 - accuracy: 0.1022 - val_loss: 11.8640 - val_accuracy: 0.0000e+00
    Epoch 195/300
    2/2 [==============================] - 29s 9s/step - loss: 5.8099 - accuracy: 0.1939 - val_loss: 13.3952 - val_accuracy: 0.0000e+00
    Epoch 196/300
    2/2 [==============================] - 29s 9s/step - loss: 5.6123 - accuracy: 0.2252 - val_loss: 18.7370 - val_accuracy: 0.0000e+00
    Epoch 197/300
    2/2 [==============================] - 27s 9s/step - loss: 5.5439 - accuracy: 0.1968 - val_loss: 15.9710 - val_accuracy: 0.0000e+00
    Epoch 198/300
    2/2 [==============================] - 26s 8s/step - loss: 5.3043 - accuracy: 0.1797 - val_loss: 14.4571 - val_accuracy: 0.0000e+00
    Epoch 199/300
    2/2 [==============================] - 29s 9s/step - loss: 4.9988 - accuracy: 0.2640 - val_loss: 16.6916 - val_accuracy: 0.3333
    Epoch 200/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9497 - accuracy: 0.2110 - val_loss: 13.2795 - val_accuracy: 0.3333
    Epoch 201/300
    2/2 [==============================] - 25s 9s/step - loss: 5.1141 - accuracy: 0.1618 - val_loss: 20.6689 - val_accuracy: 0.3333
    Epoch 202/300
    2/2 [==============================] - 25s 9s/step - loss: 4.7767 - accuracy: 0.1722 - val_loss: 26.6118 - val_accuracy: 0.0000e+00
    Epoch 203/300
    2/2 [==============================] - 28s 9s/step - loss: 5.0165 - accuracy: 0.2110 - val_loss: 20.1840 - val_accuracy: 0.0000e+00
    Epoch 204/300
    2/2 [==============================] - 25s 8s/step - loss: 4.9917 - accuracy: 0.2781 - val_loss: 19.8933 - val_accuracy: 0.0000e+00
    Epoch 205/300
    2/2 [==============================] - 27s 8s/step - loss: 4.7564 - accuracy: 0.1902 - val_loss: 15.4203 - val_accuracy: 0.0000e+00
    Epoch 206/300
    2/2 [==============================] - 27s 10s/step - loss: 4.3759 - accuracy: 0.2394 - val_loss: 12.8344 - val_accuracy: 0.0000e+00
    Epoch 207/300
    2/2 [==============================] - 29s 10s/step - loss: 4.5225 - accuracy: 0.2006 - val_loss: 19.9498 - val_accuracy: 0.0000e+00
    Epoch 208/300
    2/2 [==============================] - 30s 10s/step - loss: 4.4723 - accuracy: 0.1410 - val_loss: 18.1108 - val_accuracy: 0.0000e+00
    Epoch 209/300
    2/2 [==============================] - 31s 9s/step - loss: 4.2479 - accuracy: 0.2148 - val_loss: 16.6874 - val_accuracy: 0.0000e+00
    Epoch 210/300
    2/2 [==============================] - 28s 10s/step - loss: 4.2424 - accuracy: 0.1797 - val_loss: 15.9626 - val_accuracy: 0.0000e+00
    Epoch 211/300
    2/2 [==============================] - 31s 11s/step - loss: 4.2648 - accuracy: 0.2602 - val_loss: 14.2873 - val_accuracy: 0.0000e+00
    Epoch 212/300
    2/2 [==============================] - 28s 8s/step - loss: 4.1443 - accuracy: 0.2640 - val_loss: 12.8457 - val_accuracy: 0.0000e+00
    Epoch 213/300
    2/2 [==============================] - 25s 8s/step - loss: 4.3871 - accuracy: 0.1864 - val_loss: 9.9956 - val_accuracy: 0.0000e+00
    Epoch 214/300
    2/2 [==============================] - 23s 7s/step - loss: 5.2371 - accuracy: 0.2289 - val_loss: 9.2261 - val_accuracy: 0.0000e+00
    Epoch 215/300
    2/2 [==============================] - 25s 8s/step - loss: 4.7067 - accuracy: 0.1722 - val_loss: 11.9278 - val_accuracy: 0.0000e+00
    Epoch 216/300
    2/2 [==============================] - 28s 10s/step - loss: 4.5631 - accuracy: 0.1693 - val_loss: 8.8672 - val_accuracy: 0.0000e+00
    Epoch 217/300
    2/2 [==============================] - 27s 9s/step - loss: 4.2623 - accuracy: 0.2394 - val_loss: 9.3883 - val_accuracy: 0.0000e+00
    Epoch 218/300
    2/2 [==============================] - 29s 11s/step - loss: 3.9742 - accuracy: 0.2289 - val_loss: 9.8557 - val_accuracy: 0.0000e+00
    Epoch 219/300
    2/2 [==============================] - 32s 10s/step - loss: 4.0702 - accuracy: 0.2148 - val_loss: 6.7810 - val_accuracy: 0.0000e+00
    Epoch 220/300
    2/2 [==============================] - 30s 10s/step - loss: 3.9345 - accuracy: 0.1760 - val_loss: 7.9820 - val_accuracy: 0.0000e+00
    Epoch 221/300
    2/2 [==============================] - 29s 10s/step - loss: 3.8812 - accuracy: 0.2394 - val_loss: 7.6972 - val_accuracy: 0.0000e+00
    Epoch 222/300
    2/2 [==============================] - 31s 11s/step - loss: 4.2074 - accuracy: 0.2886 - val_loss: 7.3463 - val_accuracy: 0.0000e+00
    Epoch 223/300
    2/2 [==============================] - 29s 10s/step - loss: 4.2825 - accuracy: 0.2006 - val_loss: 8.3345 - val_accuracy: 0.0000e+00
    Epoch 224/300
    2/2 [==============================] - 29s 10s/step - loss: 4.2537 - accuracy: 0.1514 - val_loss: 6.3727 - val_accuracy: 0.0000e+00
    Epoch 225/300
    2/2 [==============================] - 29s 10s/step - loss: 4.0452 - accuracy: 0.1722 - val_loss: 7.1568 - val_accuracy: 0.0000e+00
    Epoch 226/300
    2/2 [==============================] - 29s 10s/step - loss: 3.8881 - accuracy: 0.1902 - val_loss: 7.8820 - val_accuracy: 0.0000e+00
    Epoch 227/300
    2/2 [==============================] - 27s 8s/step - loss: 3.8050 - accuracy: 0.1864 - val_loss: 6.8046 - val_accuracy: 0.0000e+00
    Epoch 228/300
    2/2 [==============================] - 27s 8s/step - loss: 3.7857 - accuracy: 0.2327 - val_loss: 6.2058 - val_accuracy: 0.0000e+00
    Epoch 229/300
    2/2 [==============================] - 27s 8s/step - loss: 3.6085 - accuracy: 0.2327 - val_loss: 6.0260 - val_accuracy: 0.0000e+00
    Epoch 230/300
    2/2 [==============================] - 25s 8s/step - loss: 3.4124 - accuracy: 0.3027 - val_loss: 5.3185 - val_accuracy: 0.0000e+00
    Epoch 231/300
    2/2 [==============================] - 27s 10s/step - loss: 3.2402 - accuracy: 0.3169 - val_loss: 6.8942 - val_accuracy: 0.0000e+00
    Epoch 232/300
    2/2 [==============================] - 31s 12s/step - loss: 3.6845 - accuracy: 0.1939 - val_loss: 5.8170 - val_accuracy: 0.3333
    Epoch 233/300
    2/2 [==============================] - 29s 10s/step - loss: 3.7404 - accuracy: 0.1760 - val_loss: 7.6226 - val_accuracy: 0.3333
    Epoch 234/300
    2/2 [==============================] - 29s 10s/step - loss: 3.6908 - accuracy: 0.1939 - val_loss: 6.5606 - val_accuracy: 0.3333
    Epoch 235/300
    2/2 [==============================] - 28s 9s/step - loss: 3.5359 - accuracy: 0.1618 - val_loss: 7.5531 - val_accuracy: 0.3333
    Epoch 236/300
    2/2 [==============================] - 27s 8s/step - loss: 3.7755 - accuracy: 0.3169 - val_loss: 6.4504 - val_accuracy: 0.3333
    Epoch 237/300
    2/2 [==============================] - 25s 8s/step - loss: 3.7769 - accuracy: 0.1939 - val_loss: 6.8759 - val_accuracy: 0.3333
    Epoch 238/300
    2/2 [==============================] - 24s 8s/step - loss: 3.5553 - accuracy: 0.2043 - val_loss: 6.7122 - val_accuracy: 0.3333
    Epoch 239/300
    2/2 [==============================] - 25s 8s/step - loss: 3.6144 - accuracy: 0.2394 - val_loss: 6.1634 - val_accuracy: 0.3333
    Epoch 240/300
    2/2 [==============================] - 24s 8s/step - loss: 3.5709 - accuracy: 0.1797 - val_loss: 5.0538 - val_accuracy: 0.0000e+00
    Epoch 241/300
    2/2 [==============================] - 25s 8s/step - loss: 3.6584 - accuracy: 0.1268 - val_loss: 5.6515 - val_accuracy: 0.0000e+00
    Epoch 242/300
    2/2 [==============================] - 25s 10s/step - loss: 3.7651 - accuracy: 0.1410 - val_loss: 5.9622 - val_accuracy: 0.0000e+00
    Epoch 243/300
    2/2 [==============================] - 23s 7s/step - loss: 3.5754 - accuracy: 0.2781 - val_loss: 5.2097 - val_accuracy: 0.0000e+00
    Epoch 244/300
    2/2 [==============================] - 25s 10s/step - loss: 3.3528 - accuracy: 0.2848 - val_loss: 5.2215 - val_accuracy: 0.0000e+00
    Epoch 245/300
    2/2 [==============================] - 25s 10s/step - loss: 3.6239 - accuracy: 0.2394 - val_loss: 5.2253 - val_accuracy: 0.3333
    Epoch 246/300
    2/2 [==============================] - 25s 10s/step - loss: 3.6851 - accuracy: 0.2289 - val_loss: 5.6341 - val_accuracy: 0.0000e+00
    Epoch 247/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8530 - accuracy: 0.1268 - val_loss: 5.1206 - val_accuracy: 0.0000e+00
    Epoch 248/300
    2/2 [==============================] - 26s 10s/step - loss: 3.3969 - accuracy: 0.2923 - val_loss: 5.2068 - val_accuracy: 0.3333
    Epoch 249/300
    2/2 [==============================] - 25s 10s/step - loss: 3.7683 - accuracy: 0.1760 - val_loss: 5.0811 - val_accuracy: 0.0000e+00
    Epoch 250/300
    2/2 [==============================] - 25s 10s/step - loss: 3.7268 - accuracy: 0.1760 - val_loss: 5.2277 - val_accuracy: 0.3333
    Epoch 251/300
    2/2 [==============================] - 25s 10s/step - loss: 3.9886 - accuracy: 0.1410 - val_loss: 6.9670 - val_accuracy: 0.0000e+00
    Epoch 252/300
    2/2 [==============================] - 25s 8s/step - loss: 4.1189 - accuracy: 0.2043 - val_loss: 9.1880 - val_accuracy: 0.3333
    Epoch 253/300
    2/2 [==============================] - 25s 10s/step - loss: 4.1858 - accuracy: 0.1977 - val_loss: 8.8310 - val_accuracy: 0.3333
    Epoch 254/300
    2/2 [==============================] - 25s 10s/step - loss: 3.9220 - accuracy: 0.2043 - val_loss: 11.1834 - val_accuracy: 0.3333
    Epoch 255/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8363 - accuracy: 0.3169 - val_loss: 7.1295 - val_accuracy: 0.0000e+00
    Epoch 256/300
    2/2 [==============================] - 25s 9s/step - loss: 3.7673 - accuracy: 0.2819 - val_loss: 7.9894 - val_accuracy: 0.0000e+00
    Epoch 257/300
    2/2 [==============================] - 25s 10s/step - loss: 3.9001 - accuracy: 0.2886 - val_loss: 6.4894 - val_accuracy: 0.0000e+00
    Epoch 258/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8276 - accuracy: 0.3378 - val_loss: 7.5134 - val_accuracy: 0.0000e+00
    Epoch 259/300
    2/2 [==============================] - 25s 10s/step - loss: 5.6313 - accuracy: 0.2006 - val_loss: 7.5355 - val_accuracy: 0.0000e+00
    Epoch 260/300
    2/2 [==============================] - 24s 9s/step - loss: 5.6279 - accuracy: 0.3027 - val_loss: 9.3331 - val_accuracy: 0.0000e+00
    Epoch 261/300
    2/2 [==============================] - 25s 10s/step - loss: 5.6047 - accuracy: 0.1797 - val_loss: 7.9717 - val_accuracy: 0.0000e+00
    Epoch 262/300
    2/2 [==============================] - 25s 10s/step - loss: 5.4721 - accuracy: 0.2706 - val_loss: 9.5335 - val_accuracy: 0.0000e+00
    Epoch 263/300
    2/2 [==============================] - 25s 10s/step - loss: 5.5562 - accuracy: 0.1514 - val_loss: 8.9101 - val_accuracy: 0.0000e+00
    Epoch 264/300
    2/2 [==============================] - 25s 10s/step - loss: 5.4632 - accuracy: 0.2781 - val_loss: 7.4789 - val_accuracy: 0.0000e+00
    Epoch 265/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9504 - accuracy: 0.4153 - val_loss: 8.6888 - val_accuracy: 0.3333
    Epoch 266/300
    2/2 [==============================] - 26s 10s/step - loss: 4.8052 - accuracy: 0.2744 - val_loss: 10.6923 - val_accuracy: 0.3333
    Epoch 267/300
    2/2 [==============================] - 25s 10s/step - loss: 5.2321 - accuracy: 0.1797 - val_loss: 10.0433 - val_accuracy: 0.0000e+00
    Epoch 268/300
    2/2 [==============================] - 26s 8s/step - loss: 5.4241 - accuracy: 0.2148 - val_loss: 9.7213 - val_accuracy: 0.0000e+00
    Epoch 269/300
    2/2 [==============================] - 28s 10s/step - loss: 4.7965 - accuracy: 0.3520 - val_loss: 7.5743 - val_accuracy: 0.0000e+00
    Epoch 270/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6318 - accuracy: 0.2252 - val_loss: 9.1596 - val_accuracy: 0.3333
    Epoch 271/300
    2/2 [==============================] - 26s 10s/step - loss: 4.2834 - accuracy: 0.3766 - val_loss: 7.5464 - val_accuracy: 0.0000e+00
    Epoch 272/300
    2/2 [==============================] - 28s 9s/step - loss: 4.3061 - accuracy: 0.2110 - val_loss: 10.2622 - val_accuracy: 0.3333
    Epoch 273/300
    2/2 [==============================] - 26s 8s/step - loss: 4.0690 - accuracy: 0.2498 - val_loss: 10.1628 - val_accuracy: 0.3333
    Epoch 274/300
    2/2 [==============================] - 27s 10s/step - loss: 4.2141 - accuracy: 0.2356 - val_loss: 9.2559 - val_accuracy: 0.3333
    Epoch 275/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8988 - accuracy: 0.4191 - val_loss: 8.5531 - val_accuracy: 0.3333
    Epoch 276/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8748 - accuracy: 0.2781 - val_loss: 8.8548 - val_accuracy: 0.3333
    Epoch 277/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8513 - accuracy: 0.2781 - val_loss: 10.2892 - val_accuracy: 0.3333
    Epoch 278/300
    2/2 [==============================] - 24s 9s/step - loss: 3.8239 - accuracy: 0.3132 - val_loss: 9.4021 - val_accuracy: 0.0000e+00
    Epoch 279/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0531 - accuracy: 0.2640 - val_loss: 10.4176 - val_accuracy: 0.3333
    Epoch 280/300
    2/2 [==============================] - 25s 10s/step - loss: 3.9858 - accuracy: 0.3728 - val_loss: 7.8237 - val_accuracy: 0.0000e+00
    Epoch 281/300
    2/2 [==============================] - 27s 10s/step - loss: 4.1091 - accuracy: 0.2602 - val_loss: 11.8528 - val_accuracy: 0.3333
    Epoch 282/300
    2/2 [==============================] - 25s 9s/step - loss: 3.9587 - accuracy: 0.2781 - val_loss: 11.0331 - val_accuracy: 0.3333
    Epoch 283/300
    2/2 [==============================] - 26s 10s/step - loss: 4.2054 - accuracy: 0.3520 - val_loss: 9.7567 - val_accuracy: 0.3333
    Epoch 284/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8246 - accuracy: 0.2110 - val_loss: 7.1044 - val_accuracy: 0.0000e+00
    Epoch 285/300
    2/2 [==============================] - 25s 10s/step - loss: 3.9917 - accuracy: 0.1551 - val_loss: 6.5174 - val_accuracy: 0.0000e+00
    Epoch 286/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8335 - accuracy: 0.2148 - val_loss: 6.1323 - val_accuracy: 0.0000e+00
    Epoch 287/300
    2/2 [==============================] - 24s 9s/step - loss: 3.8793 - accuracy: 0.2431 - val_loss: 9.6648 - val_accuracy: 0.0000e+00
    Epoch 288/300
    2/2 [==============================] - 26s 10s/step - loss: 3.9102 - accuracy: 0.2677 - val_loss: 6.7646 - val_accuracy: 0.0000e+00
    Epoch 289/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3949 - accuracy: 0.2289 - val_loss: 7.8376 - val_accuracy: 0.0000e+00
    Epoch 290/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3673 - accuracy: 0.2535 - val_loss: 6.6271 - val_accuracy: 0.0000e+00
    Epoch 291/300
    2/2 [==============================] - 25s 10s/step - loss: 4.1717 - accuracy: 0.3415 - val_loss: 8.0186 - val_accuracy: 0.0000e+00
    Epoch 292/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0953 - accuracy: 0.3415 - val_loss: 7.4227 - val_accuracy: 0.0000e+00
    Epoch 293/300
    2/2 [==============================] - 25s 10s/step - loss: 3.7091 - accuracy: 0.2923 - val_loss: 7.8059 - val_accuracy: 0.0000e+00
    Epoch 294/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8306 - accuracy: 0.3169 - val_loss: 5.8854 - val_accuracy: 0.0000e+00
    Epoch 295/300
    2/2 [==============================] - 25s 10s/step - loss: 3.4387 - accuracy: 0.3378 - val_loss: 6.0323 - val_accuracy: 0.0000e+00
    Epoch 296/300
    2/2 [==============================] - 25s 10s/step - loss: 3.4012 - accuracy: 0.4399 - val_loss: 8.9728 - val_accuracy: 0.0000e+00
    Epoch 297/300
    2/2 [==============================] - 25s 10s/step - loss: 3.4635 - accuracy: 0.3340 - val_loss: 8.0041 - val_accuracy: 0.0000e+00
    Epoch 298/300
    2/2 [==============================] - 25s 10s/step - loss: 3.4115 - accuracy: 0.3027 - val_loss: 12.4728 - val_accuracy: 0.3333
    Epoch 299/300
    2/2 [==============================] - 25s 10s/step - loss: 3.7144 - accuracy: 0.3803 - val_loss: 6.3219 - val_accuracy: 0.3333
    Epoch 300/300
    2/2 [==============================] - 25s 10s/step - loss: 3.8220 - accuracy: 0.2990 - val_loss: 6.4013 - val_accuracy: 0.0000e+00
    

### Training Results for model1


```python
plot_model(train_model,"LOG_MEL","MFCC")
```


    
![png](output_64_0.png)
    


<a id="model2"></a>
## MODEL 2 - STFT AND LOG_MEL

- Here the train data is split into two equal parts and each part is sent to extract features.
- Train1 extracts the STFT features and send these features to "stft_train1.npy" file.
- Train2 extracts the LOG_MEL features and sends it to a file 'log-mel_train2.npy'.


```python
extract_features(STFT,'train1',train1)
```

    STFTs FEATURES EXTRACTED......
    

#### Load feature to data model for training
- Then these files are called and the data is stored in data model "model2_train" 


```python
log_mel2 = np.load('log-mel_train2.npy',allow_pickle='TRUE').item()
stft1 = np.load('stft_train1.npy',allow_pickle='TRUE').item()

model2_train = {
    'spectogram':[],
    'label':[]
}


model2_train = log_mel2['spectogram']+stft1['spectogram']
model2_tlabel = log_mel2['label']+stft1['label']
```

#### Labels are binarized
Here the labels are encoded based on binary values.


```python
X = np.array(model2_train)
Y = model2_tlabel

Y = keras.utils.to_categorical(Y,num_classes = n_classes)
print(Y)
```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    

#### Reshaping


```python
X = MinMaxScaler().fit_transform(X.reshape(-1,X.shape[-1])).reshape(shape)
X = X.reshape(X.shape[0],X.shape[2],X.shape[3],X.shape[1])
print(X.shape)
```

    (50, 256, 512, 3)
    

### TRAINING MODEL2


```python
model2 = build_sc_model()
model2.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
               ,loss='categorical_crossentropy',metrics=['accuracy'])
```


```python
Epochs=300
batch_size=32
filepath = "CNNModel2.hdf5"


train_model1 = model2.fit(X,Y,validation_split = 0.05,verbose=1, epochs = Epochs, batch_size = batch_size, shuffle = True)
```

    Epoch 1/300
    2/2 [==============================] - 48s 10s/step - loss: 5.5720 - accuracy: 0.0000e+00 - val_loss: 45834.7227 - val_accuracy: 0.0000e+00
    Epoch 2/300
    2/2 [==============================] - 26s 9s/step - loss: 9.2117 - accuracy: 0.0246 - val_loss: 1365.1848 - val_accuracy: 0.0000e+00
    Epoch 3/300
    2/2 [==============================] - 24s 8s/step - loss: 9.6146 - accuracy: 0.0880 - val_loss: 532.9628 - val_accuracy: 0.0000e+00
    Epoch 4/300
    2/2 [==============================] - 25s 9s/step - loss: 10.3211 - accuracy: 0.0492 - val_loss: 237.0742 - val_accuracy: 0.0000e+00
    Epoch 5/300
    2/2 [==============================] - 27s 9s/step - loss: 11.2464 - accuracy: 0.0738 - val_loss: 314.1801 - val_accuracy: 0.0000e+00
    Epoch 6/300
    2/2 [==============================] - 25s 8s/step - loss: 12.2908 - accuracy: 0.0880 - val_loss: 437.5020 - val_accuracy: 0.0000e+00
    Epoch 7/300
    2/2 [==============================] - 24s 9s/step - loss: 12.7527 - accuracy: 0.0492 - val_loss: 2674.7676 - val_accuracy: 0.0000e+00
    Epoch 8/300
    2/2 [==============================] - 26s 9s/step - loss: 13.1724 - accuracy: 0.0984 - val_loss: 4177.8813 - val_accuracy: 0.0000e+00
    Epoch 9/300
    2/2 [==============================] - 25s 8s/step - loss: 13.3095 - accuracy: 0.0880 - val_loss: 2351.9756 - val_accuracy: 0.0000e+00
    Epoch 10/300
    2/2 [==============================] - 24s 8s/step - loss: 13.4305 - accuracy: 0.0492 - val_loss: 2348.1777 - val_accuracy: 0.0000e+00
    Epoch 11/300
    2/2 [==============================] - 24s 8s/step - loss: 13.2160 - accuracy: 0.1268 - val_loss: 2052.0781 - val_accuracy: 0.0000e+00
    Epoch 12/300
    2/2 [==============================] - 24s 8s/step - loss: 13.2877 - accuracy: 0.1268 - val_loss: 1802.3925 - val_accuracy: 0.0000e+00
    Epoch 13/300
    2/2 [==============================] - 25s 8s/step - loss: 13.1485 - accuracy: 0.1230 - val_loss: 1215.2534 - val_accuracy: 0.0000e+00
    Epoch 14/300
    2/2 [==============================] - 25s 8s/step - loss: 12.6351 - accuracy: 0.1022 - val_loss: 4699.3008 - val_accuracy: 0.0000e+00
    Epoch 15/300
    2/2 [==============================] - 24s 8s/step - loss: 12.8633 - accuracy: 0.1126 - val_loss: 2516.5745 - val_accuracy: 0.0000e+00
    Epoch 16/300
    2/2 [==============================] - 23s 8s/step - loss: 12.6032 - accuracy: 0.0634 - val_loss: 1161.9551 - val_accuracy: 0.0000e+00
    Epoch 17/300
    2/2 [==============================] - 23s 8s/step - loss: 12.0337 - accuracy: 0.1372 - val_loss: 1038.5343 - val_accuracy: 0.0000e+00
    Epoch 18/300
    2/2 [==============================] - 24s 8s/step - loss: 11.9878 - accuracy: 0.0142 - val_loss: 670.8054 - val_accuracy: 0.0000e+00
    Epoch 19/300
    2/2 [==============================] - 24s 8s/step - loss: 11.2868 - accuracy: 0.1372 - val_loss: 667.0728 - val_accuracy: 0.0000e+00
    Epoch 20/300
    2/2 [==============================] - 23s 8s/step - loss: 10.9675 - accuracy: 0.1126 - val_loss: 434.4718 - val_accuracy: 0.0000e+00
    Epoch 21/300
    2/2 [==============================] - 23s 8s/step - loss: 10.7773 - accuracy: 0.0530 - val_loss: 284.9241 - val_accuracy: 0.0000e+00
    Epoch 22/300
    2/2 [==============================] - 23s 8s/step - loss: 10.3317 - accuracy: 0.0634 - val_loss: 177.9982 - val_accuracy: 0.0000e+00
    Epoch 23/300
    2/2 [==============================] - 23s 8s/step - loss: 9.9695 - accuracy: 0.1476 - val_loss: 144.4233 - val_accuracy: 0.0000e+00
    Epoch 24/300
    2/2 [==============================] - 23s 8s/step - loss: 9.5964 - accuracy: 0.0984 - val_loss: 117.0684 - val_accuracy: 0.0000e+00
    Epoch 25/300
    2/2 [==============================] - 24s 8s/step - loss: 9.4575 - accuracy: 0.0492 - val_loss: 83.0510 - val_accuracy: 0.0000e+00
    Epoch 26/300
    2/2 [==============================] - 23s 8s/step - loss: 8.7905 - accuracy: 0.1968 - val_loss: 77.7104 - val_accuracy: 0.0000e+00
    Epoch 27/300
    2/2 [==============================] - 24s 8s/step - loss: 8.9211 - accuracy: 0.0776 - val_loss: 83.3284 - val_accuracy: 0.0000e+00
    Epoch 28/300
    2/2 [==============================] - 24s 9s/step - loss: 8.5243 - accuracy: 0.1126 - val_loss: 72.7422 - val_accuracy: 0.0000e+00
    Epoch 29/300
    2/2 [==============================] - 24s 8s/step - loss: 8.4921 - accuracy: 0.1618 - val_loss: 61.8869 - val_accuracy: 0.0000e+00
    Epoch 30/300
    2/2 [==============================] - 24s 8s/step - loss: 8.0748 - accuracy: 0.1305 - val_loss: 61.3885 - val_accuracy: 0.0000e+00
    Epoch 31/300
    2/2 [==============================] - 24s 8s/step - loss: 8.0158 - accuracy: 0.1476 - val_loss: 55.7476 - val_accuracy: 0.0000e+00
    Epoch 32/300
    2/2 [==============================] - 24s 8s/step - loss: 7.7605 - accuracy: 0.0738 - val_loss: 40.5113 - val_accuracy: 0.0000e+00
    Epoch 33/300
    2/2 [==============================] - 26s 9s/step - loss: 7.3760 - accuracy: 0.1022 - val_loss: 27.7696 - val_accuracy: 0.0000e+00
    Epoch 34/300
    2/2 [==============================] - 24s 8s/step - loss: 7.5089 - accuracy: 0.1126 - val_loss: 18.5566 - val_accuracy: 0.0000e+00
    Epoch 35/300
    2/2 [==============================] - 23s 7s/step - loss: 7.4871 - accuracy: 0.0388 - val_loss: 21.1340 - val_accuracy: 0.0000e+00
    Epoch 36/300
    2/2 [==============================] - 23s 7s/step - loss: 7.0541 - accuracy: 0.1864 - val_loss: 28.1787 - val_accuracy: 0.0000e+00
    Epoch 37/300
    2/2 [==============================] - 25s 7s/step - loss: 6.7539 - accuracy: 0.1230 - val_loss: 29.2882 - val_accuracy: 0.0000e+00
    Epoch 38/300
    2/2 [==============================] - 23s 8s/step - loss: 6.6022 - accuracy: 0.1760 - val_loss: 26.4591 - val_accuracy: 0.0000e+00
    Epoch 39/300
    2/2 [==============================] - 24s 8s/step - loss: 6.3899 - accuracy: 0.1372 - val_loss: 27.9744 - val_accuracy: 0.0000e+00
    Epoch 40/300
    2/2 [==============================] - 26s 8s/step - loss: 6.2565 - accuracy: 0.1514 - val_loss: 24.6568 - val_accuracy: 0.0000e+00
    Epoch 41/300
    2/2 [==============================] - 24s 8s/step - loss: 6.2760 - accuracy: 0.1126 - val_loss: 19.2170 - val_accuracy: 0.0000e+00
    Epoch 42/300
    2/2 [==============================] - 23s 8s/step - loss: 6.0226 - accuracy: 0.2148 - val_loss: 17.6664 - val_accuracy: 0.0000e+00
    Epoch 43/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9905 - accuracy: 0.1656 - val_loss: 16.5197 - val_accuracy: 0.0000e+00
    Epoch 44/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9123 - accuracy: 0.1618 - val_loss: 18.1789 - val_accuracy: 0.0000e+00
    Epoch 45/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9739 - accuracy: 0.1760 - val_loss: 13.4574 - val_accuracy: 0.0000e+00
    Epoch 46/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9975 - accuracy: 0.2006 - val_loss: 21.8150 - val_accuracy: 0.0000e+00
    Epoch 47/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9685 - accuracy: 0.0634 - val_loss: 16.8076 - val_accuracy: 0.0000e+00
    Epoch 48/300
    2/2 [==============================] - 25s 8s/step - loss: 6.1306 - accuracy: 0.0246 - val_loss: 13.3829 - val_accuracy: 0.0000e+00
    Epoch 49/300
    2/2 [==============================] - 23s 8s/step - loss: 5.7680 - accuracy: 0.1410 - val_loss: 15.7820 - val_accuracy: 0.0000e+00
    Epoch 50/300
    2/2 [==============================] - 23s 8s/step - loss: 5.7060 - accuracy: 0.0634 - val_loss: 16.5442 - val_accuracy: 0.0000e+00
    Epoch 51/300
    2/2 [==============================] - 23s 7s/step - loss: 5.5855 - accuracy: 0.1126 - val_loss: 13.9402 - val_accuracy: 0.0000e+00
    Epoch 52/300
    2/2 [==============================] - 23s 8s/step - loss: 5.3054 - accuracy: 0.2110 - val_loss: 15.6816 - val_accuracy: 0.0000e+00
    Epoch 53/300
    2/2 [==============================] - 24s 8s/step - loss: 5.3055 - accuracy: 0.1797 - val_loss: 11.7527 - val_accuracy: 0.0000e+00
    Epoch 54/300
    2/2 [==============================] - 23s 8s/step - loss: 5.2924 - accuracy: 0.2110 - val_loss: 10.6540 - val_accuracy: 0.0000e+00
    Epoch 55/300
    2/2 [==============================] - 23s 8s/step - loss: 5.2618 - accuracy: 0.1618 - val_loss: 11.8159 - val_accuracy: 0.0000e+00
    Epoch 56/300
    2/2 [==============================] - 23s 8s/step - loss: 5.3259 - accuracy: 0.1618 - val_loss: 10.0240 - val_accuracy: 0.0000e+00
    Epoch 57/300
    2/2 [==============================] - 24s 8s/step - loss: 5.1360 - accuracy: 0.1268 - val_loss: 8.6466 - val_accuracy: 0.0000e+00
    Epoch 58/300
    2/2 [==============================] - 23s 8s/step - loss: 4.7696 - accuracy: 0.2848 - val_loss: 10.1023 - val_accuracy: 0.0000e+00
    Epoch 59/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9984 - accuracy: 0.1760 - val_loss: 8.1595 - val_accuracy: 0.0000e+00
    Epoch 60/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8566 - accuracy: 0.2148 - val_loss: 6.8587 - val_accuracy: 0.0000e+00
    Epoch 61/300
    2/2 [==============================] - 24s 8s/step - loss: 4.8745 - accuracy: 0.0880 - val_loss: 6.8410 - val_accuracy: 0.0000e+00
    Epoch 62/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8997 - accuracy: 0.1022 - val_loss: 8.1838 - val_accuracy: 0.0000e+00
    Epoch 63/300
    2/2 [==============================] - 23s 8s/step - loss: 5.1297 - accuracy: 0.1618 - val_loss: 8.8505 - val_accuracy: 0.0000e+00
    Epoch 64/300
    2/2 [==============================] - 23s 7s/step - loss: 4.8998 - accuracy: 0.2252 - val_loss: 9.8367 - val_accuracy: 0.0000e+00
    Epoch 65/300
    2/2 [==============================] - 24s 8s/step - loss: 4.6240 - accuracy: 0.3236 - val_loss: 11.9618 - val_accuracy: 0.0000e+00
    Epoch 66/300
    2/2 [==============================] - 23s 8s/step - loss: 5.0013 - accuracy: 0.1268 - val_loss: 10.1539 - val_accuracy: 0.0000e+00
    Epoch 67/300
    2/2 [==============================] - 25s 8s/step - loss: 4.8775 - accuracy: 0.1126 - val_loss: 8.0885 - val_accuracy: 0.0000e+00
    Epoch 68/300
    2/2 [==============================] - 24s 8s/step - loss: 4.6038 - accuracy: 0.2148 - val_loss: 9.4618 - val_accuracy: 0.0000e+00
    Epoch 69/300
    2/2 [==============================] - 26s 8s/step - loss: 4.6525 - accuracy: 0.1864 - val_loss: 10.9137 - val_accuracy: 0.0000e+00
    Epoch 70/300
    2/2 [==============================] - 24s 7s/step - loss: 4.4868 - accuracy: 0.2006 - val_loss: 9.8452 - val_accuracy: 0.0000e+00
    Epoch 71/300
    2/2 [==============================] - 25s 8s/step - loss: 4.8722 - accuracy: 0.0813 - val_loss: 7.6676 - val_accuracy: 0.0000e+00
    Epoch 72/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7058 - accuracy: 0.1902 - val_loss: 6.8829 - val_accuracy: 0.0000e+00
    Epoch 73/300
    2/2 [==============================] - 23s 8s/step - loss: 4.7509 - accuracy: 0.1514 - val_loss: 6.3991 - val_accuracy: 0.0000e+00
    Epoch 74/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6585 - accuracy: 0.2006 - val_loss: 6.9210 - val_accuracy: 0.0000e+00
    Epoch 75/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6786 - accuracy: 0.1693 - val_loss: 7.6609 - val_accuracy: 0.0000e+00
    Epoch 76/300
    2/2 [==============================] - 26s 10s/step - loss: 4.6639 - accuracy: 0.1372 - val_loss: 6.7330 - val_accuracy: 0.0000e+00
    Epoch 77/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3502 - accuracy: 0.2148 - val_loss: 6.7150 - val_accuracy: 0.0000e+00
    Epoch 78/300
    2/2 [==============================] - 23s 8s/step - loss: 4.2675 - accuracy: 0.2289 - val_loss: 7.0776 - val_accuracy: 0.0000e+00
    Epoch 79/300
    2/2 [==============================] - 23s 8s/step - loss: 4.5179 - accuracy: 0.2498 - val_loss: 8.5489 - val_accuracy: 0.0000e+00
    Epoch 80/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7518 - accuracy: 0.2043 - val_loss: 6.2982 - val_accuracy: 0.0000e+00
    Epoch 81/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6972 - accuracy: 0.1268 - val_loss: 7.0570 - val_accuracy: 0.0000e+00
    Epoch 82/300
    2/2 [==============================] - 25s 10s/step - loss: 5.0797 - accuracy: 0.1372 - val_loss: 7.2109 - val_accuracy: 0.0000e+00
    Epoch 83/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9906 - accuracy: 0.2356 - val_loss: 6.8382 - val_accuracy: 0.0000e+00
    Epoch 84/300
    2/2 [==============================] - 26s 10s/step - loss: 5.1603 - accuracy: 0.1372 - val_loss: 6.6278 - val_accuracy: 0.0000e+00
    Epoch 85/300
    2/2 [==============================] - 25s 10s/step - loss: 4.8325 - accuracy: 0.2148 - val_loss: 7.6294 - val_accuracy: 0.0000e+00
    Epoch 86/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9426 - accuracy: 0.2006 - val_loss: 13.5214 - val_accuracy: 0.0000e+00
    Epoch 87/300
    2/2 [==============================] - 23s 8s/step - loss: 5.1724 - accuracy: 0.3027 - val_loss: 8.5730 - val_accuracy: 0.0000e+00
    Epoch 88/300
    2/2 [==============================] - 23s 8s/step - loss: 5.2475 - accuracy: 0.1268 - val_loss: 8.5487 - val_accuracy: 0.0000e+00
    Epoch 89/300
    2/2 [==============================] - 25s 8s/step - loss: 5.1010 - accuracy: 0.1618 - val_loss: 9.6333 - val_accuracy: 0.0000e+00
    Epoch 90/300
    2/2 [==============================] - 23s 8s/step - loss: 5.0279 - accuracy: 0.0984 - val_loss: 8.5067 - val_accuracy: 0.0000e+00
    Epoch 91/300
    2/2 [==============================] - 24s 8s/step - loss: 4.9382 - accuracy: 0.1864 - val_loss: 8.6056 - val_accuracy: 0.0000e+00
    Epoch 92/300
    2/2 [==============================] - 25s 10s/step - loss: 5.0142 - accuracy: 0.1760 - val_loss: 17.9994 - val_accuracy: 0.0000e+00
    Epoch 93/300
    2/2 [==============================] - 23s 8s/step - loss: 6.1692 - accuracy: 0.1022 - val_loss: 8.9273 - val_accuracy: 0.0000e+00
    Epoch 94/300
    2/2 [==============================] - 25s 10s/step - loss: 6.0500 - accuracy: 0.1760 - val_loss: 10.6550 - val_accuracy: 0.0000e+00
    Epoch 95/300
    2/2 [==============================] - 24s 8s/step - loss: 5.6528 - accuracy: 0.2602 - val_loss: 9.5290 - val_accuracy: 0.0000e+00
    Epoch 96/300
    2/2 [==============================] - 25s 8s/step - loss: 5.7984 - accuracy: 0.0984 - val_loss: 10.4331 - val_accuracy: 0.0000e+00
    Epoch 97/300
    2/2 [==============================] - 25s 10s/step - loss: 5.5637 - accuracy: 0.0634 - val_loss: 11.0530 - val_accuracy: 0.0000e+00
    Epoch 98/300
    2/2 [==============================] - 25s 10s/step - loss: 5.6429 - accuracy: 0.1410 - val_loss: 9.2296 - val_accuracy: 0.0000e+00
    Epoch 99/300
    2/2 [==============================] - 23s 8s/step - loss: 5.1410 - accuracy: 0.1372 - val_loss: 7.6777 - val_accuracy: 0.0000e+00
    Epoch 100/300
    2/2 [==============================] - 27s 10s/step - loss: 5.1277 - accuracy: 0.2214 - val_loss: 7.4695 - val_accuracy: 0.0000e+00
    Epoch 101/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8839 - accuracy: 0.2640 - val_loss: 7.6065 - val_accuracy: 0.0000e+00
    Epoch 102/300
    2/2 [==============================] - 24s 8s/step - loss: 4.8971 - accuracy: 0.1939 - val_loss: 7.5674 - val_accuracy: 0.0000e+00
    Epoch 103/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8175 - accuracy: 0.2677 - val_loss: 7.7755 - val_accuracy: 0.0000e+00
    Epoch 104/300
    2/2 [==============================] - 27s 10s/step - loss: 5.1332 - accuracy: 0.1126 - val_loss: 7.8308 - val_accuracy: 0.0000e+00
    Epoch 105/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7794 - accuracy: 0.1797 - val_loss: 8.3241 - val_accuracy: 0.0000e+00
    Epoch 106/300
    2/2 [==============================] - 24s 8s/step - loss: 4.8387 - accuracy: 0.1551 - val_loss: 9.6602 - val_accuracy: 0.0000e+00
    Epoch 107/300
    2/2 [==============================] - 25s 10s/step - loss: 4.8693 - accuracy: 0.1618 - val_loss: 8.1578 - val_accuracy: 0.0000e+00
    Epoch 108/300
    2/2 [==============================] - 26s 10s/step - loss: 4.5748 - accuracy: 0.3378 - val_loss: 7.7299 - val_accuracy: 0.0000e+00
    Epoch 109/300
    2/2 [==============================] - 24s 8s/step - loss: 4.5842 - accuracy: 0.2148 - val_loss: 7.3746 - val_accuracy: 0.0000e+00
    Epoch 110/300
    2/2 [==============================] - 24s 9s/step - loss: 4.5151 - accuracy: 0.1760 - val_loss: 7.5635 - val_accuracy: 0.0000e+00
    Epoch 111/300
    2/2 [==============================] - 24s 9s/step - loss: 4.3537 - accuracy: 0.1902 - val_loss: 6.2079 - val_accuracy: 0.0000e+00
    Epoch 112/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3958 - accuracy: 0.1372 - val_loss: 7.8524 - val_accuracy: 0.0000e+00
    Epoch 113/300
    2/2 [==============================] - 23s 7s/step - loss: 4.3025 - accuracy: 0.2214 - val_loss: 7.1093 - val_accuracy: 0.0000e+00
    Epoch 114/300
    2/2 [==============================] - 24s 8s/step - loss: 4.3678 - accuracy: 0.2252 - val_loss: 6.8752 - val_accuracy: 0.0000e+00
    Epoch 115/300
    2/2 [==============================] - 24s 9s/step - loss: 4.2880 - accuracy: 0.1722 - val_loss: 7.8967 - val_accuracy: 0.0000e+00
    Epoch 116/300
    2/2 [==============================] - 22s 7s/step - loss: 4.2807 - accuracy: 0.2110 - val_loss: 8.5456 - val_accuracy: 0.0000e+00
    Epoch 117/300
    2/2 [==============================] - 27s 9s/step - loss: 4.2710 - accuracy: 0.1230 - val_loss: 6.7846 - val_accuracy: 0.0000e+00
    Epoch 118/300
    2/2 [==============================] - 22s 8s/step - loss: 4.0427 - accuracy: 0.2289 - val_loss: 6.6079 - val_accuracy: 0.0000e+00
    Epoch 119/300
    2/2 [==============================] - 26s 9s/step - loss: 4.0090 - accuracy: 0.2289 - val_loss: 7.4491 - val_accuracy: 0.0000e+00
    Epoch 120/300
    2/2 [==============================] - 24s 10s/step - loss: 3.8082 - accuracy: 0.2923 - val_loss: 7.3343 - val_accuracy: 0.0000e+00
    Epoch 121/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0379 - accuracy: 0.1902 - val_loss: 7.3034 - val_accuracy: 0.0000e+00
    Epoch 122/300
    2/2 [==============================] - 24s 9s/step - loss: 4.0858 - accuracy: 0.1693 - val_loss: 7.5520 - val_accuracy: 0.0000e+00
    Epoch 123/300
    2/2 [==============================] - 24s 9s/step - loss: 4.1256 - accuracy: 0.1760 - val_loss: 7.4454 - val_accuracy: 0.0000e+00
    Epoch 124/300
    2/2 [==============================] - 25s 10s/step - loss: 4.1536 - accuracy: 0.2289 - val_loss: 8.4904 - val_accuracy: 0.0000e+00
    Epoch 125/300
    2/2 [==============================] - 24s 9s/step - loss: 4.4557 - accuracy: 0.1268 - val_loss: 8.3052 - val_accuracy: 0.0000e+00
    Epoch 126/300
    2/2 [==============================] - 24s 9s/step - loss: 4.3046 - accuracy: 0.2990 - val_loss: 9.5601 - val_accuracy: 0.0000e+00
    Epoch 127/300
    2/2 [==============================] - 24s 10s/step - loss: 4.4794 - accuracy: 0.1410 - val_loss: 10.5070 - val_accuracy: 0.0000e+00
    Epoch 128/300
    2/2 [==============================] - 22s 8s/step - loss: 4.5945 - accuracy: 0.1514 - val_loss: 11.5071 - val_accuracy: 0.0000e+00
    Epoch 129/300
    2/2 [==============================] - 24s 7s/step - loss: 4.7570 - accuracy: 0.2043 - val_loss: 7.9993 - val_accuracy: 0.0000e+00
    Epoch 130/300
    2/2 [==============================] - 24s 9s/step - loss: 4.4582 - accuracy: 0.2535 - val_loss: 8.2811 - val_accuracy: 0.0000e+00
    Epoch 131/300
    2/2 [==============================] - 24s 9s/step - loss: 4.7245 - accuracy: 0.2185 - val_loss: 8.9748 - val_accuracy: 0.0000e+00
    Epoch 132/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5346 - accuracy: 0.2214 - val_loss: 9.1521 - val_accuracy: 0.0000e+00
    Epoch 133/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5405 - accuracy: 0.1939 - val_loss: 8.4381 - val_accuracy: 0.0000e+00
    Epoch 134/300
    2/2 [==============================] - 24s 9s/step - loss: 4.3932 - accuracy: 0.2006 - val_loss: 7.4714 - val_accuracy: 0.0000e+00
    Epoch 135/300
    2/2 [==============================] - 22s 7s/step - loss: 4.4180 - accuracy: 0.1126 - val_loss: 7.4796 - val_accuracy: 0.0000e+00
    Epoch 136/300
    2/2 [==============================] - 23s 8s/step - loss: 4.1710 - accuracy: 0.1797 - val_loss: 8.3607 - val_accuracy: 0.0000e+00
    Epoch 137/300
    2/2 [==============================] - 24s 10s/step - loss: 4.0758 - accuracy: 0.1164 - val_loss: 10.1756 - val_accuracy: 0.0000e+00
    Epoch 138/300
    2/2 [==============================] - 22s 7s/step - loss: 3.8811 - accuracy: 0.3169 - val_loss: 9.9204 - val_accuracy: 0.0000e+00
    Epoch 139/300
    2/2 [==============================] - 24s 9s/step - loss: 3.7884 - accuracy: 0.2535 - val_loss: 10.4789 - val_accuracy: 0.0000e+00
    Epoch 140/300
    2/2 [==============================] - 27s 12s/step - loss: 4.4120 - accuracy: 0.2431 - val_loss: 10.1886 - val_accuracy: 0.0000e+00
    Epoch 141/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3227 - accuracy: 0.2535 - val_loss: 9.9306 - val_accuracy: 0.0000e+00
    Epoch 142/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3840 - accuracy: 0.1656 - val_loss: 10.5676 - val_accuracy: 0.0000e+00
    Epoch 143/300
    2/2 [==============================] - 24s 8s/step - loss: 4.8944 - accuracy: 0.3169 - val_loss: 9.8655 - val_accuracy: 0.0000e+00
    Epoch 144/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8093 - accuracy: 0.1551 - val_loss: 8.5308 - val_accuracy: 0.0000e+00
    Epoch 145/300
    2/2 [==============================] - 27s 10s/step - loss: 4.5328 - accuracy: 0.2744 - val_loss: 10.6246 - val_accuracy: 0.0000e+00
    Epoch 146/300
    2/2 [==============================] - 24s 8s/step - loss: 4.4826 - accuracy: 0.2602 - val_loss: 10.5580 - val_accuracy: 0.0000e+00
    Epoch 147/300
    2/2 [==============================] - 24s 8s/step - loss: 4.2392 - accuracy: 0.2535 - val_loss: 10.4916 - val_accuracy: 0.0000e+00
    Epoch 148/300
    2/2 [==============================] - 25s 8s/step - loss: 4.3248 - accuracy: 0.3595 - val_loss: 12.6175 - val_accuracy: 0.0000e+00
    Epoch 149/300
    2/2 [==============================] - 25s 8s/step - loss: 4.6335 - accuracy: 0.3728 - val_loss: 9.3120 - val_accuracy: 0.0000e+00
    Epoch 150/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9682 - accuracy: 0.1760 - val_loss: 9.8073 - val_accuracy: 0.0000e+00
    Epoch 151/300
    2/2 [==============================] - 26s 8s/step - loss: 4.8363 - accuracy: 0.3378 - val_loss: 9.7028 - val_accuracy: 0.0000e+00
    Epoch 152/300
    2/2 [==============================] - 25s 8s/step - loss: 4.9004 - accuracy: 0.2431 - val_loss: 7.6196 - val_accuracy: 0.0000e+00
    Epoch 153/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5463 - accuracy: 0.2431 - val_loss: 10.2852 - val_accuracy: 0.0000e+00
    Epoch 154/300
    2/2 [==============================] - 26s 10s/step - loss: 4.9392 - accuracy: 0.2819 - val_loss: 11.0909 - val_accuracy: 0.0000e+00
    Epoch 155/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9060 - accuracy: 0.2677 - val_loss: 11.7587 - val_accuracy: 0.0000e+00
    Epoch 156/300
    2/2 [==============================] - 25s 10s/step - loss: 5.2926 - accuracy: 0.2006 - val_loss: 10.7808 - val_accuracy: 0.0000e+00
    Epoch 157/300
    2/2 [==============================] - 25s 10s/step - loss: 5.1736 - accuracy: 0.1760 - val_loss: 10.3814 - val_accuracy: 0.0000e+00
    Epoch 158/300
    2/2 [==============================] - 24s 8s/step - loss: 4.8240 - accuracy: 0.2640 - val_loss: 7.9227 - val_accuracy: 0.0000e+00
    Epoch 159/300
    2/2 [==============================] - 27s 10s/step - loss: 4.5345 - accuracy: 0.2923 - val_loss: 9.7922 - val_accuracy: 0.0000e+00
    Epoch 160/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6944 - accuracy: 0.2677 - val_loss: 7.2195 - val_accuracy: 0.0000e+00
    Epoch 161/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2417 - accuracy: 0.3907 - val_loss: 8.7825 - val_accuracy: 0.0000e+00
    Epoch 162/300
    2/2 [==============================] - 23s 8s/step - loss: 4.5094 - accuracy: 0.2886 - val_loss: 8.7586 - val_accuracy: 0.0000e+00
    Epoch 163/300
    2/2 [==============================] - 25s 10s/step - loss: 4.4563 - accuracy: 0.2744 - val_loss: 6.2925 - val_accuracy: 0.0000e+00
    Epoch 164/300
    2/2 [==============================] - 23s 8s/step - loss: 4.8644 - accuracy: 0.1902 - val_loss: 6.5580 - val_accuracy: 0.0000e+00
    Epoch 165/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5762 - accuracy: 0.3027 - val_loss: 6.0723 - val_accuracy: 0.0000e+00
    Epoch 166/300
    2/2 [==============================] - 23s 8s/step - loss: 4.1191 - accuracy: 0.3766 - val_loss: 6.8681 - val_accuracy: 0.0000e+00
    Epoch 167/300
    2/2 [==============================] - 27s 10s/step - loss: 4.6489 - accuracy: 0.2043 - val_loss: 8.1273 - val_accuracy: 0.0000e+00
    Epoch 168/300
    2/2 [==============================] - 23s 8s/step - loss: 4.7606 - accuracy: 0.3273 - val_loss: 6.8033 - val_accuracy: 0.0000e+00
    Epoch 169/300
    2/2 [==============================] - 26s 10s/step - loss: 4.7706 - accuracy: 0.3444 - val_loss: 6.7447 - val_accuracy: 0.0000e+00
    Epoch 170/300
    2/2 [==============================] - 23s 8s/step - loss: 4.9086 - accuracy: 0.2498 - val_loss: 7.5628 - val_accuracy: 0.0000e+00
    Epoch 171/300
    2/2 [==============================] - 27s 10s/step - loss: 4.7701 - accuracy: 0.3378 - val_loss: 7.3020 - val_accuracy: 0.0000e+00
    Epoch 172/300
    2/2 [==============================] - 23s 8s/step - loss: 4.7285 - accuracy: 0.2431 - val_loss: 7.3137 - val_accuracy: 0.0000e+00
    Epoch 173/300
    2/2 [==============================] - 27s 10s/step - loss: 4.7466 - accuracy: 0.2043 - val_loss: 7.1749 - val_accuracy: 0.0000e+00
    Epoch 174/300
    2/2 [==============================] - 25s 10s/step - loss: 4.8148 - accuracy: 0.3415 - val_loss: 6.8643 - val_accuracy: 0.0000e+00
    Epoch 175/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6412 - accuracy: 0.3132 - val_loss: 7.3293 - val_accuracy: 0.0000e+00
    Epoch 176/300
    2/2 [==============================] - 27s 10s/step - loss: 4.6457 - accuracy: 0.2886 - val_loss: 6.3776 - val_accuracy: 0.0000e+00
    Epoch 177/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5605 - accuracy: 0.3624 - val_loss: 6.4687 - val_accuracy: 0.0000e+00
    Epoch 178/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5122 - accuracy: 0.2640 - val_loss: 17.3706 - val_accuracy: 0.0000e+00
    Epoch 179/300
    2/2 [==============================] - 25s 10s/step - loss: 6.5917 - accuracy: 0.3027 - val_loss: 11.3023 - val_accuracy: 0.3333
    Epoch 180/300
    2/2 [==============================] - 25s 10s/step - loss: 6.7579 - accuracy: 0.2394 - val_loss: 10.1574 - val_accuracy: 0.3333
    Epoch 181/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2553 - accuracy: 0.3586 - val_loss: 9.6768 - val_accuracy: 0.3333
    Epoch 182/300
    2/2 [==============================] - 23s 8s/step - loss: 6.2841 - accuracy: 0.1902 - val_loss: 8.5278 - val_accuracy: 0.0000e+00
    Epoch 183/300
    2/2 [==============================] - 25s 10s/step - loss: 5.8197 - accuracy: 0.3236 - val_loss: 8.3436 - val_accuracy: 0.0000e+00
    Epoch 184/300
    2/2 [==============================] - 24s 8s/step - loss: 5.6783 - accuracy: 0.3378 - val_loss: 8.0380 - val_accuracy: 0.0000e+00
    Epoch 185/300
    2/2 [==============================] - 25s 8s/step - loss: 5.5018 - accuracy: 0.2781 - val_loss: 7.2752 - val_accuracy: 0.0000e+00
    Epoch 186/300
    2/2 [==============================] - 25s 10s/step - loss: 5.3602 - accuracy: 0.2498 - val_loss: 7.2590 - val_accuracy: 0.0000e+00
    Epoch 187/300
    2/2 [==============================] - 25s 10s/step - loss: 5.0393 - accuracy: 0.2990 - val_loss: 7.0686 - val_accuracy: 0.0000e+00
    Epoch 188/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6277 - accuracy: 0.2886 - val_loss: 6.6340 - val_accuracy: 0.0000e+00
    Epoch 189/300
    2/2 [==============================] - 25s 8s/step - loss: 4.4930 - accuracy: 0.3273 - val_loss: 9.9408 - val_accuracy: 0.0000e+00
    Epoch 190/300
    2/2 [==============================] - 25s 10s/step - loss: 5.2689 - accuracy: 0.2394 - val_loss: 7.4237 - val_accuracy: 0.0000e+00
    Epoch 191/300
    2/2 [==============================] - 23s 8s/step - loss: 5.2154 - accuracy: 0.3349 - val_loss: 6.9715 - val_accuracy: 0.0000e+00
    Epoch 192/300
    2/2 [==============================] - 25s 10s/step - loss: 4.5935 - accuracy: 0.4049 - val_loss: 7.4654 - val_accuracy: 0.0000e+00
    Epoch 193/300
    2/2 [==============================] - 23s 8s/step - loss: 4.5268 - accuracy: 0.3311 - val_loss: 7.2180 - val_accuracy: 0.0000e+00
    Epoch 194/300
    2/2 [==============================] - 27s 10s/step - loss: 4.6494 - accuracy: 0.2043 - val_loss: 7.0412 - val_accuracy: 0.0000e+00
    Epoch 195/300
    2/2 [==============================] - 26s 10s/step - loss: 4.1360 - accuracy: 0.4399 - val_loss: 6.8671 - val_accuracy: 0.0000e+00
    Epoch 196/300
    2/2 [==============================] - 23s 8s/step - loss: 4.4966 - accuracy: 0.3870 - val_loss: 7.0933 - val_accuracy: 0.0000e+00
    Epoch 197/300
    2/2 [==============================] - 26s 8s/step - loss: 4.8300 - accuracy: 0.2819 - val_loss: 6.4346 - val_accuracy: 0.0000e+00
    Epoch 198/300
    2/2 [==============================] - 26s 10s/step - loss: 4.5851 - accuracy: 0.3624 - val_loss: 397.7638 - val_accuracy: 0.0000e+00
    Epoch 199/300
    2/2 [==============================] - 23s 8s/step - loss: 10.2273 - accuracy: 0.0776 - val_loss: 269.1886 - val_accuracy: 0.0000e+00
    Epoch 200/300
    2/2 [==============================] - 27s 10s/step - loss: 10.2422 - accuracy: 0.1022 - val_loss: 188.1016 - val_accuracy: 0.0000e+00
    Epoch 201/300
    2/2 [==============================] - 25s 10s/step - loss: 9.3662 - accuracy: 0.1618 - val_loss: 82.7548 - val_accuracy: 0.0000e+00
    Epoch 202/300
    2/2 [==============================] - 24s 8s/step - loss: 9.0795 - accuracy: 0.1476 - val_loss: 61.3656 - val_accuracy: 0.0000e+00
    Epoch 203/300
    2/2 [==============================] - 25s 10s/step - loss: 8.3122 - accuracy: 0.1797 - val_loss: 48.5817 - val_accuracy: 0.0000e+00
    Epoch 204/300
    2/2 [==============================] - 25s 10s/step - loss: 8.2439 - accuracy: 0.2006 - val_loss: 33.1528 - val_accuracy: 0.0000e+00
    Epoch 205/300
    2/2 [==============================] - 25s 10s/step - loss: 7.9968 - accuracy: 0.1656 - val_loss: 44.8878 - val_accuracy: 0.0000e+00
    Epoch 206/300
    2/2 [==============================] - 24s 8s/step - loss: 7.6002 - accuracy: 0.2289 - val_loss: 22.1734 - val_accuracy: 0.0000e+00
    Epoch 207/300
    2/2 [==============================] - 25s 10s/step - loss: 7.2712 - accuracy: 0.2886 - val_loss: 9.7701 - val_accuracy: 0.0000e+00
    Epoch 208/300
    2/2 [==============================] - 23s 8s/step - loss: 7.5634 - accuracy: 0.2148 - val_loss: 18.1320 - val_accuracy: 0.0000e+00
    Epoch 209/300
    2/2 [==============================] - 27s 10s/step - loss: 7.5962 - accuracy: 0.1902 - val_loss: 23.4073 - val_accuracy: 0.0000e+00
    Epoch 210/300
    2/2 [==============================] - 23s 8s/step - loss: 7.2979 - accuracy: 0.1864 - val_loss: 13.5705 - val_accuracy: 0.0000e+00
    Epoch 211/300
    2/2 [==============================] - 25s 8s/step - loss: 7.1635 - accuracy: 0.1618 - val_loss: 13.4136 - val_accuracy: 0.0000e+00
    Epoch 212/300
    2/2 [==============================] - 25s 10s/step - loss: 6.6739 - accuracy: 0.2252 - val_loss: 13.4742 - val_accuracy: 0.0000e+00
    Epoch 213/300
    2/2 [==============================] - 24s 8s/step - loss: 6.8085 - accuracy: 0.2394 - val_loss: 9.3992 - val_accuracy: 0.0000e+00
    Epoch 214/300
    2/2 [==============================] - 24s 8s/step - loss: 6.5417 - accuracy: 0.3027 - val_loss: 11.4507 - val_accuracy: 0.0000e+00
    Epoch 215/300
    2/2 [==============================] - 28s 10s/step - loss: 6.6947 - accuracy: 0.2252 - val_loss: 10.4169 - val_accuracy: 0.0000e+00
    Epoch 216/300
    2/2 [==============================] - 25s 10s/step - loss: 6.4483 - accuracy: 0.2886 - val_loss: 10.9209 - val_accuracy: 0.0000e+00
    Epoch 217/300
    2/2 [==============================] - 25s 10s/step - loss: 6.3183 - accuracy: 0.2640 - val_loss: 11.9081 - val_accuracy: 0.0000e+00
    Epoch 218/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2015 - accuracy: 0.2252 - val_loss: 9.1835 - val_accuracy: 0.0000e+00
    Epoch 219/300
    2/2 [==============================] - 23s 8s/step - loss: 6.4283 - accuracy: 0.1939 - val_loss: 8.2928 - val_accuracy: 0.0000e+00
    Epoch 220/300
    2/2 [==============================] - 26s 10s/step - loss: 5.9991 - accuracy: 0.2781 - val_loss: 13.7696 - val_accuracy: 0.0000e+00
    Epoch 221/300
    2/2 [==============================] - 23s 8s/step - loss: 6.1027 - accuracy: 0.2289 - val_loss: 8.8681 - val_accuracy: 0.0000e+00
    Epoch 222/300
    2/2 [==============================] - 25s 8s/step - loss: 6.1600 - accuracy: 0.3169 - val_loss: 8.2148 - val_accuracy: 0.0000e+00
    Epoch 223/300
    2/2 [==============================] - 25s 10s/step - loss: 5.9852 - accuracy: 0.2148 - val_loss: 8.2718 - val_accuracy: 0.0000e+00
    Epoch 224/300
    2/2 [==============================] - 24s 8s/step - loss: 5.9231 - accuracy: 0.2640 - val_loss: 9.3759 - val_accuracy: 0.0000e+00
    Epoch 225/300
    2/2 [==============================] - 25s 10s/step - loss: 5.8739 - accuracy: 0.2640 - val_loss: 8.3828 - val_accuracy: 0.0000e+00
    Epoch 226/300
    2/2 [==============================] - 25s 10s/step - loss: 5.9330 - accuracy: 0.2848 - val_loss: 8.4348 - val_accuracy: 0.0000e+00
    Epoch 227/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2834 - accuracy: 0.2185 - val_loss: 8.4785 - val_accuracy: 0.0000e+00
    Epoch 228/300
    2/2 [==============================] - 24s 8s/step - loss: 6.3079 - accuracy: 0.1514 - val_loss: 8.7399 - val_accuracy: 0.0000e+00
    Epoch 229/300
    2/2 [==============================] - 24s 8s/step - loss: 5.8445 - accuracy: 0.2781 - val_loss: 8.2607 - val_accuracy: 0.0000e+00
    Epoch 230/300
    2/2 [==============================] - 25s 10s/step - loss: 5.6436 - accuracy: 0.1447 - val_loss: 9.3876 - val_accuracy: 0.0000e+00
    Epoch 231/300
    2/2 [==============================] - 25s 10s/step - loss: 6.0462 - accuracy: 0.2498 - val_loss: 8.6130 - val_accuracy: 0.0000e+00
    Epoch 232/300
    2/2 [==============================] - 23s 8s/step - loss: 6.1130 - accuracy: 0.2535 - val_loss: 8.0332 - val_accuracy: 0.0000e+00
    Epoch 233/300
    2/2 [==============================] - 25s 8s/step - loss: 6.2583 - accuracy: 0.2535 - val_loss: 8.2855 - val_accuracy: 0.0000e+00
    Epoch 234/300
    2/2 [==============================] - 25s 10s/step - loss: 5.9242 - accuracy: 0.3236 - val_loss: 7.2154 - val_accuracy: 0.0000e+00
    Epoch 235/300
    2/2 [==============================] - 24s 8s/step - loss: 5.9856 - accuracy: 0.2535 - val_loss: 7.7533 - val_accuracy: 0.0000e+00
    Epoch 236/300
    2/2 [==============================] - 23s 8s/step - loss: 5.9612 - accuracy: 0.2781 - val_loss: 7.6686 - val_accuracy: 0.0000e+00
    Epoch 237/300
    2/2 [==============================] - 25s 8s/step - loss: 5.4448 - accuracy: 0.3945 - val_loss: 8.5258 - val_accuracy: 0.0000e+00
    Epoch 238/300
    2/2 [==============================] - 27s 10s/step - loss: 5.7412 - accuracy: 0.2535 - val_loss: 8.9677 - val_accuracy: 0.0000e+00
    Epoch 239/300
    2/2 [==============================] - 24s 8s/step - loss: 5.7822 - accuracy: 0.2781 - val_loss: 8.6608 - val_accuracy: 0.0000e+00
    Epoch 240/300
    2/2 [==============================] - 27s 9s/step - loss: 5.4907 - accuracy: 0.3207 - val_loss: 8.0375 - val_accuracy: 0.0000e+00
    Epoch 241/300
    2/2 [==============================] - 27s 10s/step - loss: 5.5086 - accuracy: 0.3907 - val_loss: 13.0289 - val_accuracy: 0.0000e+00
    Epoch 242/300
    2/2 [==============================] - 26s 10s/step - loss: 7.7460 - accuracy: 0.1022 - val_loss: 9.2460 - val_accuracy: 0.0000e+00
    Epoch 243/300
    2/2 [==============================] - 27s 10s/step - loss: 7.0742 - accuracy: 0.2535 - val_loss: 8.6890 - val_accuracy: 0.0000e+00
    Epoch 244/300
    2/2 [==============================] - 24s 8s/step - loss: 7.1093 - accuracy: 0.2043 - val_loss: 7.6926 - val_accuracy: 0.0000e+00
    Epoch 245/300
    2/2 [==============================] - 26s 11s/step - loss: 6.7064 - accuracy: 0.3624 - val_loss: 7.6723 - val_accuracy: 0.0000e+00
    Epoch 246/300
    2/2 [==============================] - 26s 8s/step - loss: 6.3214 - accuracy: 0.2781 - val_loss: 8.2369 - val_accuracy: 0.0000e+00
    Epoch 247/300
    2/2 [==============================] - 26s 10s/step - loss: 6.5867 - accuracy: 0.3273 - val_loss: 8.1612 - val_accuracy: 0.0000e+00
    Epoch 248/300
    2/2 [==============================] - 25s 8s/step - loss: 6.8063 - accuracy: 0.2043 - val_loss: 8.7408 - val_accuracy: 0.0000e+00
    Epoch 249/300
    2/2 [==============================] - 26s 8s/step - loss: 6.3957 - accuracy: 0.2289 - val_loss: 8.1762 - val_accuracy: 0.0000e+00
    Epoch 250/300
    2/2 [==============================] - 27s 10s/step - loss: 6.2894 - accuracy: 0.2640 - val_loss: 8.0054 - val_accuracy: 0.0000e+00
    Epoch 251/300
    2/2 [==============================] - 25s 8s/step - loss: 5.8926 - accuracy: 0.2043 - val_loss: 7.9241 - val_accuracy: 0.0000e+00
    Epoch 252/300
    2/2 [==============================] - 26s 8s/step - loss: 5.5411 - accuracy: 0.2394 - val_loss: 7.5014 - val_accuracy: 0.0000e+00
    Epoch 253/300
    2/2 [==============================] - 27s 8s/step - loss: 5.2084 - accuracy: 0.3907 - val_loss: 7.5424 - val_accuracy: 0.0000e+00
    Epoch 254/300
    2/2 [==============================] - 27s 10s/step - loss: 5.4054 - accuracy: 0.3065 - val_loss: 7.1468 - val_accuracy: 0.0000e+00
    Epoch 255/300
    2/2 [==============================] - 25s 10s/step - loss: 5.2952 - accuracy: 0.1551 - val_loss: 7.3014 - val_accuracy: 0.0000e+00
    Epoch 256/300
    2/2 [==============================] - 25s 10s/step - loss: 5.3273 - accuracy: 0.2886 - val_loss: 8.1271 - val_accuracy: 0.0000e+00
    Epoch 257/300
    2/2 [==============================] - 26s 10s/step - loss: 5.4778 - accuracy: 0.4012 - val_loss: 7.3426 - val_accuracy: 0.0000e+00
    Epoch 258/300
    2/2 [==============================] - 23s 8s/step - loss: 5.1909 - accuracy: 0.2677 - val_loss: 8.0391 - val_accuracy: 0.0000e+00
    Epoch 259/300
    2/2 [==============================] - 23s 8s/step - loss: 5.3811 - accuracy: 0.2110 - val_loss: 7.3563 - val_accuracy: 0.0000e+00
    Epoch 260/300
    2/2 [==============================] - 24s 8s/step - loss: 5.2312 - accuracy: 0.3586 - val_loss: 7.5882 - val_accuracy: 0.0000e+00
    Epoch 261/300
    2/2 [==============================] - 28s 10s/step - loss: 5.0802 - accuracy: 0.3624 - val_loss: 7.9820 - val_accuracy: 0.3333
    Epoch 262/300
    2/2 [==============================] - 25s 8s/step - loss: 5.3543 - accuracy: 0.3557 - val_loss: 7.3139 - val_accuracy: 0.0000e+00
    Epoch 263/300
    2/2 [==============================] - 25s 8s/step - loss: 5.3129 - accuracy: 0.2886 - val_loss: 7.3830 - val_accuracy: 0.0000e+00
    Epoch 264/300
    2/2 [==============================] - 24s 8s/step - loss: 5.1601 - accuracy: 0.3870 - val_loss: 7.3108 - val_accuracy: 0.0000e+00
    Epoch 265/300
    2/2 [==============================] - 26s 8s/step - loss: 5.0043 - accuracy: 0.3661 - val_loss: 7.1007 - val_accuracy: 0.3333
    Epoch 266/300
    2/2 [==============================] - 25s 8s/step - loss: 5.1690 - accuracy: 0.4087 - val_loss: 6.4821 - val_accuracy: 0.0000e+00
    Epoch 267/300
    2/2 [==============================] - 25s 10s/step - loss: 4.7239 - accuracy: 0.4333 - val_loss: 6.4445 - val_accuracy: 0.0000e+00
    Epoch 268/300
    2/2 [==============================] - 26s 10s/step - loss: 4.7337 - accuracy: 0.3482 - val_loss: 6.8918 - val_accuracy: 0.0000e+00
    Epoch 269/300
    2/2 [==============================] - 25s 10s/step - loss: 4.8406 - accuracy: 0.3132 - val_loss: 7.8530 - val_accuracy: 0.0000e+00
    Epoch 270/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9612 - accuracy: 0.4750 - val_loss: 11.4407 - val_accuracy: 0.0000e+00
    Epoch 271/300
    2/2 [==============================] - 26s 10s/step - loss: 5.0852 - accuracy: 0.3311 - val_loss: 10.2482 - val_accuracy: 0.0000e+00
    Epoch 272/300
    2/2 [==============================] - 25s 10s/step - loss: 5.5694 - accuracy: 0.2573 - val_loss: 11.8028 - val_accuracy: 0.0000e+00
    Epoch 273/300
    2/2 [==============================] - 23s 8s/step - loss: 9.0656 - accuracy: 0.2781 - val_loss: 15.3100 - val_accuracy: 0.3333
    Epoch 274/300
    2/2 [==============================] - 28s 11s/step - loss: 9.9288 - accuracy: 0.2006 - val_loss: 16.5221 - val_accuracy: 0.0000e+00
    Epoch 275/300
    2/2 [==============================] - 24s 8s/step - loss: 9.9238 - accuracy: 0.1514 - val_loss: 14.7241 - val_accuracy: 0.0000e+00
    Epoch 276/300
    2/2 [==============================] - 23s 8s/step - loss: 9.8051 - accuracy: 0.3027 - val_loss: 12.9164 - val_accuracy: 0.0000e+00
    Epoch 277/300
    2/2 [==============================] - 28s 10s/step - loss: 9.5721 - accuracy: 0.2744 - val_loss: 13.5919 - val_accuracy: 0.0000e+00
    Epoch 278/300
    2/2 [==============================] - 24s 8s/step - loss: 9.5113 - accuracy: 0.1618 - val_loss: 14.5293 - val_accuracy: 0.0000e+00
    Epoch 279/300
    2/2 [==============================] - 25s 10s/step - loss: 9.4173 - accuracy: 0.2886 - val_loss: 17.1075 - val_accuracy: 0.0000e+00
    Epoch 280/300
    2/2 [==============================] - 25s 8s/step - loss: 9.1869 - accuracy: 0.2573 - val_loss: 12.1731 - val_accuracy: 0.0000e+00
    Epoch 281/300
    2/2 [==============================] - 26s 8s/step - loss: 8.8086 - accuracy: 0.3557 - val_loss: 13.3419 - val_accuracy: 0.0000e+00
    Epoch 282/300
    2/2 [==============================] - 24s 8s/step - loss: 8.8475 - accuracy: 0.4012 - val_loss: 11.2286 - val_accuracy: 0.0000e+00
    Epoch 283/300
    2/2 [==============================] - 27s 10s/step - loss: 8.2537 - accuracy: 0.3766 - val_loss: 9.9485 - val_accuracy: 0.0000e+00
    Epoch 284/300
    2/2 [==============================] - 24s 8s/step - loss: 8.0447 - accuracy: 0.4645 - val_loss: 11.0526 - val_accuracy: 0.0000e+00
    Epoch 285/300
    2/2 [==============================] - 24s 8s/step - loss: 8.0244 - accuracy: 0.3169 - val_loss: 11.0625 - val_accuracy: 0.0000e+00
    Epoch 286/300
    2/2 [==============================] - 27s 10s/step - loss: 8.3351 - accuracy: 0.2886 - val_loss: 11.0281 - val_accuracy: 0.0000e+00
    Epoch 287/300
    2/2 [==============================] - 25s 9s/step - loss: 7.8946 - accuracy: 0.3482 - val_loss: 9.9717 - val_accuracy: 0.0000e+00
    Epoch 288/300
    2/2 [==============================] - 25s 10s/step - loss: 7.6883 - accuracy: 0.3340 - val_loss: 10.0632 - val_accuracy: 0.0000e+00
    Epoch 289/300
    2/2 [==============================] - 23s 8s/step - loss: 7.4868 - accuracy: 0.2990 - val_loss: 11.2036 - val_accuracy: 0.0000e+00
    Epoch 290/300
    2/2 [==============================] - 25s 8s/step - loss: 9.1655 - accuracy: 0.2923 - val_loss: 10.1441 - val_accuracy: 0.0000e+00
    Epoch 291/300
    2/2 [==============================] - 27s 10s/step - loss: 8.6558 - accuracy: 0.3065 - val_loss: 12.4266 - val_accuracy: 0.0000e+00
    Epoch 292/300
    2/2 [==============================] - 23s 8s/step - loss: 8.5363 - accuracy: 0.2886 - val_loss: 11.0165 - val_accuracy: 0.0000e+00
    Epoch 293/300
    2/2 [==============================] - 26s 10s/step - loss: 7.9796 - accuracy: 0.3207 - val_loss: 9.6630 - val_accuracy: 0.0000e+00
    Epoch 294/300
    2/2 [==============================] - 25s 10s/step - loss: 7.2127 - accuracy: 0.4541 - val_loss: 9.2583 - val_accuracy: 0.0000e+00
    Epoch 295/300
    2/2 [==============================] - 25s 10s/step - loss: 7.2478 - accuracy: 0.3661 - val_loss: 9.2984 - val_accuracy: 0.0000e+00
    Epoch 296/300
    2/2 [==============================] - 26s 8s/step - loss: 7.0934 - accuracy: 0.2744 - val_loss: 9.3598 - val_accuracy: 0.0000e+00
    Epoch 297/300
    2/2 [==============================] - 25s 10s/step - loss: 6.6416 - accuracy: 0.3132 - val_loss: 8.6419 - val_accuracy: 0.0000e+00
    Epoch 298/300
    2/2 [==============================] - 28s 9s/step - loss: 6.5439 - accuracy: 0.1760 - val_loss: 10.3303 - val_accuracy: 0.0000e+00
    Epoch 299/300
    2/2 [==============================] - 25s 8s/step - loss: 7.6977 - accuracy: 0.3349 - val_loss: 10.7887 - val_accuracy: 0.0000e+00
    Epoch 300/300
    2/2 [==============================] - 26s 8s/step - loss: 7.6723 - accuracy: 0.3273 - val_loss: 10.5058 - val_accuracy: 0.0000e+00
    

### TRAINING RESULTS FOR MODEL2


```python
plot_model(train_model1,"LOG_MEL","STFT")
```


    
![png](output_77_0.png)
    


<a id="model3"></a>
## MODEL3 - STFT AND MFCC

- Here the train data is split into two equal parts and each part is sent to extract features.
- Train2 extracts the STFT features and send these features to "stft_train2.npy" file.
- Train1 extracts the MFCC features and sends it to a file 'mfcc_train1.npy'.


```python
extract_features(STFT,'train2',train2)
```

    STFTs FEATURES EXTRACTED......
    


```python
stft2 = np.load('stft_train2.npy',allow_pickle='TRUE').item()
mfcc1 = np.load('mfcc_train1.npy',allow_pickle='TRUE').item()

model3_train = {
    'spectogram':[],
    'label':[]
}


model3_train = log_mel2['spectogram']+stft1['spectogram']
model3_tlabel = log_mel2['label']+stft1['label']
```

#### Binarize Labels


```python
X = np.array(model3_train)
Y = model3_tlabel

Y = keras.utils.to_categorical(Y,num_classes = n_classes)
print(Y)
```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    

#### Reshaping 


```python
X = MinMaxScaler().fit_transform(X.reshape(-1,X.shape[-1])).reshape(shape)
X = X.reshape(X.shape[0],X.shape[2],X.shape[3],X.shape[1])
print(X.shape)
```

    (50, 256, 512, 3)
    

### TRAINING MODEL3


```python
model3 = build_sc_model()
model3.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
               ,loss='categorical_crossentropy',metrics=['accuracy'])
```


```python
Epochs=300
batch_size=32
filepath = "CNNModel3.hdf5"


train_model2 = model3.fit(X,Y,validation_split = 0.05,verbose=1, epochs = Epochs, batch_size = batch_size, shuffle = True)
```

    Epoch 1/300
    2/2 [==============================] - 24s 8s/step - loss: 13.2792 - accuracy: 0.0213 - val_loss: 535.6122 - val_accuracy: 0.0000e+00
    Epoch 2/300
    2/2 [==============================] - 25s 8s/step - loss: 13.0466 - accuracy: 0.0638 - val_loss: 539.4006 - val_accuracy: 0.0000e+00
    Epoch 3/300
    2/2 [==============================] - 25s 10s/step - loss: 12.9743 - accuracy: 0.1702 - val_loss: 385.2667 - val_accuracy: 0.0000e+00
    Epoch 4/300
    2/2 [==============================] - 23s 8s/step - loss: 12.6265 - accuracy: 0.0638 - val_loss: 347.7052 - val_accuracy: 0.0000e+00
    Epoch 5/300
    2/2 [==============================] - 27s 10s/step - loss: 12.5592 - accuracy: 0.0851 - val_loss: 352.4946 - val_accuracy: 0.0000e+00
    Epoch 6/300
    2/2 [==============================] - 23s 8s/step - loss: 12.2921 - accuracy: 0.0426 - val_loss: 271.0294 - val_accuracy: 0.0000e+00
    Epoch 7/300
    2/2 [==============================] - 24s 8s/step - loss: 11.8723 - accuracy: 0.0000e+00 - val_loss: 191.1571 - val_accuracy: 0.0000e+00
    Epoch 8/300
    2/2 [==============================] - 25s 8s/step - loss: 11.2554 - accuracy: 0.1489 - val_loss: 154.9715 - val_accuracy: 0.0000e+00
    Epoch 9/300
    2/2 [==============================] - 25s 8s/step - loss: 10.9391 - accuracy: 0.0426 - val_loss: 139.3941 - val_accuracy: 0.0000e+00
    Epoch 10/300
    2/2 [==============================] - 26s 10s/step - loss: 10.5262 - accuracy: 0.1915 - val_loss: 124.2015 - val_accuracy: 0.0000e+00
    Epoch 11/300
    2/2 [==============================] - 26s 10s/step - loss: 10.2083 - accuracy: 0.1064 - val_loss: 75.3529 - val_accuracy: 0.0000e+00
    Epoch 12/300
    2/2 [==============================] - 23s 8s/step - loss: 9.8133 - accuracy: 0.1277 - val_loss: 64.3173 - val_accuracy: 0.0000e+00
    Epoch 13/300
    2/2 [==============================] - 25s 10s/step - loss: 9.4224 - accuracy: 0.1702 - val_loss: 48.1351 - val_accuracy: 0.0000e+00
    Epoch 14/300
    2/2 [==============================] - 24s 10s/step - loss: 9.2746 - accuracy: 0.0851 - val_loss: 36.3058 - val_accuracy: 0.0000e+00
    Epoch 15/300
    2/2 [==============================] - 24s 8s/step - loss: 9.0149 - accuracy: 0.1064 - val_loss: 30.5821 - val_accuracy: 0.0000e+00
    Epoch 16/300
    2/2 [==============================] - 25s 8s/step - loss: 8.5843 - accuracy: 0.1064 - val_loss: 26.1785 - val_accuracy: 0.0000e+00
    Epoch 17/300
    2/2 [==============================] - 25s 8s/step - loss: 8.2279 - accuracy: 0.1064 - val_loss: 35.5532 - val_accuracy: 0.0000e+00
    Epoch 18/300
    2/2 [==============================] - 27s 10s/step - loss: 7.8409 - accuracy: 0.1702 - val_loss: 23.4427 - val_accuracy: 0.0000e+00
    Epoch 19/300
    2/2 [==============================] - 25s 10s/step - loss: 7.7031 - accuracy: 0.2128 - val_loss: 24.0244 - val_accuracy: 0.0000e+00
    Epoch 20/300
    2/2 [==============================] - 25s 10s/step - loss: 7.6512 - accuracy: 0.1277 - val_loss: 21.7085 - val_accuracy: 0.0000e+00
    Epoch 21/300
    2/2 [==============================] - 25s 9s/step - loss: 7.3781 - accuracy: 0.1702 - val_loss: 20.5974 - val_accuracy: 0.0000e+00
    Epoch 22/300
    2/2 [==============================] - 23s 8s/step - loss: 7.3032 - accuracy: 0.1277 - val_loss: 15.8455 - val_accuracy: 0.0000e+00
    Epoch 23/300
    2/2 [==============================] - 25s 10s/step - loss: 7.0951 - accuracy: 0.0851 - val_loss: 24.2417 - val_accuracy: 0.0000e+00
    Epoch 24/300
    2/2 [==============================] - 23s 8s/step - loss: 6.8453 - accuracy: 0.1277 - val_loss: 16.8117 - val_accuracy: 0.0000e+00
    Epoch 25/300
    2/2 [==============================] - 27s 10s/step - loss: 6.9112 - accuracy: 0.1702 - val_loss: 13.8845 - val_accuracy: 0.0000e+00
    Epoch 26/300
    2/2 [==============================] - 25s 10s/step - loss: 6.5036 - accuracy: 0.1489 - val_loss: 11.6106 - val_accuracy: 0.0000e+00
    Epoch 27/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2735 - accuracy: 0.2553 - val_loss: 9.9755 - val_accuracy: 0.0000e+00
    Epoch 28/300
    2/2 [==============================] - 26s 10s/step - loss: 6.3917 - accuracy: 0.1489 - val_loss: 10.0174 - val_accuracy: 0.0000e+00
    Epoch 29/300
    2/2 [==============================] - 22s 8s/step - loss: 6.0171 - accuracy: 0.2340 - val_loss: 9.9309 - val_accuracy: 0.0000e+00
    Epoch 30/300
    2/2 [==============================] - 25s 8s/step - loss: 6.2348 - accuracy: 0.0638 - val_loss: 7.4605 - val_accuracy: 0.0000e+00
    Epoch 31/300
    2/2 [==============================] - 25s 8s/step - loss: 5.7397 - accuracy: 0.1489 - val_loss: 7.7926 - val_accuracy: 0.0000e+00
    Epoch 32/300
    2/2 [==============================] - 22s 8s/step - loss: 5.7347 - accuracy: 0.2340 - val_loss: 8.2211 - val_accuracy: 0.0000e+00
    Epoch 33/300
    2/2 [==============================] - 24s 8s/step - loss: 5.8923 - accuracy: 0.2340 - val_loss: 11.9516 - val_accuracy: 0.0000e+00
    Epoch 34/300
    2/2 [==============================] - 25s 10s/step - loss: 5.7561 - accuracy: 0.2128 - val_loss: 8.4545 - val_accuracy: 0.0000e+00
    Epoch 35/300
    2/2 [==============================] - 25s 10s/step - loss: 5.2041 - accuracy: 0.2128 - val_loss: 7.0689 - val_accuracy: 0.0000e+00
    Epoch 36/300
    2/2 [==============================] - 23s 8s/step - loss: 5.3833 - accuracy: 0.1915 - val_loss: 5.7346 - val_accuracy: 0.0000e+00
    Epoch 37/300
    2/2 [==============================] - 25s 8s/step - loss: 5.1030 - accuracy: 0.3191 - val_loss: 6.8243 - val_accuracy: 0.0000e+00
    Epoch 38/300
    2/2 [==============================] - 26s 8s/step - loss: 4.9983 - accuracy: 0.3404 - val_loss: 5.6793 - val_accuracy: 0.0000e+00
    Epoch 39/300
    2/2 [==============================] - 25s 10s/step - loss: 5.0780 - accuracy: 0.1489 - val_loss: 6.1250 - val_accuracy: 0.0000e+00
    Epoch 40/300
    2/2 [==============================] - 24s 9s/step - loss: 5.1992 - accuracy: 0.2128 - val_loss: 6.8846 - val_accuracy: 0.0000e+00
    Epoch 41/300
    2/2 [==============================] - 24s 10s/step - loss: 5.1415 - accuracy: 0.2128 - val_loss: 6.2741 - val_accuracy: 0.0000e+00
    Epoch 42/300
    2/2 [==============================] - 24s 10s/step - loss: 4.8542 - accuracy: 0.1702 - val_loss: 5.8628 - val_accuracy: 0.0000e+00
    Epoch 43/300
    2/2 [==============================] - 25s 9s/step - loss: 4.8199 - accuracy: 0.2128 - val_loss: 6.5944 - val_accuracy: 0.0000e+00
    Epoch 44/300
    2/2 [==============================] - 24s 9s/step - loss: 4.8215 - accuracy: 0.2340 - val_loss: 7.0089 - val_accuracy: 0.0000e+00
    Epoch 45/300
    2/2 [==============================] - 24s 10s/step - loss: 4.4635 - accuracy: 0.2979 - val_loss: 5.6018 - val_accuracy: 0.0000e+00
    Epoch 46/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6093 - accuracy: 0.2128 - val_loss: 5.1064 - val_accuracy: 0.0000e+00
    Epoch 47/300
    2/2 [==============================] - 27s 10s/step - loss: 4.4291 - accuracy: 0.2340 - val_loss: 5.9905 - val_accuracy: 0.0000e+00
    Epoch 48/300
    2/2 [==============================] - 24s 10s/step - loss: 4.3965 - accuracy: 0.2553 - val_loss: 6.7326 - val_accuracy: 0.0000e+00
    Epoch 49/300
    2/2 [==============================] - 24s 9s/step - loss: 4.4372 - accuracy: 0.2128 - val_loss: 8.2075 - val_accuracy: 0.0000e+00
    Epoch 50/300
    2/2 [==============================] - 26s 8s/step - loss: 4.2358 - accuracy: 0.2128 - val_loss: 4.7152 - val_accuracy: 0.0000e+00
    Epoch 51/300
    2/2 [==============================] - 27s 10s/step - loss: 4.1984 - accuracy: 0.1702 - val_loss: 4.7173 - val_accuracy: 0.0000e+00
    Epoch 52/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0575 - accuracy: 0.2766 - val_loss: 6.1491 - val_accuracy: 0.0000e+00
    Epoch 53/300
    2/2 [==============================] - 25s 10s/step - loss: 4.4366 - accuracy: 0.2340 - val_loss: 5.3622 - val_accuracy: 0.0000e+00
    Epoch 54/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3891 - accuracy: 0.2340 - val_loss: 5.0428 - val_accuracy: 0.0000e+00
    Epoch 55/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2110 - accuracy: 0.2128 - val_loss: 5.1600 - val_accuracy: 0.0000e+00
    Epoch 56/300
    2/2 [==============================] - 22s 7s/step - loss: 4.0224 - accuracy: 0.2979 - val_loss: 7.8103 - val_accuracy: 0.0000e+00
    Epoch 57/300
    2/2 [==============================] - 25s 8s/step - loss: 4.0006 - accuracy: 0.1915 - val_loss: 5.1681 - val_accuracy: 0.3333
    Epoch 58/300
    2/2 [==============================] - 25s 10s/step - loss: 4.3883 - accuracy: 0.2766 - val_loss: 6.5267 - val_accuracy: 0.0000e+00
    Epoch 59/300
    2/2 [==============================] - 25s 10s/step - loss: 4.2263 - accuracy: 0.3191 - val_loss: 5.1972 - val_accuracy: 0.0000e+00
    Epoch 60/300
    2/2 [==============================] - 24s 9s/step - loss: 4.2616 - accuracy: 0.1277 - val_loss: 5.4074 - val_accuracy: 0.0000e+00
    Epoch 61/300
    2/2 [==============================] - 23s 8s/step - loss: 4.4322 - accuracy: 0.2553 - val_loss: 5.8040 - val_accuracy: 0.0000e+00
    Epoch 62/300
    2/2 [==============================] - 27s 10s/step - loss: 4.0921 - accuracy: 0.2766 - val_loss: 5.1666 - val_accuracy: 0.0000e+00
    Epoch 63/300
    2/2 [==============================] - 24s 10s/step - loss: 3.9869 - accuracy: 0.2553 - val_loss: 5.1095 - val_accuracy: 0.0000e+00
    Epoch 64/300
    2/2 [==============================] - 23s 8s/step - loss: 3.9901 - accuracy: 0.2553 - val_loss: 5.7876 - val_accuracy: 0.0000e+00
    Epoch 65/300
    2/2 [==============================] - 24s 8s/step - loss: 3.9910 - accuracy: 0.3404 - val_loss: 6.7203 - val_accuracy: 0.0000e+00
    Epoch 66/300
    2/2 [==============================] - 26s 10s/step - loss: 4.0936 - accuracy: 0.2340 - val_loss: 5.3624 - val_accuracy: 0.0000e+00
    Epoch 67/300
    2/2 [==============================] - 25s 10s/step - loss: 4.0987 - accuracy: 0.2766 - val_loss: 10.9031 - val_accuracy: 0.3333
    Epoch 68/300
    2/2 [==============================] - 23s 7s/step - loss: 4.3452 - accuracy: 0.2340 - val_loss: 15.2311 - val_accuracy: 0.3333
    Epoch 69/300
    2/2 [==============================] - 26s 8s/step - loss: 4.1105 - accuracy: 0.3191 - val_loss: 12.1651 - val_accuracy: 0.3333
    Epoch 70/300
    2/2 [==============================] - 27s 10s/step - loss: 4.0089 - accuracy: 0.2128 - val_loss: 10.0403 - val_accuracy: 0.3333
    Epoch 71/300
    2/2 [==============================] - 23s 8s/step - loss: 3.8320 - accuracy: 0.3404 - val_loss: 8.5471 - val_accuracy: 0.0000e+00
    Epoch 72/300
    2/2 [==============================] - 25s 8s/step - loss: 3.9329 - accuracy: 0.2128 - val_loss: 7.7347 - val_accuracy: 0.0000e+00
    Epoch 73/300
    2/2 [==============================] - 25s 7s/step - loss: 4.2378 - accuracy: 0.2766 - val_loss: 10.5880 - val_accuracy: 0.0000e+00
    Epoch 74/300
    2/2 [==============================] - 25s 8s/step - loss: 3.9476 - accuracy: 0.2766 - val_loss: 10.7557 - val_accuracy: 0.0000e+00
    Epoch 75/300
    2/2 [==============================] - 25s 8s/step - loss: 4.6666 - accuracy: 0.2128 - val_loss: 10.1447 - val_accuracy: 0.3333
    Epoch 76/300
    2/2 [==============================] - 25s 10s/step - loss: 4.4705 - accuracy: 0.2553 - val_loss: 9.2926 - val_accuracy: 0.0000e+00
    Epoch 77/300
    2/2 [==============================] - 23s 8s/step - loss: 4.2938 - accuracy: 0.3830 - val_loss: 11.7422 - val_accuracy: 0.3333
    Epoch 78/300
    2/2 [==============================] - 25s 8s/step - loss: 4.3853 - accuracy: 0.3617 - val_loss: 9.4256 - val_accuracy: 0.0000e+00
    Epoch 79/300
    2/2 [==============================] - 25s 8s/step - loss: 4.4541 - accuracy: 0.2766 - val_loss: 7.8171 - val_accuracy: 0.3333
    Epoch 80/300
    2/2 [==============================] - 23s 8s/step - loss: 4.0650 - accuracy: 0.3404 - val_loss: 7.2788 - val_accuracy: 0.3333
    Epoch 81/300
    2/2 [==============================] - 28s 10s/step - loss: 4.1907 - accuracy: 0.3191 - val_loss: 7.2225 - val_accuracy: 0.3333
    Epoch 82/300
    2/2 [==============================] - 27s 10s/step - loss: 4.0508 - accuracy: 0.3191 - val_loss: 6.8196 - val_accuracy: 0.3333
    Epoch 83/300
    2/2 [==============================] - 24s 8s/step - loss: 4.0679 - accuracy: 0.2553 - val_loss: 6.2202 - val_accuracy: 0.0000e+00
    Epoch 84/300
    2/2 [==============================] - 25s 8s/step - loss: 3.8485 - accuracy: 0.3830 - val_loss: 6.1222 - val_accuracy: 0.3333
    Epoch 85/300
    2/2 [==============================] - 25s 8s/step - loss: 4.1304 - accuracy: 0.2128 - val_loss: 6.5017 - val_accuracy: 0.0000e+00
    Epoch 86/300
    2/2 [==============================] - 29s 11s/step - loss: 3.7425 - accuracy: 0.4468 - val_loss: 5.7077 - val_accuracy: 0.3333
    Epoch 87/300
    2/2 [==============================] - 25s 8s/step - loss: 3.7714 - accuracy: 0.2766 - val_loss: 5.3976 - val_accuracy: 0.0000e+00
    Epoch 88/300
    2/2 [==============================] - 25s 8s/step - loss: 3.5863 - accuracy: 0.2553 - val_loss: 6.0301 - val_accuracy: 0.3333
    Epoch 89/300
    2/2 [==============================] - 26s 10s/step - loss: 3.7605 - accuracy: 0.2766 - val_loss: 6.7496 - val_accuracy: 0.3333
    Epoch 90/300
    2/2 [==============================] - 24s 8s/step - loss: 4.0133 - accuracy: 0.2340 - val_loss: 8.0261 - val_accuracy: 0.3333
    Epoch 91/300
    2/2 [==============================] - 25s 8s/step - loss: 3.8647 - accuracy: 0.2766 - val_loss: 8.3020 - val_accuracy: 0.0000e+00
    Epoch 92/300
    2/2 [==============================] - 28s 10s/step - loss: 4.0249 - accuracy: 0.2979 - val_loss: 6.2138 - val_accuracy: 0.0000e+00
    Epoch 93/300
    2/2 [==============================] - 26s 11s/step - loss: 4.6960 - accuracy: 0.2128 - val_loss: 7.6280 - val_accuracy: 0.0000e+00
    Epoch 94/300
    2/2 [==============================] - 27s 11s/step - loss: 5.0041 - accuracy: 0.1702 - val_loss: 28.4607 - val_accuracy: 0.0000e+00
    Epoch 95/300
    2/2 [==============================] - 23s 8s/step - loss: 4.6157 - accuracy: 0.3830 - val_loss: 23.5604 - val_accuracy: 0.3333
    Epoch 96/300
    2/2 [==============================] - 26s 8s/step - loss: 4.3949 - accuracy: 0.2979 - val_loss: 21.6064 - val_accuracy: 0.3333
    Epoch 97/300
    2/2 [==============================] - 27s 9s/step - loss: 4.3239 - accuracy: 0.2553 - val_loss: 24.4067 - val_accuracy: 0.0000e+00
    Epoch 98/300
    2/2 [==============================] - 26s 10s/step - loss: 4.3792 - accuracy: 0.2340 - val_loss: 14.1245 - val_accuracy: 0.0000e+00
    Epoch 99/300
    2/2 [==============================] - 25s 9s/step - loss: 4.8112 - accuracy: 0.2766 - val_loss: 14.7046 - val_accuracy: 0.0000e+00
    Epoch 100/300
    2/2 [==============================] - 27s 9s/step - loss: 4.8130 - accuracy: 0.1702 - val_loss: 12.3039 - val_accuracy: 0.3333
    Epoch 101/300
    2/2 [==============================] - 27s 8s/step - loss: 4.9454 - accuracy: 0.1915 - val_loss: 9.8337 - val_accuracy: 0.3333
    Epoch 102/300
    2/2 [==============================] - 29s 10s/step - loss: 4.4518 - accuracy: 0.3191 - val_loss: 16.9407 - val_accuracy: 0.0000e+00
    Epoch 103/300
    2/2 [==============================] - 24s 8s/step - loss: 4.5928 - accuracy: 0.2766 - val_loss: 18.4107 - val_accuracy: 0.3333
    Epoch 104/300
    2/2 [==============================] - 27s 8s/step - loss: 4.3366 - accuracy: 0.2340 - val_loss: 12.5886 - val_accuracy: 0.0000e+00
    Epoch 105/300
    2/2 [==============================] - 26s 8s/step - loss: 4.1245 - accuracy: 0.3404 - val_loss: 10.1169 - val_accuracy: 0.3333
    Epoch 106/300
    2/2 [==============================] - 26s 8s/step - loss: 3.9869 - accuracy: 0.4255 - val_loss: 13.5350 - val_accuracy: 0.3333
    Epoch 107/300
    2/2 [==============================] - 26s 8s/step - loss: 4.2199 - accuracy: 0.4255 - val_loss: 10.3934 - val_accuracy: 0.3333
    Epoch 108/300
    2/2 [==============================] - 26s 8s/step - loss: 3.9413 - accuracy: 0.4043 - val_loss: 10.1584 - val_accuracy: 0.3333
    Epoch 109/300
    2/2 [==============================] - 24s 9s/step - loss: 4.1012 - accuracy: 0.1915 - val_loss: 7.8577 - val_accuracy: 0.3333
    Epoch 110/300
    2/2 [==============================] - 26s 8s/step - loss: 4.3630 - accuracy: 0.2766 - val_loss: 8.3269 - val_accuracy: 0.3333
    Epoch 111/300
    2/2 [==============================] - 25s 8s/step - loss: 4.2398 - accuracy: 0.2128 - val_loss: 12.2229 - val_accuracy: 0.3333
    Epoch 112/300
    2/2 [==============================] - 27s 9s/step - loss: 4.2672 - accuracy: 0.3617 - val_loss: 6.8717 - val_accuracy: 0.3333
    Epoch 113/300
    2/2 [==============================] - 26s 8s/step - loss: 3.9268 - accuracy: 0.3830 - val_loss: 7.2828 - val_accuracy: 0.3333
    Epoch 114/300
    2/2 [==============================] - 27s 8s/step - loss: 4.1052 - accuracy: 0.2979 - val_loss: 5.2941 - val_accuracy: 0.3333
    Epoch 115/300
    2/2 [==============================] - 26s 10s/step - loss: 3.7417 - accuracy: 0.2553 - val_loss: 6.2702 - val_accuracy: 0.3333
    Epoch 116/300
    2/2 [==============================] - 25s 10s/step - loss: 4.9976 - accuracy: 0.2340 - val_loss: 7.4175 - val_accuracy: 0.3333
    Epoch 117/300
    2/2 [==============================] - 25s 10s/step - loss: 4.8642 - accuracy: 0.2979 - val_loss: 6.6854 - val_accuracy: 0.3333
    Epoch 118/300
    2/2 [==============================] - 24s 8s/step - loss: 5.1592 - accuracy: 0.2979 - val_loss: 7.7626 - val_accuracy: 0.6667
    Epoch 119/300
    2/2 [==============================] - 26s 8s/step - loss: 4.7742 - accuracy: 0.3191 - val_loss: 6.7947 - val_accuracy: 0.3333
    Epoch 120/300
    2/2 [==============================] - 27s 8s/step - loss: 5.1234 - accuracy: 0.1702 - val_loss: 6.7912 - val_accuracy: 0.3333
    Epoch 121/300
    2/2 [==============================] - 26s 8s/step - loss: 4.6634 - accuracy: 0.3191 - val_loss: 6.5115 - val_accuracy: 0.3333
    Epoch 122/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6982 - accuracy: 0.2128 - val_loss: 7.5942 - val_accuracy: 0.3333
    Epoch 123/300
    2/2 [==============================] - 25s 10s/step - loss: 4.6349 - accuracy: 0.2766 - val_loss: 7.2432 - val_accuracy: 0.0000e+00
    Epoch 124/300
    2/2 [==============================] - 23s 8s/step - loss: 4.7433 - accuracy: 0.3404 - val_loss: 7.3282 - val_accuracy: 0.0000e+00
    Epoch 125/300
    2/2 [==============================] - 26s 8s/step - loss: 4.2647 - accuracy: 0.2766 - val_loss: 6.6948 - val_accuracy: 0.0000e+00
    Epoch 126/300
    2/2 [==============================] - 25s 8s/step - loss: 4.1722 - accuracy: 0.2128 - val_loss: 5.9886 - val_accuracy: 0.0000e+00
    Epoch 127/300
    2/2 [==============================] - 25s 8s/step - loss: 3.9863 - accuracy: 0.3830 - val_loss: 7.4505 - val_accuracy: 0.0000e+00
    Epoch 128/300
    2/2 [==============================] - 27s 10s/step - loss: 3.8508 - accuracy: 0.4468 - val_loss: 6.6801 - val_accuracy: 0.3333
    Epoch 129/300
    2/2 [==============================] - 24s 8s/step - loss: 3.8042 - accuracy: 0.3404 - val_loss: 381.8166 - val_accuracy: 0.0000e+00
    Epoch 130/300
    2/2 [==============================] - 27s 10s/step - loss: 7.5955 - accuracy: 0.0426 - val_loss: 166.9691 - val_accuracy: 0.0000e+00
    Epoch 131/300
    2/2 [==============================] - 23s 8s/step - loss: 8.1973 - accuracy: 0.1277 - val_loss: 73.3603 - val_accuracy: 0.0000e+00
    Epoch 132/300
    2/2 [==============================] - 26s 8s/step - loss: 8.1952 - accuracy: 0.1064 - val_loss: 42.1380 - val_accuracy: 0.0000e+00
    Epoch 133/300
    2/2 [==============================] - 25s 10s/step - loss: 8.0654 - accuracy: 0.1702 - val_loss: 49.5203 - val_accuracy: 0.0000e+00
    Epoch 134/300
    2/2 [==============================] - 25s 10s/step - loss: 7.6522 - accuracy: 0.1489 - val_loss: 204.6612 - val_accuracy: 0.0000e+00
    Epoch 135/300
    2/2 [==============================] - 25s 10s/step - loss: 8.0931 - accuracy: 0.1064 - val_loss: 29.5406 - val_accuracy: 0.0000e+00
    Epoch 136/300
    2/2 [==============================] - 26s 10s/step - loss: 7.8266 - accuracy: 0.1702 - val_loss: 190.0964 - val_accuracy: 0.0000e+00
    Epoch 137/300
    2/2 [==============================] - 25s 10s/step - loss: 9.0805 - accuracy: 0.1489 - val_loss: 197.2767 - val_accuracy: 0.0000e+00
    Epoch 138/300
    2/2 [==============================] - 23s 8s/step - loss: 10.2826 - accuracy: 0.1702 - val_loss: 201.1810 - val_accuracy: 0.0000e+00
    Epoch 139/300
    2/2 [==============================] - 25s 8s/step - loss: 9.8825 - accuracy: 0.1915 - val_loss: 202.8156 - val_accuracy: 0.0000e+00
    Epoch 140/300
    2/2 [==============================] - 23s 8s/step - loss: 9.6900 - accuracy: 0.1702 - val_loss: 129.5552 - val_accuracy: 0.0000e+00
    Epoch 141/300
    2/2 [==============================] - 27s 10s/step - loss: 9.1473 - accuracy: 0.2766 - val_loss: 132.7404 - val_accuracy: 0.0000e+00
    Epoch 142/300
    2/2 [==============================] - 23s 8s/step - loss: 8.8788 - accuracy: 0.1915 - val_loss: 100.6116 - val_accuracy: 0.0000e+00
    Epoch 143/300
    2/2 [==============================] - 26s 8s/step - loss: 8.2492 - accuracy: 0.2340 - val_loss: 86.3572 - val_accuracy: 0.0000e+00
    Epoch 144/300
    2/2 [==============================] - 25s 10s/step - loss: 7.7552 - accuracy: 0.1702 - val_loss: 66.5715 - val_accuracy: 0.0000e+00
    Epoch 145/300
    2/2 [==============================] - 23s 8s/step - loss: 7.3345 - accuracy: 0.2979 - val_loss: 79.6432 - val_accuracy: 0.0000e+00
    Epoch 146/300
    2/2 [==============================] - 27s 10s/step - loss: 7.4520 - accuracy: 0.2553 - val_loss: 56.9065 - val_accuracy: 0.0000e+00
    Epoch 147/300
    2/2 [==============================] - 25s 10s/step - loss: 7.4244 - accuracy: 0.1702 - val_loss: 44.6259 - val_accuracy: 0.0000e+00
    Epoch 148/300
    2/2 [==============================] - 25s 10s/step - loss: 7.0135 - accuracy: 0.2553 - val_loss: 46.4183 - val_accuracy: 0.0000e+00
    Epoch 149/300
    2/2 [==============================] - 25s 10s/step - loss: 7.1690 - accuracy: 0.1915 - val_loss: 49.5826 - val_accuracy: 0.0000e+00
    Epoch 150/300
    2/2 [==============================] - 25s 9s/step - loss: 7.5488 - accuracy: 0.2553 - val_loss: 39.7196 - val_accuracy: 0.0000e+00
    Epoch 151/300
    2/2 [==============================] - 26s 10s/step - loss: 7.2608 - accuracy: 0.1277 - val_loss: 41.5252 - val_accuracy: 0.0000e+00
    Epoch 152/300
    2/2 [==============================] - 25s 10s/step - loss: 7.1450 - accuracy: 0.2553 - val_loss: 44.3113 - val_accuracy: 0.0000e+00
    Epoch 153/300
    2/2 [==============================] - 25s 10s/step - loss: 7.3941 - accuracy: 0.2340 - val_loss: 37.7347 - val_accuracy: 0.0000e+00
    Epoch 154/300
    2/2 [==============================] - 23s 8s/step - loss: 6.8472 - accuracy: 0.2766 - val_loss: 21.9400 - val_accuracy: 0.0000e+00
    Epoch 155/300
    2/2 [==============================] - 27s 10s/step - loss: 6.5791 - accuracy: 0.2766 - val_loss: 16.3748 - val_accuracy: 0.0000e+00
    Epoch 156/300
    2/2 [==============================] - 25s 10s/step - loss: 6.5801 - accuracy: 0.2340 - val_loss: 16.1017 - val_accuracy: 0.0000e+00
    Epoch 157/300
    2/2 [==============================] - 25s 10s/step - loss: 6.0926 - accuracy: 0.3404 - val_loss: 14.8094 - val_accuracy: 0.0000e+00
    Epoch 158/300
    2/2 [==============================] - 24s 8s/step - loss: 6.1684 - accuracy: 0.2979 - val_loss: 12.9806 - val_accuracy: 0.0000e+00
    Epoch 159/300
    2/2 [==============================] - 23s 8s/step - loss: 7.1879 - accuracy: 0.1489 - val_loss: 11.4672 - val_accuracy: 0.0000e+00
    Epoch 160/300
    2/2 [==============================] - 25s 8s/step - loss: 6.9874 - accuracy: 0.2128 - val_loss: 11.3674 - val_accuracy: 0.0000e+00
    Epoch 161/300
    2/2 [==============================] - 23s 8s/step - loss: 6.8770 - accuracy: 0.1702 - val_loss: 11.5230 - val_accuracy: 0.0000e+00
    Epoch 162/300
    2/2 [==============================] - 25s 8s/step - loss: 6.8171 - accuracy: 0.2979 - val_loss: 12.0493 - val_accuracy: 0.0000e+00
    Epoch 163/300
    2/2 [==============================] - 27s 10s/step - loss: 6.9865 - accuracy: 0.1489 - val_loss: 10.9130 - val_accuracy: 0.0000e+00
    Epoch 164/300
    2/2 [==============================] - 25s 10s/step - loss: 6.7159 - accuracy: 0.2766 - val_loss: 11.2696 - val_accuracy: 0.0000e+00
    Epoch 165/300
    2/2 [==============================] - 24s 8s/step - loss: 6.6479 - accuracy: 0.2128 - val_loss: 10.4003 - val_accuracy: 0.0000e+00
    Epoch 166/300
    2/2 [==============================] - 26s 9s/step - loss: 6.2853 - accuracy: 0.2979 - val_loss: 8.7568 - val_accuracy: 0.0000e+00
    Epoch 167/300
    2/2 [==============================] - 24s 9s/step - loss: 6.0872 - accuracy: 0.2979 - val_loss: 8.9627 - val_accuracy: 0.0000e+00
    Epoch 168/300
    2/2 [==============================] - 24s 10s/step - loss: 6.3110 - accuracy: 0.1915 - val_loss: 11.1843 - val_accuracy: 0.0000e+00
    Epoch 169/300
    2/2 [==============================] - 25s 10s/step - loss: 6.1979 - accuracy: 0.2766 - val_loss: 9.2972 - val_accuracy: 0.0000e+00
    Epoch 170/300
    2/2 [==============================] - 24s 9s/step - loss: 6.1693 - accuracy: 0.2766 - val_loss: 8.5713 - val_accuracy: 0.0000e+00
    Epoch 171/300
    2/2 [==============================] - 23s 8s/step - loss: 6.1474 - accuracy: 0.1277 - val_loss: 9.0164 - val_accuracy: 0.0000e+00
    Epoch 172/300
    2/2 [==============================] - 25s 8s/step - loss: 5.9493 - accuracy: 0.3191 - val_loss: 8.0816 - val_accuracy: 0.0000e+00
    Epoch 173/300
    2/2 [==============================] - 24s 9s/step - loss: 5.8720 - accuracy: 0.2340 - val_loss: 8.4727 - val_accuracy: 0.0000e+00
    Epoch 174/300
    2/2 [==============================] - 24s 9s/step - loss: 6.0322 - accuracy: 0.2553 - val_loss: 10.5503 - val_accuracy: 0.0000e+00
    Epoch 175/300
    2/2 [==============================] - 24s 10s/step - loss: 7.1320 - accuracy: 0.2553 - val_loss: 10.7047 - val_accuracy: 0.0000e+00
    Epoch 176/300
    2/2 [==============================] - 25s 10s/step - loss: 6.6508 - accuracy: 0.2553 - val_loss: 9.2608 - val_accuracy: 0.0000e+00
    Epoch 177/300
    2/2 [==============================] - 24s 9s/step - loss: 6.4937 - accuracy: 0.2979 - val_loss: 8.3958 - val_accuracy: 0.0000e+00
    Epoch 178/300
    2/2 [==============================] - 25s 10s/step - loss: 6.2005 - accuracy: 0.2340 - val_loss: 9.8365 - val_accuracy: 0.0000e+00
    Epoch 179/300
    2/2 [==============================] - 24s 9s/step - loss: 6.0496 - accuracy: 0.2766 - val_loss: 9.3915 - val_accuracy: 0.0000e+00
    Epoch 180/300
    2/2 [==============================] - 25s 10s/step - loss: 5.9031 - accuracy: 0.2979 - val_loss: 9.0731 - val_accuracy: 0.0000e+00
    Epoch 181/300
    2/2 [==============================] - 24s 9s/step - loss: 6.4725 - accuracy: 0.3404 - val_loss: 9.7463 - val_accuracy: 0.0000e+00
    Epoch 182/300
    2/2 [==============================] - 24s 9s/step - loss: 6.2210 - accuracy: 0.2979 - val_loss: 8.9014 - val_accuracy: 0.0000e+00
    Epoch 183/300
    2/2 [==============================] - 24s 10s/step - loss: 6.2594 - accuracy: 0.1489 - val_loss: 7.4989 - val_accuracy: 0.3333
    Epoch 184/300
    2/2 [==============================] - 25s 10s/step - loss: 5.9538 - accuracy: 0.2979 - val_loss: 7.5737 - val_accuracy: 0.0000e+00
    Epoch 185/300
    2/2 [==============================] - 24s 9s/step - loss: 5.6603 - accuracy: 0.2979 - val_loss: 7.1247 - val_accuracy: 0.0000e+00
    Epoch 186/300
    2/2 [==============================] - 24s 10s/step - loss: 5.8058 - accuracy: 0.2553 - val_loss: 7.9423 - val_accuracy: 0.3333
    Epoch 187/300
    2/2 [==============================] - 23s 8s/step - loss: 5.6091 - accuracy: 0.2340 - val_loss: 7.7799 - val_accuracy: 0.0000e+00
    Epoch 188/300
    2/2 [==============================] - 27s 10s/step - loss: 5.5977 - accuracy: 0.3191 - val_loss: 7.3222 - val_accuracy: 0.0000e+00
    Epoch 189/300
    2/2 [==============================] - 26s 8s/step - loss: 5.4251 - accuracy: 0.2979 - val_loss: 7.3122 - val_accuracy: 0.0000e+00
    Epoch 190/300
    2/2 [==============================] - 25s 10s/step - loss: 5.4557 - accuracy: 0.2979 - val_loss: 7.8997 - val_accuracy: 0.0000e+00
    Epoch 191/300
    2/2 [==============================] - 25s 10s/step - loss: 5.3411 - accuracy: 0.2979 - val_loss: 7.6023 - val_accuracy: 0.0000e+00
    Epoch 192/300
    2/2 [==============================] - 25s 10s/step - loss: 5.3724 - accuracy: 0.2979 - val_loss: 8.8186 - val_accuracy: 0.0000e+00
    Epoch 193/300
    2/2 [==============================] - 24s 8s/step - loss: 5.4003 - accuracy: 0.2766 - val_loss: 9.5091 - val_accuracy: 0.0000e+00
    Epoch 194/300
    2/2 [==============================] - 26s 8s/step - loss: 5.2829 - accuracy: 0.3191 - val_loss: 9.5058 - val_accuracy: 0.0000e+00
    Epoch 195/300
    2/2 [==============================] - 25s 8s/step - loss: 5.4232 - accuracy: 0.2766 - val_loss: 9.8844 - val_accuracy: 0.0000e+00
    Epoch 196/300
    2/2 [==============================] - 25s 10s/step - loss: 5.4056 - accuracy: 0.2553 - val_loss: 9.5723 - val_accuracy: 0.0000e+00
    Epoch 197/300
    2/2 [==============================] - 24s 8s/step - loss: 5.6120 - accuracy: 0.3404 - val_loss: 10.9377 - val_accuracy: 0.0000e+00
    Epoch 198/300
    2/2 [==============================] - 24s 8s/step - loss: 5.7458 - accuracy: 0.3191 - val_loss: 11.7786 - val_accuracy: 0.0000e+00
    Epoch 199/300
    2/2 [==============================] - 26s 8s/step - loss: 6.0085 - accuracy: 0.2979 - val_loss: 10.6695 - val_accuracy: 0.0000e+00
    Epoch 200/300
    2/2 [==============================] - 28s 10s/step - loss: 5.7347 - accuracy: 0.2979 - val_loss: 10.9147 - val_accuracy: 0.0000e+00
    Epoch 201/300
    2/2 [==============================] - 24s 8s/step - loss: 5.7016 - accuracy: 0.2128 - val_loss: 11.2058 - val_accuracy: 0.0000e+00
    Epoch 202/300
    2/2 [==============================] - 25s 8s/step - loss: 5.9232 - accuracy: 0.2340 - val_loss: 8.8522 - val_accuracy: 0.0000e+00
    Epoch 203/300
    2/2 [==============================] - 28s 10s/step - loss: 5.6721 - accuracy: 0.3617 - val_loss: 8.6458 - val_accuracy: 0.0000e+00
    Epoch 204/300
    2/2 [==============================] - 26s 10s/step - loss: 5.1299 - accuracy: 0.3617 - val_loss: 7.6526 - val_accuracy: 0.3333
    Epoch 205/300
    2/2 [==============================] - 27s 11s/step - loss: 5.3319 - accuracy: 0.2979 - val_loss: 8.8254 - val_accuracy: 0.3333
    Epoch 206/300
    2/2 [==============================] - 27s 10s/step - loss: 5.3621 - accuracy: 0.3191 - val_loss: 8.4866 - val_accuracy: 0.0000e+00
    Epoch 207/300
    2/2 [==============================] - 24s 8s/step - loss: 5.4331 - accuracy: 0.3404 - val_loss: 9.2725 - val_accuracy: 0.0000e+00
    Epoch 208/300
    2/2 [==============================] - 27s 11s/step - loss: 5.2564 - accuracy: 0.3617 - val_loss: 9.2874 - val_accuracy: 0.3333
    Epoch 209/300
    2/2 [==============================] - 26s 10s/step - loss: 5.4140 - accuracy: 0.4255 - val_loss: 8.4435 - val_accuracy: 0.0000e+00
    Epoch 210/300
    2/2 [==============================] - 26s 10s/step - loss: 5.5229 - accuracy: 0.3191 - val_loss: 9.4546 - val_accuracy: 0.0000e+00
    Epoch 211/300
    2/2 [==============================] - 24s 8s/step - loss: 5.1496 - accuracy: 0.4043 - val_loss: 8.5208 - val_accuracy: 0.3333
    Epoch 212/300
    2/2 [==============================] - 27s 9s/step - loss: 5.3042 - accuracy: 0.3191 - val_loss: 9.8423 - val_accuracy: 0.0000e+00
    Epoch 213/300
    2/2 [==============================] - 26s 10s/step - loss: 5.2458 - accuracy: 0.2766 - val_loss: 8.6963 - val_accuracy: 0.0000e+00
    Epoch 214/300
    2/2 [==============================] - 26s 10s/step - loss: 5.1787 - accuracy: 0.2128 - val_loss: 7.2053 - val_accuracy: 0.3333
    Epoch 215/300
    2/2 [==============================] - 26s 11s/step - loss: 4.9454 - accuracy: 0.3404 - val_loss: 8.3741 - val_accuracy: 0.0000e+00
    Epoch 216/300
    2/2 [==============================] - 26s 9s/step - loss: 4.8992 - accuracy: 0.4043 - val_loss: 8.2869 - val_accuracy: 0.0000e+00
    Epoch 217/300
    2/2 [==============================] - 28s 10s/step - loss: 4.8544 - accuracy: 0.3404 - val_loss: 7.6513 - val_accuracy: 0.3333
    Epoch 218/300
    2/2 [==============================] - 26s 10s/step - loss: 4.8941 - accuracy: 0.4468 - val_loss: 8.4066 - val_accuracy: 0.0000e+00
    Epoch 219/300
    2/2 [==============================] - 24s 8s/step - loss: 5.0440 - accuracy: 0.2553 - val_loss: 10.5134 - val_accuracy: 0.0000e+00
    Epoch 220/300
    2/2 [==============================] - 29s 10s/step - loss: 5.2134 - accuracy: 0.2766 - val_loss: 10.8861 - val_accuracy: 0.0000e+00
    Epoch 221/300
    2/2 [==============================] - 26s 10s/step - loss: 5.0683 - accuracy: 0.3404 - val_loss: 10.4644 - val_accuracy: 0.0000e+00
    Epoch 222/300
    2/2 [==============================] - 26s 10s/step - loss: 5.3071 - accuracy: 0.2979 - val_loss: 9.7724 - val_accuracy: 0.0000e+00
    Epoch 223/300
    2/2 [==============================] - 26s 10s/step - loss: 5.2477 - accuracy: 0.2553 - val_loss: 7.9826 - val_accuracy: 0.0000e+00
    Epoch 224/300
    2/2 [==============================] - 27s 10s/step - loss: 5.3968 - accuracy: 0.2553 - val_loss: 7.9223 - val_accuracy: 0.0000e+00
    Epoch 225/300
    2/2 [==============================] - 24s 8s/step - loss: 5.3507 - accuracy: 0.3617 - val_loss: 6.7553 - val_accuracy: 0.3333
    Epoch 226/300
    2/2 [==============================] - 29s 10s/step - loss: 5.2565 - accuracy: 0.3617 - val_loss: 7.3255 - val_accuracy: 0.0000e+00
    Epoch 227/300
    2/2 [==============================] - 27s 11s/step - loss: 5.3888 - accuracy: 0.3617 - val_loss: 7.5570 - val_accuracy: 0.0000e+00
    Epoch 228/300
    2/2 [==============================] - 26s 10s/step - loss: 5.4549 - accuracy: 0.3830 - val_loss: 7.7901 - val_accuracy: 0.3333
    Epoch 229/300
    2/2 [==============================] - 27s 11s/step - loss: 5.6232 - accuracy: 0.2979 - val_loss: 7.9327 - val_accuracy: 0.0000e+00
    Epoch 230/300
    2/2 [==============================] - 27s 10s/step - loss: 5.3907 - accuracy: 0.2553 - val_loss: 9.0099 - val_accuracy: 0.0000e+00
    Epoch 231/300
    2/2 [==============================] - 26s 10s/step - loss: 5.3587 - accuracy: 0.3830 - val_loss: 8.2299 - val_accuracy: 0.0000e+00
    Epoch 232/300
    2/2 [==============================] - 24s 8s/step - loss: 5.0500 - accuracy: 0.3617 - val_loss: 8.0615 - val_accuracy: 0.0000e+00
    Epoch 233/300
    2/2 [==============================] - 27s 8s/step - loss: 4.9607 - accuracy: 0.3617 - val_loss: 8.6512 - val_accuracy: 0.0000e+00
    Epoch 234/300
    2/2 [==============================] - 28s 10s/step - loss: 4.7209 - accuracy: 0.4468 - val_loss: 8.3021 - val_accuracy: 0.0000e+00
    Epoch 235/300
    2/2 [==============================] - 24s 8s/step - loss: 5.0352 - accuracy: 0.3404 - val_loss: 843.8481 - val_accuracy: 0.0000e+00
    Epoch 236/300
    2/2 [==============================] - 27s 8s/step - loss: 8.7277 - accuracy: 0.0426 - val_loss: 649.7225 - val_accuracy: 0.3333
    Epoch 237/300
    2/2 [==============================] - 25s 8s/step - loss: 10.0311 - accuracy: 0.1702 - val_loss: 918.6096 - val_accuracy: 0.0000e+00
    Epoch 238/300
    2/2 [==============================] - 29s 10s/step - loss: 9.7283 - accuracy: 0.1915 - val_loss: 681.0095 - val_accuracy: 0.0000e+00
    Epoch 239/300
    2/2 [==============================] - 27s 11s/step - loss: 9.6996 - accuracy: 0.2979 - val_loss: 380.2573 - val_accuracy: 0.0000e+00
    Epoch 240/300
    2/2 [==============================] - 24s 8s/step - loss: 10.2018 - accuracy: 0.1277 - val_loss: 321.7005 - val_accuracy: 0.0000e+00
    Epoch 241/300
    2/2 [==============================] - 26s 8s/step - loss: 9.9110 - accuracy: 0.1702 - val_loss: 309.2009 - val_accuracy: 0.0000e+00
    Epoch 242/300
    2/2 [==============================] - 26s 8s/step - loss: 9.6154 - accuracy: 0.1702 - val_loss: 230.3245 - val_accuracy: 0.0000e+00
    Epoch 243/300
    2/2 [==============================] - 27s 10s/step - loss: 9.6813 - accuracy: 0.2128 - val_loss: 161.2798 - val_accuracy: 0.0000e+00
    Epoch 244/300
    2/2 [==============================] - 26s 10s/step - loss: 9.9749 - accuracy: 0.2128 - val_loss: 154.3258 - val_accuracy: 0.0000e+00
    Epoch 245/300
    2/2 [==============================] - 25s 10s/step - loss: 10.0617 - accuracy: 0.2340 - val_loss: 76.5403 - val_accuracy: 0.0000e+00
    Epoch 246/300
    2/2 [==============================] - 26s 10s/step - loss: 10.4079 - accuracy: 0.1915 - val_loss: 183.5860 - val_accuracy: 0.0000e+00
    Epoch 247/300
    2/2 [==============================] - 26s 10s/step - loss: 10.4903 - accuracy: 0.2340 - val_loss: 168.8780 - val_accuracy: 0.0000e+00
    Epoch 248/300
    2/2 [==============================] - 26s 10s/step - loss: 10.2257 - accuracy: 0.2340 - val_loss: 174.1894 - val_accuracy: 0.0000e+00
    Epoch 249/300
    2/2 [==============================] - 24s 8s/step - loss: 9.9662 - accuracy: 0.1915 - val_loss: 142.7682 - val_accuracy: 0.0000e+00
    Epoch 250/300
    2/2 [==============================] - 27s 8s/step - loss: 9.7653 - accuracy: 0.2553 - val_loss: 107.7319 - val_accuracy: 0.0000e+00
    Epoch 251/300
    2/2 [==============================] - 28s 10s/step - loss: 9.5524 - accuracy: 0.3191 - val_loss: 82.6017 - val_accuracy: 0.0000e+00
    Epoch 252/300
    2/2 [==============================] - 24s 8s/step - loss: 10.2137 - accuracy: 0.2340 - val_loss: 105.9199 - val_accuracy: 0.0000e+00
    Epoch 253/300
    2/2 [==============================] - 27s 8s/step - loss: 9.8379 - accuracy: 0.1915 - val_loss: 65.9089 - val_accuracy: 0.0000e+00
    Epoch 254/300
    2/2 [==============================] - 24s 8s/step - loss: 9.7689 - accuracy: 0.2340 - val_loss: 66.4379 - val_accuracy: 0.0000e+00
    Epoch 255/300
    2/2 [==============================] - 28s 10s/step - loss: 9.4802 - accuracy: 0.2340 - val_loss: 42.0179 - val_accuracy: 0.0000e+00
    Epoch 256/300
    2/2 [==============================] - 26s 10s/step - loss: 9.5559 - accuracy: 0.2979 - val_loss: 45.1938 - val_accuracy: 0.0000e+00
    Epoch 257/300
    2/2 [==============================] - 25s 9s/step - loss: 9.4285 - accuracy: 0.2553 - val_loss: 46.9381 - val_accuracy: 0.0000e+00
    Epoch 258/300
    2/2 [==============================] - 27s 9s/step - loss: 9.3100 - accuracy: 0.2766 - val_loss: 38.5072 - val_accuracy: 0.0000e+00
    Epoch 259/300
    2/2 [==============================] - 28s 10s/step - loss: 9.0322 - accuracy: 0.2340 - val_loss: 33.5944 - val_accuracy: 0.0000e+00
    Epoch 260/300
    2/2 [==============================] - 26s 10s/step - loss: 8.9335 - accuracy: 0.2766 - val_loss: 40.6609 - val_accuracy: 0.0000e+00
    Epoch 261/300
    2/2 [==============================] - 26s 11s/step - loss: 8.9319 - accuracy: 0.1277 - val_loss: 37.2037 - val_accuracy: 0.0000e+00
    Epoch 262/300
    2/2 [==============================] - 26s 10s/step - loss: 8.7755 - accuracy: 0.2979 - val_loss: 26.3033 - val_accuracy: 0.0000e+00
    Epoch 263/300
    2/2 [==============================] - 26s 10s/step - loss: 8.5852 - accuracy: 0.3191 - val_loss: 20.3349 - val_accuracy: 0.3333
    Epoch 264/300
    2/2 [==============================] - 27s 10s/step - loss: 8.6198 - accuracy: 0.3617 - val_loss: 25.4459 - val_accuracy: 0.3333
    Epoch 265/300
    2/2 [==============================] - 26s 10s/step - loss: 8.7971 - accuracy: 0.2128 - val_loss: 22.3820 - val_accuracy: 0.0000e+00
    Epoch 266/300
    2/2 [==============================] - 25s 9s/step - loss: 8.4080 - accuracy: 0.3404 - val_loss: 32.4917 - val_accuracy: 0.0000e+00
    Epoch 267/300
    2/2 [==============================] - 27s 8s/step - loss: 10.0312 - accuracy: 0.1702 - val_loss: 23.7541 - val_accuracy: 0.3333
    Epoch 268/300
    2/2 [==============================] - 26s 8s/step - loss: 9.7165 - accuracy: 0.2979 - val_loss: 21.1377 - val_accuracy: 0.0000e+00
    Epoch 269/300
    2/2 [==============================] - 27s 9s/step - loss: 9.3922 - accuracy: 0.2979 - val_loss: 26.3590 - val_accuracy: 0.0000e+00
    Epoch 270/300
    2/2 [==============================] - 27s 8s/step - loss: 10.1263 - accuracy: 0.3404 - val_loss: 26.7975 - val_accuracy: 0.0000e+00
    Epoch 271/300
    2/2 [==============================] - 29s 10s/step - loss: 9.9990 - accuracy: 0.3617 - val_loss: 31.0863 - val_accuracy: 0.0000e+00
    Epoch 272/300
    2/2 [==============================] - 25s 9s/step - loss: 10.1557 - accuracy: 0.2979 - val_loss: 23.4631 - val_accuracy: 0.0000e+00
    Epoch 273/300
    2/2 [==============================] - 28s 10s/step - loss: 10.0230 - accuracy: 0.3191 - val_loss: 15.0136 - val_accuracy: 0.3333
    Epoch 274/300
    2/2 [==============================] - 26s 9s/step - loss: 9.8610 - accuracy: 0.2128 - val_loss: 12.8542 - val_accuracy: 0.3333
    Epoch 275/300
    2/2 [==============================] - 27s 8s/step - loss: 9.9647 - accuracy: 0.3191 - val_loss: 13.3425 - val_accuracy: 0.0000e+00
    Epoch 276/300
    2/2 [==============================] - 29s 11s/step - loss: 9.5320 - accuracy: 0.3191 - val_loss: 9.2954 - val_accuracy: 0.3333
    Epoch 277/300
    2/2 [==============================] - 27s 10s/step - loss: 9.6962 - accuracy: 0.1702 - val_loss: 44.9584 - val_accuracy: 0.0000e+00
    Epoch 278/300
    2/2 [==============================] - 25s 8s/step - loss: 9.5555 - accuracy: 0.3404 - val_loss: 24.9013 - val_accuracy: 0.0000e+00
    Epoch 279/300
    2/2 [==============================] - 26s 8s/step - loss: 9.2310 - accuracy: 0.3617 - val_loss: 21.4655 - val_accuracy: 0.0000e+00
    Epoch 280/300
    2/2 [==============================] - 28s 10s/step - loss: 9.0059 - accuracy: 0.4043 - val_loss: 22.0997 - val_accuracy: 0.0000e+00
    Epoch 281/300
    2/2 [==============================] - 25s 8s/step - loss: 9.0636 - accuracy: 0.2553 - val_loss: 18.8740 - val_accuracy: 0.0000e+00
    Epoch 282/300
    2/2 [==============================] - 29s 10s/step - loss: 9.0271 - accuracy: 0.3191 - val_loss: 12.0315 - val_accuracy: 0.0000e+00
    Epoch 283/300
    2/2 [==============================] - 24s 8s/step - loss: 9.6180 - accuracy: 0.2553 - val_loss: 12.1312 - val_accuracy: 0.0000e+00
    Epoch 284/300
    2/2 [==============================] - 27s 11s/step - loss: 9.6123 - accuracy: 0.4043 - val_loss: 19.0593 - val_accuracy: 0.0000e+00
    Epoch 285/300
    2/2 [==============================] - 27s 11s/step - loss: 9.5047 - accuracy: 0.2128 - val_loss: 13.1153 - val_accuracy: 0.0000e+00
    Epoch 286/300
    2/2 [==============================] - 24s 8s/step - loss: 9.0439 - accuracy: 0.3191 - val_loss: 12.8590 - val_accuracy: 0.3333
    Epoch 287/300
    2/2 [==============================] - 26s 8s/step - loss: 9.4264 - accuracy: 0.1702 - val_loss: 57.9203 - val_accuracy: 0.0000e+00
    Epoch 288/300
    2/2 [==============================] - 26s 8s/step - loss: 12.0461 - accuracy: 0.0851 - val_loss: 65.2695 - val_accuracy: 0.0000e+00
    Epoch 289/300
    2/2 [==============================] - 27s 8s/step - loss: 11.2238 - accuracy: 0.1064 - val_loss: 63.7125 - val_accuracy: 0.0000e+00
    Epoch 290/300
    2/2 [==============================] - 26s 9s/step - loss: 10.4028 - accuracy: 0.3191 - val_loss: 50.3963 - val_accuracy: 0.0000e+00
    Epoch 291/300
    2/2 [==============================] - 27s 8s/step - loss: 10.0882 - accuracy: 0.1489 - val_loss: 52.5436 - val_accuracy: 0.0000e+00
    Epoch 292/300
    2/2 [==============================] - 28s 9s/step - loss: 9.6765 - accuracy: 0.2340 - val_loss: 46.8566 - val_accuracy: 0.0000e+00
    Epoch 293/300
    2/2 [==============================] - 26s 8s/step - loss: 8.8649 - accuracy: 0.3617 - val_loss: 43.0911 - val_accuracy: 0.0000e+00
    Epoch 294/300
    2/2 [==============================] - 26s 8s/step - loss: 8.7046 - accuracy: 0.1915 - val_loss: 31.7497 - val_accuracy: 0.0000e+00
    Epoch 295/300
    2/2 [==============================] - 27s 8s/step - loss: 8.2415 - accuracy: 0.2766 - val_loss: 26.0380 - val_accuracy: 0.0000e+00
    Epoch 296/300
    2/2 [==============================] - 29s 10s/step - loss: 8.0092 - accuracy: 0.3830 - val_loss: 27.0478 - val_accuracy: 0.0000e+00
    Epoch 297/300
    2/2 [==============================] - 25s 10s/step - loss: 7.7130 - accuracy: 0.3617 - val_loss: 32.2018 - val_accuracy: 0.0000e+00
    Epoch 298/300
    2/2 [==============================] - 25s 10s/step - loss: 7.9154 - accuracy: 0.2766 - val_loss: 17.1687 - val_accuracy: 0.0000e+00
    Epoch 299/300
    2/2 [==============================] - 23s 7s/step - loss: 7.8340 - accuracy: 0.3404 - val_loss: 14.4879 - val_accuracy: 0.0000e+00
    Epoch 300/300
    2/2 [==============================] - 25s 7s/step - loss: 8.1219 - accuracy: 0.4043 - val_loss: 12.9961 - val_accuracy: 0.0000e+00
    

### TRAINING MODEL3 RESULTS


```python
plot_model(train_model2,"MFCC","STFT")
```


    
![png](output_89_0.png)
    


<a id="model1p"></a>
## TESTING MODELS

### Extracting Test Features


```python
def extract_test(audio, shape,sr=44200, power=2.0):
    results = []
    
    audio = np.split(audio, 10)
    for sub_array in audio:
        spectogram = librosa.feature.melspectrogram(sub_array, sr=sr, power=power, fmin=F_MIN, fmax=F_MAX, n_mels=SHAPE[0])
        spectogram_to_db = librosa.core.amplitude_to_db(np.abs(spectogram))
        spectogram_to_db = resize(spectogram_to_db, shape)
        spectogram_to_db = spectogram_to_db - np.min(spectogram_to_db)
        spectogram_to_db = spectogram_to_db / np.max(spectogram_to_db)
        spectogram_to_db = np.stack((spectogram_to_db,spectogram_to_db,spectogram_to_db))
        spectogram_to_db = spectogram_to_db.reshape(spectogram_to_db.shape[1], spectogram_to_db.shape[2], spectogram_to_db.shape[0])
        results.append(spectogram_to_db)    
        
    return results
```

### MODEL 1 PREDICTIONS


```python
predictions = []
test_path = 'test/'
for root, dirs, files in os.walk(test_path):
    total = len(files)
    for file in files:
        loaded_audio, sample_rate = librosa.load(test_path+file, sr=None)
        test_list = extract_test(loaded_audio, shape=SHAPE, power=2)
        predicted = model1.predict(np.array(test_list))        
        mean_predicted = np.mean(predicted, axis=0)
        predictions.append(mean_predicted) 
        

pred_df= pd.DataFrame(predictions)
pred_df['recording_id']=files
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>recording_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.080504e-09</td>
      <td>0.007347</td>
      <td>3.345541e-07</td>
      <td>0.016200</td>
      <td>3.567303e-01</td>
      <td>1.124120e-08</td>
      <td>0.615388</td>
      <td>1.068712e-10</td>
      <td>1.420055e-10</td>
      <td>4.315502e-03</td>
      <td>...</td>
      <td>1.665693e-11</td>
      <td>3.623006e-14</td>
      <td>4.248965e-13</td>
      <td>7.402348e-10</td>
      <td>1.210503e-06</td>
      <td>0.000018</td>
      <td>7.639517e-08</td>
      <td>3.171967e-20</td>
      <td>8.547511e-08</td>
      <td>0a45adab2.flac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.529331e-20</td>
      <td>0.000046</td>
      <td>9.067871e-13</td>
      <td>0.000056</td>
      <td>2.601832e-18</td>
      <td>2.115314e-34</td>
      <td>0.884496</td>
      <td>3.017063e-15</td>
      <td>1.368813e-08</td>
      <td>1.643283e-15</td>
      <td>...</td>
      <td>6.868946e-19</td>
      <td>0.000000e+00</td>
      <td>3.068396e-33</td>
      <td>8.084235e-04</td>
      <td>2.469096e-36</td>
      <td>0.113859</td>
      <td>7.356232e-04</td>
      <td>3.371799e-25</td>
      <td>1.625317e-08</td>
      <td>0a64c67f8.flac</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.132677e-10</td>
      <td>0.014043</td>
      <td>1.498624e-07</td>
      <td>0.007973</td>
      <td>5.910011e-09</td>
      <td>1.752820e-20</td>
      <td>0.977974</td>
      <td>1.529625e-09</td>
      <td>5.089672e-07</td>
      <td>1.105688e-08</td>
      <td>...</td>
      <td>1.464642e-13</td>
      <td>1.206017e-24</td>
      <td>5.408910e-19</td>
      <td>8.433912e-09</td>
      <td>1.815115e-20</td>
      <td>0.000010</td>
      <td>3.171049e-08</td>
      <td>2.427871e-18</td>
      <td>5.855332e-10</td>
      <td>0a6a36fb6.flac</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.451705e-10</td>
      <td>0.005174</td>
      <td>9.978023e-07</td>
      <td>0.009980</td>
      <td>2.795960e-05</td>
      <td>2.792010e-12</td>
      <td>0.852403</td>
      <td>1.274098e-08</td>
      <td>1.136118e-06</td>
      <td>6.773809e-05</td>
      <td>...</td>
      <td>7.957595e-09</td>
      <td>1.526996e-17</td>
      <td>5.563854e-15</td>
      <td>4.891077e-04</td>
      <td>2.172680e-12</td>
      <td>0.128712</td>
      <td>3.018875e-03</td>
      <td>1.407660e-14</td>
      <td>1.246509e-04</td>
      <td>0a930ca0f.flac</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.515664e-14</td>
      <td>0.001270</td>
      <td>1.621581e-09</td>
      <td>0.002200</td>
      <td>3.906065e-06</td>
      <td>5.887386e-16</td>
      <td>0.975332</td>
      <td>9.313724e-13</td>
      <td>4.180321e-09</td>
      <td>3.596958e-06</td>
      <td>...</td>
      <td>6.164497e-13</td>
      <td>3.447883e-24</td>
      <td>1.576844e-21</td>
      <td>5.034917e-06</td>
      <td>1.624590e-15</td>
      <td>0.020995</td>
      <td>1.832303e-04</td>
      <td>6.372899e-22</td>
      <td>7.790573e-06</td>
      <td>0ae65727d.flac</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



<a id="model2p"></a>
### MODEL2 PREDICTIONS


```python
predictions1 = []
test_path = 'test/'
for root, dirs, files in os.walk(test_path):
    total = len(files)
    for file in files:
        loaded_audio, sample_rate = librosa.load(test_path+file, sr=None)
        test_list = extract_test(loaded_audio, shape=SHAPE, power=2)
        predicted1 = model2.predict(np.array(test_list))        
        mean_predicted1 = np.mean(predicted1, axis=0)
        predictions1.append(mean_predicted1) 
        

pred_df1= pd.DataFrame(predictions1)
pred_df1['recording_id']=files
pred_df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>recording_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001069</td>
      <td>0.005527</td>
      <td>8.457954e-05</td>
      <td>0.042035</td>
      <td>0.028368</td>
      <td>0.093404</td>
      <td>2.599506e-03</td>
      <td>0.310025</td>
      <td>0.008337</td>
      <td>0.042003</td>
      <td>...</td>
      <td>0.127973</td>
      <td>0.003791</td>
      <td>0.003538</td>
      <td>0.000970</td>
      <td>0.095804</td>
      <td>0.000352</td>
      <td>0.000153</td>
      <td>0.000710</td>
      <td>0.000246</td>
      <td>0a45adab2.flac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000002</td>
      <td>0.001980</td>
      <td>2.088225e-08</td>
      <td>0.000135</td>
      <td>0.000022</td>
      <td>0.000120</td>
      <td>1.755373e-07</td>
      <td>0.879222</td>
      <td>0.000918</td>
      <td>0.000688</td>
      <td>...</td>
      <td>0.069875</td>
      <td>0.000006</td>
      <td>0.000847</td>
      <td>0.001385</td>
      <td>0.000479</td>
      <td>0.002068</td>
      <td>0.000087</td>
      <td>0.000060</td>
      <td>0.000151</td>
      <td>0a64c67f8.flac</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000006</td>
      <td>0.003205</td>
      <td>1.292973e-07</td>
      <td>0.001158</td>
      <td>0.000153</td>
      <td>0.000916</td>
      <td>2.429449e-06</td>
      <td>0.738253</td>
      <td>0.002705</td>
      <td>0.001980</td>
      <td>...</td>
      <td>0.158096</td>
      <td>0.000061</td>
      <td>0.002852</td>
      <td>0.001208</td>
      <td>0.005310</td>
      <td>0.001119</td>
      <td>0.000073</td>
      <td>0.000209</td>
      <td>0.000160</td>
      <td>0a6a36fb6.flac</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001355</td>
      <td>0.045003</td>
      <td>3.424767e-04</td>
      <td>0.014415</td>
      <td>0.003903</td>
      <td>0.033007</td>
      <td>2.108815e-03</td>
      <td>0.165922</td>
      <td>0.065613</td>
      <td>0.031725</td>
      <td>...</td>
      <td>0.399590</td>
      <td>0.014082</td>
      <td>0.037211</td>
      <td>0.011848</td>
      <td>0.065041</td>
      <td>0.007549</td>
      <td>0.002139</td>
      <td>0.015882</td>
      <td>0.003318</td>
      <td>0a930ca0f.flac</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000217</td>
      <td>0.007114</td>
      <td>1.348310e-05</td>
      <td>0.011648</td>
      <td>0.004095</td>
      <td>0.016474</td>
      <td>3.158124e-04</td>
      <td>0.565328</td>
      <td>0.008651</td>
      <td>0.014219</td>
      <td>...</td>
      <td>0.187429</td>
      <td>0.001119</td>
      <td>0.004463</td>
      <td>0.001564</td>
      <td>0.031276</td>
      <td>0.000878</td>
      <td>0.000166</td>
      <td>0.000692</td>
      <td>0.000305</td>
      <td>0ae65727d.flac</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



<a id="model3p"></a>
### MODEL3 PREDICTIONS


```python
predictions2 = []
test_path = 'test/'
for root, dirs, files in os.walk(test_path):
    total = len(files)
    for file in files:
        loaded_audio, sample_rate = librosa.load(test_path+file, sr=None)
        test_list = extract_test(loaded_audio, shape=SHAPE, power=2)
        predicted2 = model3.predict(np.array(test_list))        
        mean_predicted2 = np.mean(predicted2, axis=0)
        predictions2.append(mean_predicted2) 
        

pred_df2= pd.DataFrame(predictions2)
pred_df2['recording_id']=files
pred_df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>recording_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000927</td>
      <td>2.190333e-09</td>
      <td>0.000039</td>
      <td>0.000008</td>
      <td>0.002049</td>
      <td>0.287415</td>
      <td>4.129200e-10</td>
      <td>4.607855e-09</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>...</td>
      <td>6.363754e-09</td>
      <td>0.082213</td>
      <td>1.299671e-08</td>
      <td>1.809983e-10</td>
      <td>0.000139</td>
      <td>2.444151e-11</td>
      <td>7.230742e-12</td>
      <td>0.093117</td>
      <td>3.879982e-12</td>
      <td>0a45adab2.flac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002749</td>
      <td>1.755330e-01</td>
      <td>0.000644</td>
      <td>0.003182</td>
      <td>0.001097</td>
      <td>0.001093</td>
      <td>2.889503e-03</td>
      <td>3.211440e-02</td>
      <td>0.039874</td>
      <td>0.032843</td>
      <td>...</td>
      <td>8.273365e-02</td>
      <td>0.000092</td>
      <td>1.461090e-03</td>
      <td>1.432713e-01</td>
      <td>0.000314</td>
      <td>1.694520e-01</td>
      <td>1.016418e-01</td>
      <td>0.000192</td>
      <td>1.203398e-01</td>
      <td>0a64c67f8.flac</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.013320</td>
      <td>1.057429e-01</td>
      <td>0.007988</td>
      <td>0.007166</td>
      <td>0.005406</td>
      <td>0.019079</td>
      <td>6.660588e-03</td>
      <td>4.581717e-02</td>
      <td>0.112930</td>
      <td>0.040356</td>
      <td>...</td>
      <td>1.409234e-01</td>
      <td>0.004641</td>
      <td>2.439690e-02</td>
      <td>1.124347e-01</td>
      <td>0.006617</td>
      <td>1.805575e-01</td>
      <td>4.358528e-02</td>
      <td>0.012877</td>
      <td>4.150203e-02</td>
      <td>0a6a36fb6.flac</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.015137</td>
      <td>1.606136e-05</td>
      <td>0.004763</td>
      <td>0.000101</td>
      <td>0.004175</td>
      <td>0.199478</td>
      <td>2.848240e-06</td>
      <td>1.173499e-05</td>
      <td>0.000832</td>
      <td>0.000295</td>
      <td>...</td>
      <td>3.961018e-05</td>
      <td>0.085555</td>
      <td>1.351801e-04</td>
      <td>1.936770e-05</td>
      <td>0.001427</td>
      <td>3.455422e-05</td>
      <td>3.531031e-06</td>
      <td>0.333597</td>
      <td>9.464250e-07</td>
      <td>0a930ca0f.flac</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.036667</td>
      <td>6.840981e-02</td>
      <td>0.004765</td>
      <td>0.021863</td>
      <td>0.048079</td>
      <td>0.184686</td>
      <td>3.339795e-03</td>
      <td>1.083848e-02</td>
      <td>0.052520</td>
      <td>0.093388</td>
      <td>...</td>
      <td>3.004480e-02</td>
      <td>0.021823</td>
      <td>5.210736e-04</td>
      <td>1.230688e-02</td>
      <td>0.004958</td>
      <td>2.082502e-03</td>
      <td>4.488600e-03</td>
      <td>0.031726</td>
      <td>1.138453e-02</td>
      <td>0ae65727d.flac</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



<a id="compare"></a>
## COMPARISION OF MODELS

### MODEL 1


```python
species = list(pred_df.columns)
species.remove('recording_id')
plt.figure(figsize=(10,5))
pred_df.boxplot(species)

plt.xlabel('Types of species based on songtype and species')
plt.ylabel('Percentage of sound classified')
plt.title("Predictions of different species by LOG_MEL and MFCC Model run")
```




    Text(0.5, 1.0, 'Predictions of different species by LOG_MEL and MFCC Model run')




    
![png](output_101_1.png)
    


#### MODEL 2


```python
species1 = list(pred_df1.columns)
species1.remove('recording_id')

plt.figure(figsize=(15,8))
pred_df1.boxplot(species1)

plt.xlabel('Types of species based on songtype and species')
plt.ylabel('Percentage of sound classified')
plt.title("Predictions of different species by LOG_MEL and STFT Model run")
```




    Text(0.5, 1.0, 'Predictions of different species by LOG_MEL and STFT Model run')




    
![png](output_103_1.png)
    


#### MODEL 3


```python
species2 = list(pred_df2.columns)
species2.remove('recording_id')

plt.figure(figsize=(15,8))
pred_df2.boxplot(species2)
plt.xlabel('Types of species based on songtype and species')
plt.ylabel('Percentage of sound classified')
plt.title("Predictions of different species by MFCC and STFT Model run")
```




    Text(0.5, 1.0, 'Predictions of different species by MFCC and STFT Model run')




    
![png](output_105_1.png)
    


<a id="conclusion"></a>
## CONCLUSION

- The model2 and model3 have good prediction rates based on the boxplots of each.
- The STFT model tends to produce good training results and accuracy as the combination of STFT with other spectrograms had better results than the one without it.
- Model 3 is successful to predict most of the type of species.
