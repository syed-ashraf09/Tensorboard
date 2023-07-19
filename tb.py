import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\Ashraf\\Documents\\Refactored_Py_DS_ML_Bootcamp-master\\22-Deep Learning\\cancer_classification.csv")
x=df.drop('benign_0__mal_1',axis=1).values
y=df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
xtr=scaler.fit_transform(xtr)
xte=scaler.transform(xte)
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
model=Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping,TensorBoard
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

log_directory='logs\\fit'# to save each time as keeping logs

board=TensorBoard(log_dir=log_directory,
                  histogram_freq=1,
                  write_graph=True,
                  write_images=True,
                  update_freq='epoch',
                  profile_batch=2,
                  embeddings_freq=1)
model.fit(x=xtr,y=ytr,epochs=600,validation_data=(xte,yte),verbose=1,callbacks=[early_stop,board])

#terminal in pycharm, go to the logs directory ie
#C:\Users\Ashraf\PycharmProjects\pythonProject\ML\deep in this case
# then tensorflow --logdir logs\fit



