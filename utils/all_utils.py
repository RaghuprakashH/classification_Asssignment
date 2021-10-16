from typing import cast
import pandas as pd
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split


def prepare_data(random_state,path):
    new_df = []
    path1 = path
    #path1 = ['C:\CassFlipkratScrappingProject\S1_Dataset','C:\CassFlipkratScrappingProject\S2_Dataset']

    var = []
    j = 0
    tst1 = []
    tst3 = pd.DataFrame()
    for  path2 in path1:
        A = (os.listdir(path2))
        for i in A:

            j = j + 1
            #print(i)
            if i != 'README.txt':
                #tst1.append(['a','b','c','d','e','f','g','h','i'])
                cols = ['Time', 'Acceler_Front', 'Acceler_Vert', 'Acceler_later', 'Id_sensor', 'RSSI', 'Phase', 'Frequency', 'Label']
                tst1.append(pd.read_csv(os.path.join(path2,i),sep=',',names=cols))
    #print(tst1)

       #print(tst1.isnull().sum())
       #print(tst1.head(5))

    A = pd.concat(tst1,ignore_index=False)
    df = pd.DataFrame(A)    
    df.reset_index(inplace=True)
    df.drop(columns=['index'],inplace=True)
    df.Id_sensor = df.Id_sensor.astype('float64')
    data = df.copy()
    X = data.iloc[:,:-1]
    y = data.Label
    x_train , x_test, y_train,y_test = train_test_split(X,y,random_state = random_state)
    return x_train , x_test, y_train,y_test




