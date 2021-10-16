from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np
import  pandas as pd
from utils.all_utils import prepare_data
from utils.model import Baggage_classify_mod
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import pickle
from  cassandra.query import SimpleStatement,BatchStatement
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster


def main(path):
    if path != ' ':    
        x_train , x_test, y_train,y_test = prepare_data(random_state=20,path=path)
        bag_dt_dect = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100)
        Baggage_classify_mod(modelname=bag_dt_dect,save_modelname='ABC')
        bag_dt_dect = Baggage_classify_mod.fit_func(modelname=bag_dt_dect,x_train=x_train,y_train=y_train)
        pickle.dump(bag_dt_dect, open('tutor_model.pkl','wb'))

#path= ['C:\CassFlipkratScrappingProject\S1_Dataset','C:\CassFlipkratScrappingProject\S2_Dataset']
path = ' '
main(path=path)
app = Flask(__name__)
model = pickle.load(open('tutor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def index():    

    if request.method == "POST":
        try:
            if request.form:
                cloud_config = {
                    'secure_connect_bundle': 'secure-connect-test1.zip'
                }

                auth_provider = PlainTextAuthProvider('DzgSXTSzQgWNpYocAWPpQzAX',
                                                      '27C9coC--crqmF0MiZldjv9Kg8NyhTzMP66SOPbHtiaNOWcidhyBz1FuOIuUp.,p2CajK266pu2QEhLkCNs4Zkt6qQaSce2cS_+10a9clpH6UhkdUkNtuBoTczw8sK_X')
                cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
                session = cluster.connect()

                countrec = session.execute("SELECT COUNT(*) FROM ineuron3.Health_db;").one()
                cno1 = countrec[0]
                a_cno1 = cno1 + 1

                data = dict(request.form).values()
                print(data)
                H = 0
                for G in data:
                    H = H + 1
                    if H == 5:
                       A5 = float(G)   
                       
                    if H == 1:
                       A1 = float(G)
                       
                    if H == 2:
                       A2 = float(G)
                       
                    if H == 3:
                       A3 = float(G)
                       
                    if H == 4:
                       A4 = float(G)
                       
                    if H == 6:
                       A6 = float(G)
                       
                    if H == 7:
                       A7 = float(G)
                       
                    if H == 8:
                       A8 = float(G)
                       

                data1 = pd.DataFrame({'Time': [A1],'Acceler_Front': [A2],'Acceler_Vert': [A3],'Acceler_later': [A4],'Id_sensor': [A5],'RSSI': [A6],'Phase': [A7],'Frequency': [A8]})
                #data1 = [list(map(float, data))]
                #features = [x for x in request.form.values()]
                #int_features =[]    
                model_predict = Baggage_classify_mod.predict_func(model,data1)                                           
                response = model_predict
                A9 = str(response[0])
                final_data = [(a_cno1,A1,A2,A3,A4,A5,A6,A7,A8,A9)]

                qr1 = 'INSERT INTO ineuron3.Health_db(a_cno,b_Time,c_Acceler_Front,d_Acceler_Vert,e_Acceler_later,f_Id_sensor,g_RSSI,h_Phase,i_Frequency,j_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                batch = BatchStatement()
                for i in final_data:
                    final_data1 = i
                batch.add(qr1, final_data1)
                session.execute(batch)
                return render_template("index.html", response=response[0])
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)  

if __name__ == "__main__":        
    app.run()


    