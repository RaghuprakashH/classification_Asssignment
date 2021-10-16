class Baggage_classify_mod:
    def __init__(self,modelname,save_modelname):
        self.modelname = modelname
        self.save_modelname = save_modelname
        
        
    
    def fit_func(modelname,x_train,y_train):
        modelname.fit(x_train,y_train)
        return modelname
        
    def predict_func(model,x_test):        
        pred = model.predict(x_test)
        return pred
    
    
