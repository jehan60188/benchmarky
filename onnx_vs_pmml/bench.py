from nyoka import skl_to_pmml
import numpy as np
import skl2onnx
import pandas
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from skl2onnx import convert_sklearn
from sklearn.metrics import f1_score
from skl2onnx.common.data_types import FloatTensorType
import os
import timeit

import pypmml
from pypmml import Model
import onnxruntime as rt


def get_data():
    iris_df = pandas.read_csv("iris.csv")
    X = iris_df[iris_df.columns.difference(["variety"])]
    y = iris_df["variety"]
    features = X.columns
    initial_types = X[:1].astype(np.float32).values
    return X, y, features, initial_types

def get_model():
    _model = RandomForestClassifier
    specs = {'n_estimators': 100, 'random_state': 0}
    return _model, specs
def get_pipe(_model, specs):
    pipe =  Pipeline([('model', _model(**specs))])
    return pipe




def make_pmml():
    skl_to_pmml(pipeline = pipe, col_names=features, target_name = "species", pmml_f_name = "rf.pmml")

def make_onnx():
    onx = skl2onnx.to_onnx(pipe, initial_types)
    with open('rf.onnx', 'wb') as f:
        f.write(onx.SerializeToString())



def main():
    fname_p = 'rf.pmml'
    fname_o = 'rf.onnx'

    X, y, features, initial_types = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = .8, random_state=0)

    _model, specs = get_model()
    pipe = get_pipe(_model, specs)
    pipe.fit(X_train, y_train)
    #file size
    p_size, o_size = os.path.getsize(fname_p), os.path.getsize(fname_o)
    #time writing
    p_time = timeit.timeit(lambda: make_pmml, number = 25)
    o_time = timeit.timeit(lambda: make_onnx, number = 25)    
    #time loading
    load_p = timeit.Timer(lambda: Model.load(fname_p))
    load_p = load_p.timeit(number = 25)
    load_o = timeit.Timer(lambda: rt.InferenceSession(fname_o))
    load_o = load_o.timeit(number = 25)
    

    #time prediction
    model_p = Model.load(fname_p)
    predi_p = timeit.timeit(lambda: model_p.predict(X_test), number = 25)
       
    sess = rt.InferenceSession(fname_o)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    predi_o = timeit.timeit(lambda: sess.run([label_name], {input_name: X_test.astype(np.float32).values})[0], number = 25)
    
    #prediction difference
    dd = {'Virginica':0, 'Versicolor':1, 'Setosa':2}     
    _base = pipe.predict(X_test)
    pp = model_p.predict(X_test)['predicted_species']
    oo = sess.run([label_name], {input_name: X_test.astype(np.float32).values})[0]
    '''
    _base = [dd[x] for x in _base]
    pp = [dd[x] for x in pp]
    oo = [dd[x] for x in oo]
    y_test = [dd[x] for x in y_test]
    '''
    err_p = f1_score(pp, y_test, average = 'micro')
    err_o = f1_score(oo, y_test, average = 'micro')
    err_base = f1_score(_base, y_test, average = 'micro')

    print('file size (p, o):', p_size, o_size)
    print('creation time (p,o):', p_time, o_time)
    print('loading time:', load_p, load_o)
    print('predict time:', predi_p, predi_o)
    print('f1 (base, p, o):', err_base, err_p, err_o)

if __name__ == "__main__":
    main()