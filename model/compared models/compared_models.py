#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional,Conv1D, Flatten, Reshape,MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time


df1 = pd.read_csv('E:/我的文档/手机大数据//手机大数据文件/数据/sf训练集（扩充）230202-1.csv',header=None)
df2 = pd.read_csv('E:/我的文档/手机大数据/手机大数据文件/数据/sf训练集（扩充）230202-2.csv',header=None)
df3 = pd.read_csv('E:/我的文档/手机大数据/手机大数据文件/数据/sf测试集（扩充）230202.csv',header=None)

y1 = df1[14]
y2 = df2[14]
y3 = df3[14]

x1 = df1[[0,2,3,4,5,6,7,8,9,10,11,12,13]]
x2 = df2[[0,2,3,4,5,6,7,8,9,10,11,12,13]]
x3 = df3[[0,2,3,4,5,6,7,8,9,10,11,12,13]]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2)

x1_0 = x1.drop([0],axis = 1)
x1_0_train = x1_train.drop([0],axis = 1)
x1_0_test = x1_test.drop([0],axis = 1)
x2_0_test = x2.drop([0],axis = 1)
x3_0_test = x3.drop([0],axis = 1)

# k-means
# def kmeans_model():
#     kmeans = KMeans(n_clusters=5)
#     kmeans.fit(x3_0_test_scaled)
#     #f1_x3 = f1_score(y3, kmeans.labels_,average='weighted')
#     report = classification_report(y3, kmeans.labels_, labels=range(5), output_dict=True)
#     f1_class = []
#     for i in [0,1,2,3,4,'weighted avg']:
#         f1 = report[str(i)]['f1-score']
#         f1_class.append(f1)
#     row_str = ','.join(str(num) for num in f1_class)
#     return row_str

# kmeans_res = kmeans_model()
# print(kmeans_res)

# knn
def knn_model():
    x1_train1, x1_test1, y1_train1, y1_test1 = train_test_split(x1, y1, test_size = 0.2)
    x1_0_train1 = x1_train1.drop([0],axis = 1)
    x1_0_test1 = x1_test1.drop([0],axis = 1)
    x3_0_test1 = x3.drop([0],axis = 1)
    knn = KNeighborsClassifier(n_neighbors=3)
    start_train_time = time.time()
    knn.fit(x1_0_train1, y1_train1)
    train_time = time.time() - start_train_time
    start_test_time = time.time()
    knn.fit(x1_0_train1, y1_train1)
    test_time = time.time() - start_test_time
    #y1_0_test_predict = knn.predict(x1_0_test)
    y3_0_test_predict = knn.predict(x3_0_test1)
    
    report = classification_report(y3, y3_0_test_predict, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)
    y3_mt = y3.values.reshape(y3.size,1)
    x3_mt = x3.values
    y3_0_test_predict_mt = y3_0_test_predict.reshape(y3_0_test_predict.size,1)
    output_mt = np.concatenate((x3_mt,y3_0_test_predict_mt,y3_mt),axis=1)
    output_df = pd.DataFrame(output_mt)
    # output_df.to_csv('output/knn_output.csv', index=False, header=False)
    return row_str, train_time, test_time

knn_res, train_time, test_time = knn_model()
print(knn_res)
print(train_time)
print(test_time)

def rf_model():
    clf = RandomForestClassifier(max_depth=10, n_estimators=200,max_features=10)
    start_train_time = time.time()
    clf_model = clf.fit(x1_0_train, y1_train)
    train_time = time.time() - start_train_time
    
    start_test_time = time.time() 
    y3_0_test_predict = clf_model.predict(x3_0_test)
    test_time = time.time() - start_test_time
#     f1_x3 = f1_score(y3, y3_0_test_predict,average='weighted')
    report = classification_report(y3, y3_0_test_predict, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)
    y3_mt = y3.values.reshape(y3.size,1)
    x3_mt = x3.values
    y3_0_test_predict_mt = y3_0_test_predict.reshape(y3_0_test_predict.size,1)
    output_mt = np.concatenate((x3_mt,y3_0_test_predict_mt,y3_mt),axis=1)
    output_df = pd.DataFrame(output_mt)
    # output_df.to_csv('output/rf_output.csv', index=False, header=False)
    return row_str, train_time, test_time

rf_res, train_time, test_time = rf_model()
print(rf_res)
print(train_time)
print(test_time)

def xgb_model():
    x1_train1, x1_test1, y1_train1, y1_test1 = train_test_split(x1, y1, test_size = 0.2)
    x1_0_train1 = x1_train1.drop([0],axis = 1)
    x1_0_test1 = x1_test1.drop([0],axis = 1)
    x3_0_test1 = x3.drop([0],axis = 1)
    
    model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.01,subsample=0.8,colsample_bytree=0.8, )
    start_train_time = time.time()
    model.fit(x1_0_train1, y1_train1)
    train_time = time.time() - start_train_time
  
    start_test_time = time.time()
    y3_0_test_predict = model.predict(x3_0_test1)
    test_time = time.time() - start_test_time
    
    report = classification_report(y3, y3_0_test_predict, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)
        
    y3_mt = y3.values.reshape(y3.size,1)
    x3_mt = x3.values
    y3_0_test_predict_mt = y3_0_test_predict.reshape(y3_0_test_predict.size,1)
    output_mt = np.concatenate((x3_mt,y3_0_test_predict_mt,y3_mt),axis=1)
    output_df = pd.DataFrame(output_mt)
    # output_df.to_csv('output/xgb_output.csv', index=False, header=False)
    return row_str, train_time, test_time

xgb_res, train_time, test_time = xgb_model()
print(xgb_res)
print(train_time)
print(test_time)

def create_dataset(X, look_back):
    lenth = X.shape[1]-1
    mat = X[0,1:]
    fillmat = np.full([look_back - 1, lenth],0)
    dataX = np.vstack((fillmat,mat))
    premdn = X[0,0]
    i = 1
    count = 1
    while i < len(X):
        if (X[i][0] == premdn):
            if(count < look_back):
                temp = X[i-count:i+1, 1:]
                filltemp = fill_zeros(temp,look_back,lenth)
                dataX = np.vstack((dataX,filltemp))
            else:
                dataX = np.vstack((dataX, X[i-look_back+1:i + 1, 1:]))
            i += 1
            count += 1
        else:
            mat = X[i, 1:]
            fillmat = np.full([look_back - 1, lenth],0)
            fillmat = np.vstack((fillmat,mat))
            dataX = np.vstack((dataX,fillmat))
            premdn = X[i, 0]
            count = 1
            i += 1
    return np.array(dataX)

def fill_zeros(mat,look_back,lenth):
    if mat.shape[0] < look_back:
        temp = np.full([look_back-mat.shape[0],lenth],0)
        temp = np.vstack((temp,mat))
        return temp
    else:
        return mat

def lstm_model():
    y1_5 = to_categorical(y1, num_classes=5).reshape(-1,5)
    y3_5 = to_categorical(y3, num_classes=5).reshape(-1,5)
 
    look_back = 3
    trainX = create_dataset(np.array(x1_0),look_back)
    trainy = np.array(y1_5)
    testX = create_dataset(np.array(x3_0_test), look_back)
    testy = np.array(y3_5)
    
    train_n1 = int(trainX.shape[0]/look_back)
    test_n1 = int(testX.shape[0]/look_back)
    trainX = trainX.reshape((train_n1, look_back,trainX.shape[1])).astype('float32')
    testX = testX.reshape((test_n1, look_back,testX.shape[1])).astype('float32')
    trainy = trainy.reshape((train_n1, 5)).astype('float32')
    testy = testy.reshape((test_n1, 5)).astype('float32')
    
    model = Sequential()
    model.add(LSTM(32,input_shape=(trainX.shape[1],trainX.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    start_train_time = time.time()
    model.fit(trainX, trainy, epochs=50, batch_size=72, verbose=0)
    train_time = time.time() - start_train_time
    
    start_test_time = time.time()
    pred = model.predict(testX)
    test_time = time.time() - start_test_time
    
    yhat = np.argmax(pred, axis=1)
    y = np.argmax(testy, axis=1)
    
    report = classification_report(y, yhat, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)
    
    x3_mt = x3.values
    output_mt = np.concatenate((x3_mt,yhat.reshape(yhat.size,1),y.reshape(y.size,1)),axis = 1)
    output_df = pd.DataFrame(output_mt)
    #output_df.to_csv('output/lstm_output.csv', index=False, header=False)
    return row_str, train_time, test_time

lstm_res, train_time, test_time = lstm_model()
print(lstm_res)
print(train_time)
print(test_time)

def cnn_bilstm_model():
    y1_5 = to_categorical(y1, num_classes=5).reshape(-1,5)
    y3_5 = to_categorical(y3, num_classes=5).reshape(-1,5)  

    look_back = 3
    trainX = create_dataset(np.array(x1_0),look_back)
    trainy = np.array(y1_5)
    testX = create_dataset(np.array(x3_0_test), look_back)
    testy = np.array(y3_5)

    train_n1 = int(trainX.shape[0]/look_back)
    test_n1 = int(testX.shape[0]/look_back)
    trainX = trainX.reshape((train_n1, look_back,trainX.shape[1])).astype('float32')
    testX = testX.reshape((test_n1, look_back,testX.shape[1])).astype('float32')
    trainy = trainy.reshape((train_n1, 5)).astype('float32')
    testy = testy.reshape((test_n1, 5)).astype('float32')
    
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Reshape((-1, 16)),
        Bidirectional(LSTM(32, activation='relu')),
        Dropout(0.3),
        Dense(5, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start_train_time = time.time()
    model.fit(trainX, trainy, epochs=50, batch_size=72, verbose=0)
    train_time = time.time() - start_train_time
    start_test_time = time.time()
    pred = model.predict(testX)
    test_time = time.time() - start_test_time
    
    yhat = np.argmax(pred, axis=1)
    y = np.argmax(testy, axis=1)
    
    report = classification_report(y, yhat, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)

    x3_mt = x3.values
    output_mt = np.concatenate((x3_mt,yhat.reshape(yhat.size,1),y.reshape(y.size,1)),axis = 1)
    output_df = pd.DataFrame(output_mt)
    output_df.to_csv('output/cnn_bilstm_output.csv', index=False, header=False)
    return row_str, train_time, test_time


cnn_bilstm_res, train_time, test_time = cnn_bilstm_model()
print(cnn_bilstm_res)
print(train_time)
print(test_time)

def bilstm_model():
    y1_5 = to_categorical(y1, num_classes=5).reshape(-1,5)
    y3_5 = to_categorical(y3, num_classes=5).reshape(-1,5)

    look_back = 3
    trainX = create_dataset(np.array(x1_0),look_back)
    trainy = np.array(y1_5)
    testX = create_dataset(np.array(x3_0_test), look_back)
    testy = np.array(y3_5)
    
    train_n1 = int(trainX.shape[0]/look_back)
    test_n1 = int(testX.shape[0]/look_back)
    trainX = trainX.reshape((train_n1, look_back,trainX.shape[1])).astype('float32')
    testX = testX.reshape((test_n1, look_back,testX.shape[1])).astype('float32')
    trainy = trainy.reshape((train_n1, 5)).astype('float32')
    testy = testy.reshape((test_n1, 5)).astype('float32')
    
    model = Sequential()
    model.add(Bidirectional(LSTM(32,input_shape=(trainX.shape[1],trainX.shape[2]))))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    start_train_time = time.time()
    model.fit(trainX, trainy, epochs=50, batch_size=72, verbose=0)
    train_time = time.time() - start_train_time
    
    start_test_time = time.time()
    pred = model.predict(testX)
    test_time = time.time() - start_test_time
    
    yhat = np.argmax(pred, axis=1)
    y = np.argmax(testy, axis=1)
    
    report = classification_report(y, yhat, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)

    x3_mt = x3.values
    output_mt = np.concatenate((x3_mt,yhat.reshape(yhat.size,1),y.reshape(y.size,1)),axis = 1)
    output_df = pd.DataFrame(output_mt)
    #output_df.to_csv('output/bilstm_output.csv', index=False, header=False)
    return row_str, train_time, test_time

bilstm_res, train_time, test_time = bilstm_model()
print(bilstm_res)
print(train_time)
print(test_time)

def hybrid_model():
    #rfe
    clf = RandomForestClassifier(max_depth=9, n_estimators=200,max_features=10)
    start_train_time = time.time()
    clf_model = clf.fit(x1_0, y1)
    testy2pred_prob = clf_model.predict_proba(x2_0_test)
    train_time = time.time() - start_train_time
    testy2pred = clf_model.predict(x2_0_test)

    start_test_time = time.time()
    testy3pred_prob = clf_model.predict_proba(x3_0_test)
    test_time = time.time() - start_test_time
    testy3pred = clf_model.predict(x3_0_test)
    
    testy2_mt = y2.values.reshape(y2.size,1)
    testx20_mt = x2.values
    testy2pred_mt = testy2pred.reshape(testy2pred.size,1)

    testy3_mt = y3.values.reshape(y3.size,1)
    testx30_mt = x3.values
    testy3pred_mt = testy3pred.reshape(testy3pred.size,1)

    result2_mt = np.concatenate((testx20_mt,testy2pred_prob,testy2pred_mt,testy2_mt),axis = 1)
    result3_mt = np.concatenate((testx30_mt,testy3pred_prob,testy3pred_mt,testy3_mt),axis = 1)

    y2_5 = to_categorical(result2_mt[:,19], num_classes=5).reshape(-1,5)
    y3_5 = to_categorical(result3_mt[:,19], num_classes=5).reshape(-1,5)

    x2_mt = result2_mt[:, [0,13,14,15,16,17]]
    x3_mt = result3_mt[:, [0,13,14,15,16,17]]

#   windows
    look_back = 3

    trainX = create_dataset(x2_mt,look_back)
#    trainX = np.array(x2_0)
    trainy = np.array(y2_5)
    testX = create_dataset(x3_mt, look_back)
    testy = np.array(y3_5)
    #testX = np.array(x3_0_test)

    train_n1 = int(trainX.shape[0]/look_back)
    test_n1 = int(testX.shape[0]/look_back)
    trainX = trainX.reshape((train_n1, look_back,trainX.shape[1])).astype('float32')
    testX = testX.reshape((test_n1, look_back,testX.shape[1])).astype('float32')
    trainy = trainy.reshape((train_n1, 5)).astype('float32')
    testy = testy.reshape((test_n1, 5)).astype('float32')

    model = Sequential()
    model.add(Bidirectional(LSTM(32,input_shape=(trainX.shape[1],trainX.shape[2]))))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    start_train_time = time.time()
    model.fit(trainX, trainy, epochs=50, batch_size=72, verbose=0)
    train_time = time.time() - start_train_time + train_time
    
    start_test_time = time.time()
    pred = model.predict(testX)
    test_time = time.time() - start_test_time + test_time
    yhat = np.argmax(pred, axis=1)
    y = np.argmax(testy, axis=1)

    report = classification_report(y, yhat, labels=range(5), output_dict=True)
    f1_class = []
    for i in [0,1,2,3,4,'weighted avg']:
        f1 = report[str(i)]['f1-score']
        f1_class.append(f1)
    row_str = ','.join(str(num) for num in f1_class)
    
    output_mt = np.concatenate((testx30_mt,yhat.reshape(yhat.size,1),y.reshape(y.size,1)),axis = 1)
    output_df = pd.DataFrame(output_mt)
   # output_df.to_csv('output/hybrid_output.csv', index=False, header=False)

    return row_str, train_time, test_time


hybrid_res, train_time, test_time = hybrid_model()
print(hybrid_res)
print(train_time)
print(test_time)


# file = open('output/compared100.txt', 'w')
# label = ','.join(['knn_0', 'knn_1','knn_2','knn_3','knn_4','knn_avg','rf_0', 'rf_1','rf_2','rf_3','rf_4','rf_avg',
#                   'xgb_0','xgb_1', 'xgb_2','xgb_3','xgb_4','xgb_avg','lstm_0', 'lstm_1','lstm_2','lstm_3','lstm_4','lstm_avg',
#                  'bilstm_0', 'bilstm_1', 'bilstm_2', 'bilstm_3', 'bilstm_4', 'bilstm_avg',
#                  'cnn_bilstm_0', 'cnn_bilstm_1', 'cnn_bilstm_2', 'cnn_bilstm_3', 'cnn_bilstm_4', 'cnn_bilstm_avg',
#                   'hybrid_0','hybrid_1','hybrid_2','hybrid_3','hybrid_4','hybrid_avg'])
# file.write(label + '\n')
# for i in range(100):
# #     kmeans_res,_,_ = kmeans_model()
#     knn_res,_,_ = knn_model()
#     rf_res,_,_ = rf_model()
#     xgb_res,_,_ = xgb_model()
#     lstm_res,_,_ = lstm_model()
#     bilstm_res,_,_ = bilstm_model()
#     cnn_bilstm_res,_,_ = cnn_bilstm_model()
#     hybrid_res,_,_ = hybrid_model()
#     result = ','.join([knn_res, rf_res, xgb_res, lstm_res, cnn_bilstm_res, bilstm_res, hybrid_res])
#     print('finish step',i)
#     file.write(result + '\n')
# file.close()
