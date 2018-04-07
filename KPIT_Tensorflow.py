# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:32:44 2018

@author: Sahil Basera
"""

import tensorflow as tf
from  sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("KPIT_bus.csv")


le = preprocessing.LabelEncoder()
data = pd.read_csv("KPIT_bus.csv")

#X = data.drop(['Population' , 'day' , 'time'] , 1)
X = data[['date' , 'time' , 'stop' , 'day']]
Y = data['Population']

#le= LabelEncoder()
le= preprocessing.LabelEncoder()
le.fit(["8:00" , "12:30" , "17:00"])
for col in X :
   if col == "time" :
       X[col] = le.transform(X[col])

le1 = preprocessing.LabelEncoder()
le1.fit(["Sunday" , "Monday" , "Tuesday" , "Wednesday" , "Thursday" , "Friday" , "Saturday" , "Sunday"])
for col in X :
   if col =="day":
       X[col] = le1.transform(X[col])       
       
X = SelectKBest(mutual_info_regression , k = 3).fit_transform(X , Y)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
#X = preprocessing.scale(X)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.1 , random_state = 2)




learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1] , dtype=float)
n_dim = X.shape[1]


X_tensor = tf.placeholder('float')  
Y_tensor = tf.placeholder('float')
W = tf.Variable(tf.ones([n_dim,1]))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    y = tf.matmul(X_tensor , W)
    cost = tf.reduce_mean(tf.square(y - Y_tensor))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    for epoch in range(training_epochs) :
        epoch_loss = 0
        sess.run(training_step , feed_dict={X_tensor:X_train , Y_tensor: Y_train})
        c = sess.run(cost , feed_dict={X_tensor : X_train , Y_tensor : Y_train})
        epoch_loss +=c
        cost_history = np.append(cost_history , c)
        #print("Epoch" , epoch, "out of " , training_epochs , "cost is :" , epoch_loss)
    pred_y = sess.run(y , feed_dict = {X_tensor :X_test})
    mse = tf.reduce_mean(tf.square(pred_y - X_test))
    #print("MSE: %.4f" % sess.run(mse)   
    for i in range(23) :
     print(pred_y[i])
    

"""    
    a = [i for i in range(23)]  
    plt.scatter(a , Y_test , color = 'red')
    plt.plot(a , pred_y , color = 'blue')
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.show()
"""   
  