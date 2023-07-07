import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import tqdm

from formatter import formatter
import nodal 
from graphs import INModel

#train
#find loss via algorithm. (how much neural network graph deviates from truth path)
#optimize
#test

class training:
    def __init__(self, model):
        self.model = model
        
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        self.opt = snt.optimizers.Adam(learning_rate=0.01)
        
    def step(self, gr_train, gr_test=None):
        test_loss=0
        if gr_test is not None:
            pred = self.model(gr_test)
            logits = pred.edges
            labels = gr_train['labels']
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            
            test_loss = tf.reduce_mean(loss)
            
        with tf.GradientTape() as tape:
            pred = self.model(gr_train)
            #print(pred)       
            logits=pred.edges
            labels = gr_train['labels']
            #labels = [[0., 1],[1.,0.],[0.,1.],[0.,1.]] 
            
            #print(logits)
            print(labels)
            
            #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            #loss = tf.reduce_mean(loss)
            loss = tf.keras.losses.categorical_crossentropy(y_true = labels, y_pred=logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            
            logits = tf.nn.softmax(logits)
            print(logits)
            
        
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)
        
        self.stat=pd.concat([self.stat,pd.DataFrame({'train_loss':float(loss), 'test_loss':float(test_loss)}, index=[len(self.stat)])])
        print(self.stat)
        
        return loss
    
    
##################MAIN#####################

model = INModel()
t = training(model)
epochs = 100

gr_train = formatter(10, truth_value = True)
#gr_test = formatter(2, truth_value = True)

fig_s,ax_s=plt.subplots(ncols=3, figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

for epoch in tqdm.trange(epochs):
    loss=float(t.step(gr_train))#, gr_test))
    
# Plot the status of the training
ax_t.clear()
ax_t.plot(t.stat.train_loss.tolist(),label='Training')
ax_t.set_yscale('log')
ax_t.set_ylabel('loss')
ax_t.set_xlabel('epoch')
ax_t.legend()