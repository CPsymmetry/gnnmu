import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import tqdm

from formatter import formatter
from graphs import INModel
from bunch import bunch

class training:
    def __init__(self, model):
        self.model = model
        
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        self.opt = snt.optimizers.Adam(learning_rate=0.001)
        
    def step(self, gr_train, gr_test=None):
        test_loss=0
        if gr_test is not None:
            pred = self.model(gr_test)
            logits = pred.nodes
            labels = gr_train['labels']
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            
            test_loss = tf.reduce_mean(loss)
            
        with tf.GradientTape() as tape:
            pred = self.model(gr_train)     
            logits=pred.nodes
            labels = gr_train['labels']
            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            
            logits = tf.nn.softmax(logits)
            
        
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)
        
        self.stat=pd.concat([self.stat,pd.DataFrame({'train_loss':float(loss), 'test_loss':float(test_loss)}, index=[len(self.stat)])])
        print(self.stat)
        
        return loss
    
    
##################MAIN#####################

model = INModel()
t = training(model)
epochs = 10

samples = bunch.trackPerf_to_bunch('/home/kali/sim/data')

gr_train = formatter(samples)
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
