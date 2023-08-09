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
import analysis

class training:
    def __init__(self, model):
        self.model = model
        #training parameters
        self.settings = {
        'drop':.985,             #drop of learning rate
        'epoch_drop':2,         #number of epochs that pass before learning rate is dropped
        'initial_rate':.0105,   #initial learning rate
        }
        self.rate = []
        self.logits = []
        self.labels = []
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        #Adam optimizer
        self.opt = snt.optimizers.Adam(learning_rate=self.settings['initial_rate'])
        #self.opt.beta1 = self.settings['beta1']
        #self.opt.beta2 = self.settings['beta2']
        
        
    def step(self, epoch, gr_train, gr_test=None):
        """
        Parameters
        ----------
        epoch : int
            The current epoch
        gr_train : {'dgraphs':[],'labels':[],'systems':[]}
            training set to be inputted into the gnn model
        gr_test : {'dgraphs':[],'labels':[],'systems':[]}, optional
            test set
        Returns
        -------
        loss : array
            losses
        """
        test_loss=0
        self.opt.learning_rate=self.step_loss(epoch)
        if gr_test is not None:
            pred = self.model(gr_test)
            logits = pred.globals
            labels = gr_train['labels']
            test_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            test_loss = tf.reduce_mean(test_loss)
            
            logits = tf.nn.softmax(logits)
            self.test_logits = logits
            
        with tf.GradientTape() as tape:
            pred = self.model(gr_train)     
            logits=pred.globals
            labels = gr_train['labels']
            self.labels=labels
            
            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            
            logits = tf.nn.softmax(logits)
            self.logits = logits
            
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)
        
        self.stat=pd.concat([self.stat,pd.DataFrame({'train_loss':float(loss), 'test_loss':float(test_loss)}, index=[len(self.stat)])])
        print(self.stat)
        
        return loss
    
    def step_loss(self, epoch):
        """
        Reduces the optimizers learning rate as epochs increases
        """
        loss=self.settings['initial_rate']*np.power(self.settings['drop'],np.floor((1+epoch)/self.settings['epoch_drop']))
        self.rate.append(loss)
        return loss



##################MAIN#####################

model = INModel()
t = training(model)
epochs = 150

#samples to be formatted
samples = bunch.assign_bunches('/home/kali/sim/data',max_events=2, track_perf=True)

print(samples)

training_samples = [samples[0]]
testing_samples = [samples[1]]

#training and testing sets are set up
gr_train = formatter(training_samples)
gr_test =  formatter(testing_samples)

for epoch in tqdm.trange(epochs):
    loss=float(t.step(epoch, gr_train, gr_test))

analysis.loss_over_time(t)
analysis.identification_efficiency(t)
