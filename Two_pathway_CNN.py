# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 23:33:48 2017

@author: Muneeza
"""


import tensorflow as tf


def Conv_network(X,weights,biases,dropout):
    '''
    Consturcts the two_pathway_CNN and returns label prediction for an input patch
    '''
    
    classes=[1,2,3,5]    
    dropout=tf.constant(dropout,tf.float32)

    def Conv_local(self,X,w,b,P):
        '''
        Convolution layer for the local path followed by activation(maxout) and pooling
        Inputs:
            1- X : Input tensor of training patches [batch,row(M),height(M),modality]
            2- K : Kernel size for convolution
            3- N : Number of feature maps for maxout (2 only for now, will need to alter maxout def if more needed)
            4- P : Pooling window size for maxpooling  
        Output:
            Feature map of size 1+((M-K+1)-P)/S)
        '''
        #Changing data type to avoid initilization error  
        X=tf.cast(X,tf.float32)
        #Two convolution filters 
        L1_out1= tf.nn.conv2d(X, w, strides=[1, 1,1, 1], padding='valid')
        L1_out1= tf.nn.bias_add(L1_out1, b)
        L1_out2= tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='valid')
        L1_out2= tf.nn.bias_add(L1_out2, b)
        
        #Maxout on the two resulting feature maps
        L1_out=tf.maximum(L1_out1,L1_out2)
        #Maxpool to get the layer output
        L1_out=tf.nn.max_pool(L1_out, ksize=[1, P, P, 1], strides=[1, 1, 1, 1],padding='valid')
        return L1_out
 
    def Conv_glob(self,X,w,b):
        '''
        Convolution layer for the Global path followed by pooling
        Inputs:
            1- X : Input tensor of training patches [batch,row,height,modality]
            2- K : Kernel size for convolution
        Output:
            Feature map of size (M-K+1)
        '''
        #Changing data type to avoid initilization error  
        X=tf.cast(X,tf.float32)
        #Two convolution filters 
        G1_out1= tf.nn.conv2d(X, w, strides=[1,1,11], padding='valid')
        G1_out1= tf.nn.bias_add(G1_out1, b)
        G1_out2= tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='valid')
        G1_out2= tf.nn.bias_add(G1_out2, b)
        #Maxout on the two resulting feature maps
        G1_out=tf.maximum(G1_out1,G1_out2)
        return G1_out

    #Local layer 1
    conv_l1 = Conv_local(X, weights['w_l1'], biases['b_l1'],4)

    #Local layer 2
    conv_l2 =Conv_local(conv_l1, weights['w_l2'], biases['b_l2'],2)
   
    #Global layer
    conv_g1 =Conv_glob(X, weights['w_g1'], biases['b_g1'])
    
    #Merge 
    Merge=tf.concat([conv_l2,conv_g1],0)
    
    # Apply Dropout
    Merge = tf.nn.dropout(Merge, dropout)
    
    #Fully connected layer
    FC_out= tf.nn.conv2d(Merge, weights['w_fc'], strides=[1,1,1,1], padding='valid')
    FC_out= tf.nn.bias_add(FC_out, biases['b_fc'])
    
    # Output, class prediction
    dist = tf.nn.softmax(FC_out)
    samples = tf.multinomial([tf.log(dist)], 1) # note log-prob
    predic=classes[tf.cast(samples[0][0], tf.int32)]       
    return predic
