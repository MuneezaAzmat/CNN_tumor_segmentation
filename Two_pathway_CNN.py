# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 23:33:48 2017

@author: Muneeza
"""


import tensorflow as tf
from tensorflow.contrib.layers import conv2d


l1l2=tf.contrib.keras.regularizers.L1L2
init_weight=tf.contrib.keras.initializers.RandomUniform
init_bias=tf.contrib.keras.initializers.Zeros


class Two_pathway_CNN(object):
    def __init__(self,batch_size=200,n_epochs=10,n_modalities=4,pre_trained='false'):
        self.epochs=n_epochs
        self.channels=n_modalities
        self.batch_size=batch_size
        if self.pre_trained:
            path_trained_net=str(input('enter path to pre-trained network'))
            self.learned_net= self.load_pre_trained_net(path_trained_net)
        else:
            self.learned_net=self.train_net()
    

    def Conv_local(self,X,K,N,P):
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
        L1_out1=conv2d(X, 64*self.channels, K, padding='valid', activation_fn=None, weights_initializer=init_weight(-0.005,0.005), weights_regularizer=l1l2(l1=0.01, l2=0.01),biases_initializer=init_bias)
        L1_out2=conv2d(X, 64*self.channels, K, padding='valid', activation_fn=None, weights_initializer=init_weight(-0.005,0.005), weight_regularizer=l1l2(l1=0.01, l2=0.01),biases_initializer=init_bias)
        #Maxout on the two resulting feature maps
        L1_out=tf.maximum(L1_out1,L1_out2)
        #Maxpool to get the layer output
        L1_out=tf.nn.max_pool(L1_out, ksize=[1, P, P, 1], strides=[1, 1, 1, 1],padding='valid')
        return L1_out
 
    def Conv_glob(self,X,K):
        '''
        Convolution layer for the Globas path followed by pooling
        Inputs:
            1- X : Input tensor of training patches [batch,row,height,modality]
            2- K : Kernel size for convolution
        Output:
            Feature map of size (M-K+1)
        '''
        #Changing data type to avoid initilization error  
        X=tf.cast(X,tf.float32)
        #Two convolution filters 
        G1_out1=conv2d(X, 64*self.channels, K, padding='valid', activation_fn=None, weights_initializer=init_weight(-0.005,0.005), weights_regularizer=l1l2(l1=0.01, l2=0.01),biases_initializer=init_bias)
        G1_out2=conv2d(X, 64*self.channels, K, padding='valid', activation_fn=None, weights_initializer=init_weight(-0.005,0.005), weight_regularizer=l1l2(l1=0.01, l2=0.01),biases_initializer=init_bias)
        #Maxout on the two resulting feature maps
        G1_out=tf.maximum(G1_out1,G1_out2)
        return G1_out
     
        
    def Create_net(self):
        '''
        Consturcts the two_pathway_CNN
        '''
    
    
    
    
    def train_net(self,X,Y):
        '''
        Trains the network
        '''
        # Construct model
        Out= Create_net(X)
        # Define loss and optimizer
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Out, labels=Y))
        optimizer = tf.train.MomentumOptimizer(0.005,0.5).minimize(Loss)
        
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(Out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
                    
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")
