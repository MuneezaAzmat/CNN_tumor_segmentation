# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:43:32 2017

@author: Muneeza
"""

import tensorflow as tf
from Pre_processing import pre_proc 
from Patch_lib import patching
from Two_pathway_CNN import Conv_network


file_path='C:/Users/Muneeza/Desktop/Brain_seg/Training_data'   
file_name=open('C:/Users/Muneeza/Desktop/file_names.txt','r').readlines()
img_batch=(len(file_name))
file_name=[file_name[i].rstrip() for i in range(img_batch)]
#global img_batch, file_path

Total_patches=60000
Training_patients=20
patches_per_patient=60000/20
batch_size=100
training_iters=600
disp_step=10


weights = {
#local_1 7x7 conv, 4 inputs, 64 outputs
'w_l1': tf.Variable(tf.random_uniform([5, 5, 4, 64],-0.005,0.005)),
#local_2 3x3 conv, 64 inputs, 64 outputs
'w_l2': tf.Variable(tf.random_uniform([5, 5, 64, 64],-0.005,0.005)),
#global_1 13x13 conv, 4 inputs, 160 outputs
'w_g1': tf.Variable(tf.random_uniform([13,13,4,160],-0.005,0.005)),
#fully connected conv layer
'w_fc': tf.Variable(tf.random_uniform([21,21,224,5],-0.005,0.005))
}

biases = {
'b_l1': tf.Variable(tf.zeros([64])),
'b_l2': tf.Variable(tf.zeros([64])),
'b_g1': tf.Variable(tf.zeros([160])),
'b_fc': tf.Variable(tf.zeros([5]))
#        tf.log([98,0.18,1.1,0.12,0.38]
} 

X=tf.placeholder(tf.float32,[None])
Y=tf.placeholder(tf.float32,[None])
dropout=tf.placeholder(tf.float32)

# Construct model
Pred=Conv_network(X,weights,biases,dropout)
# Define loss and optimizer
Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred, labels=Y))
optimizer = tf.train.MomentumOptimizer(0.005,0.5).minimize(Loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(Pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    img_data=pre_proc(file_path,file_name[0])
    Patch=patching(img_data,(33,33),patches_per_patient,file_path,file_name)
    Patches,Labels=Patch.create_patches()

    for i in range(1,Training_patients):
        img_data=pre_proc(file_path,file_name[i])
        Patch=patching(img_data,(33,33),patches_per_patient,file_path,file_name)
        x,y=Patch.create_patches()
        Patches=tf.concat([Patches,x],0)
        Labels=tf.concat([Labels.y],0)
    
    
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: Patches, Y: Labels, dropout:0.75})
        if step % disp_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([Loss, accuracy], feed_dict={X: Patches, Y: Labels,dropout: 1})
            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.6f}".format(acc))
        
        step=step+1
    
    print('Optimization Finished!')
