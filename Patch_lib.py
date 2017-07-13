# -*- coding: utf-8 -*-
"""
Muneeza

"""


import tensorflow as tf


class patching(object):
    def __init__(self,img_data, patch_size, N_patch, high_entropy=False):
        self.img_data=img_data
        self.patch_size=patch_size
        self.N_patch=N_patch
        self.classes=[0,1,2,3,4]
        #declare variables to enforce (padding=VALID) later on
        self.row=int((self.patch_size[0]-1)/2)
        self.col=int((self.patch_size[1]-1)/2)
                
        if high_entropy==False :
            self.sampling=self.unif_class
        else:
            self.sampling=self.high_entropy
  
      
    def unif_class(self,class_num,N_per_class):
        print('Sampling C_pixels for class_'+str(class_num))
        #Tensor containing indices of all pixels labelled as class_num dims:[len,3]
        #also enforcing padding
        class_idx=tf.where(tf.equal(self.img_data[:,self.row:240-self.row,self.col:240-self.col,4],class_num))
        class_idx=tf.cast(class_idx,tf.int64)
        
        N_total=len(class_idx.eval())
        print('created class_idx')
        #Random selection of numbers from uniform dist. pointing to the element number dims:[N_per_class]
        idx=tf.random_uniform([N_per_class],0,N_total,tf.int64)
        
        print('created idx')
        #Pixel indices correspoinding  to the ith element of class_idx
        sample=tf.map_fn(lambda i : class_idx[i,:] ,idx)
        print('created sample')
        #Assign indices to a constant tensor so the sampled values dont change everytime the tensor
        #is called, avoids repetition and unnecessary work. 
        p_idx=tf.constant(sample.eval())
        print('created p_idx')
        #creates patches
        print('creating patches')
        Patches=tf.zeros([1,self.patch_size[0],self.patch_size[1],4])
        Labels=tf.zeros((1))
        for i in range(N_per_class):
            p=p_idx[i,:].eval()
            patch=self.img_data[p[0],p[1]:p[1]+(2*self.row)+1,p[2]:p[2]+(2*self.col)+1,:]
            label=patch[self.row,self.col,4]
            label=tf.expand_dims(label,0)
            patch=tf.expand_dims(patch[:,:,0:4],0)
            Patches=tf.concat([Patches,patch],0)
            Labels=tf.concat([Labels,label],0)
        Patches=Patches[1:N_per_class,:,:,:]
        Labels=Labels[1:N_per_class]
        print('created patches')
        return(Patches,Labels)
        
      
    def high_entropy(self):    
        return []      
       
    def create_patches(self):
        N_per_class=int(self.N_patch/len(self.classes))
        for i in range(len(self.classes)):
            Patches,Labels=self.sampling(self.classes[i],N_per_class)
        return Patches , Labels        
     
    
          
    






















