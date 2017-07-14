# -*- coding: utf-8 -*-
"""
Muneeza

"""


import tensorflow as tf
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from Pre_processing import pre_proc


class patching(object):
    def __init__(self,img_data, patch_size, N_patch, file_path,file_names,high_entropy=False):
        self.img_data=img_data
        self.patch_size=patch_size
        self.N_patch=N_patch
        self.classes=[1,2,3,4,5]
        #declare variables to enforce (padding=VALID) later on
        self.row=int((self.patch_size[0]-1)/2)
        self.col=int((self.patch_size[1]-1)/2)
        
        self.file_path=file_path
        self.file_names=file_names
        
        self.high_entropy=high_entropy
      
    def unif_class(self,img_data,class_num,N_per_class):
        print('Sampling C_pixels for class_'+str(class_num))
        #Tensor containing indices of all pixels labelled as class_num dims:[len,3]
        #also enforcing padding
        class_idx=tf.where(tf.equal(img_data[:,self.row:240-self.row,self.col:240-self.col,4],class_num))
        class_idx=tf.cast(class_idx,tf.int32)
        print('created class_idx')
        N_total=len(class_idx.eval())
        miss=0
        
        if N_total==0:
            while (N_total==0):
                img_data=[]
                print('class_'+str(class_num)+' not found in given patient, selecting another')
                i=int(np.random.choice(10,1))
                print(self.file_names[i])
                img_data=pre_proc(self.file_path,self.file_names[i])
                class_idx=tf.where(tf.equal(img_data[:,self.row:240-self.row,self.col:240-self.col,4],class_num))
                class_idx=tf.cast(class_idx,tf.int32)
                N_total=len(class_idx.eval())
            
            
        if (N_total<N_per_class):
            print('not enought samples of class_'+str(class_num)+' found in given patient')    
            idx=tf.random_uniform([N_total],0,N_total,tf.int32)
            miss=N_per_class-N_total
        else:    
            #Random selection of numbers from uniform dist. pointing to the element number dims:[N_per_class]
            idx=tf.random_uniform([N_per_class],0,N_total,tf.int32)
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
            patch=img_data[p[0],p[1]:p[1]+(2*self.row)+1,p[2]:p[2]+(2*self.col)+1,:]
            label=patch[self.row,self.col,4]
            label=tf.expand_dims(label,0)
            patch=tf.expand_dims(patch[:,:,0:4],0)
            Patches=tf.concat([Patches,patch],0)
            Labels=tf.concat([Labels,label],0)
        Patches=Patches[1:N_per_class,:,:,:]
        Labels=Labels[1:N_per_class]
        print('created patches')
        return(Patches,Labels,miss)
    
      
#    def high_entropy(self,self.N_patch):    
#        #looking at the segmentation for entropy
#        ent_img_data=tf.cast(self.img_data[:,:,:,4],tf.uint8)
#        
#        slice=np.random.choice(154,1)
#        img_entropy=entropy(ent_img_data[slice[0],:,:].eval(), disk(10))
#        high_ent = np.percentile(img_entropy, 90)
#        p_idx = np.argwhere(img_entropy >= high_ent)
#        
#            
#        
#        return       
       
    
    def create_patches(self):
        N_per_class=int(self.N_patch/len(self.classes))

        if self.high_entropy==False :
            Patches,Labels,miss=self.unif_class(self.img_data,self.classes[0],N_per_class)
            
            for C in range(1,len(self.classes)):
                patch,label,miss=self.unif_class(self.img_data,self.classes[C],N_per_class)
                if miss !=0:
                    i=np.random.choice(10,1)
                    img_data=pre_proc(self.file_path,self.file_names[i])
                    print('Resampling '+str(miss[0])+' patches of class_'+str(self.classes[C])+' from another patient')
                    print(self.file_names[i])
                    patch,label,miss=self.unif_class(img_data,self.classes[C],N_per_class)
                    
                Patches=tf.concat([Patches,patch],0)
                Labels=tf.concat([Labels,label],0)

        else:
            self.sampling=self.high_entropy
            
        
        return Patches , Labels
        
