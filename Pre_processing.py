# -*- coding: utf-8 -*-
"""
Auth: Muneeza
Loads all modalities for one paitent
Standardizes the data according to two options 
    per 2D slice (DEFAULT)
    per 3D image
Input:
    1-file_path : path containing training data 
    2-file_name : file/paitent name
    3-bool for standatdization method set to False for per_img_std
Output:
    img_data: 4D tensor [slices,img_row,img_col,(mod+GT)]

"""

import tensorflow as tf
import nibabel as nib


def pre_proc(file_path,file_name,std_per_slice=True,new_GT=True):

    def per_slice_std(img):
        '''
        Takes nibabel image obj and standardizes it within each 2D slice
        Input:
            nibabel image obj [rows,cols,slices]
        Output:
            tf.Tensor containing slices of standardized images [slice,row,col]
        '''
        img=tf.transpose(img.get_data(),[2,0,1])
        img=tf.expand_dims(img,3)
        img=tf.cast(img,tf.float32)
        std_img=tf.map_fn(lambda slice: tf.image.per_image_standardization(slice), img)
        print('Standardizing')
        return std_img
    
    def per_img_std(img):
        '''
        Takes nibabel image obj and standardizes it on the whole 3D image
        Input:
            nibabel image obj [rows,cols,slices]
        Output:
            tf.Tensor containing slices of standardized images [slice,row,col]
        '''
        img=tf.cast(img.get_data(),tf.float32)
        std_img=tf.image.per_image_standardization(img)
        std_img=tf.expand_dims(tf.transpose(std_img,[2,0,1]),3)
        return std_img
    
    #assign -1 to background in the ground truth 
    def GT(img_flair,img_seg):
        
        img_flair=tf.constant(img_flair.get_data())
        img_seg=tf.cast(img_seg.get_data(),tf.int32)
        bool=tf.not_equal(img_flair,0)
        Bkg=tf.constant(0,tf.int32,shape=(240,240,155))  
        Healthy=tf.constant(1,tf.int32,shape=(240,240,155))  
        New_seg=tf.cast(tf.where(bool, Healthy,Bkg),tf.int32)
        New_seg=New_seg+img_seg
        print('New ground truth created')
        return New_seg
    
    if std_per_slice==True:
        std=per_slice_std
    else :
        std=per_img_std
        
    #Load 3D images
    img_flair=nib.load(file_path+file_name+file_name+'_flair.nii.gz')
    img_t1=nib.load(file_path+file_name+file_name+'_t1.nii.gz')
    img_t1ce=nib.load(file_path+file_name+file_name+'_t1ce.nii.gz')
    img_t2=nib.load(file_path+file_name+file_name+'_t2.nii.gz')
    img_seg=nib.load(file_path+file_name+file_name+'_seg.nii.gz')
    #Standardize images 
    std_img_flair=std(img_flair)
    std_img_t1=std(img_t1)
    std_img_t1ce=std(img_t1ce)
    std_img_t2=std(img_t2)
    if new_GT==True:
        new_img_seg=GT(img_flair,img_seg)
    else:
        new_img_seg=tf.constant(img_seg.get_data())
    
    #Load groud truth as 5th channel
    new_img_seg=tf.cast(tf.transpose(new_img_seg,[2,0,1]),tf.float32)
    new_img_seg=tf.expand_dims(new_img_seg,3)
    #Concat all modalities into 4D tensor [img_slices,img_row,img_col,modalities+1(GT)]
    img_data=tf.concat([std_img_flair,std_img_t1,std_img_t1ce,std_img_t2,new_img_seg],3)
    
    return img_data
