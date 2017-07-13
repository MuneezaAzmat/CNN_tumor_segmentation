# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:43:32 2017

@author: Muneeza
"""

import tensorflow as tf
from pre_processing import pre_proc 
from Patch_lib import patching
import nibabel as nib
import numpy as np

file_path='.../Brain_seg/Training_data'   
file_name=open('..p/file_names.txt','r').readlines()
img_batch=(len(file_name))
file_name=[file_name[i].rstrip() for i in range(img_batch)]
#global img_batch, file_path

tf.InteractiveSession()

img_data=pre_proc(file_path,file_name[0])
Patch=patching(img_data,(33,33),100)
Patches,labels=Patch.create_patches()
