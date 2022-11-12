# -*- coding: utf-8
import time
import os
import torch
import sys
import cv2
import warnings
import numpy as np
import torch

class global_variables(object):
    num_gpu=1
    local_rank=0
    iter_each_train_epoch=0
    # iter_each_test_epoch=0
    #global variables
    seed=1234
    
    time_=time.localtime()
    time_=time.strftime('%Y-%m-%d-%H-%M',time_)
    log_root_path=''
    log_txt_path=''
    
    '''
    data process
    '''
    comment=''
    dataset='FSC_147'
    dataroot_FSC_147=''
    dataroot_FSC_147_edge=''
    train_split=['train']
    test_split=['val']
    use_flexible_gt=True
    feat_aug_list=[]

    img_interpolate_mode='bilinear'
    mean_std=([0.485,0.456,0.406], [0.229,0.224,0.225])
    gt_factor=1.0
    num_box=3

    num_workers=16
    train_batch_size   = 1
    test_batch_size   = 1

    #model
    model=''
    model_for_load=''
    model_scale_factor=8
    model_interpolate_mode='bilinear'
    exemplar_scales=[0.9,1.0,1.1]
    regressor_in_channel=12
    pool='mean'#'max'
    # vision
    vision_runtime=True
    save_start_epochs=0
    vision_each_epoch=0
    vision_frequency=10   

    def __init__(self, **kwself):
        for k, v in kwself.items():
            if k=='--local_rank':
                k='local_rank'
            if not hasattr(self, k):
                print("Warning: opt has not attribut {}".format(k))
                import pdb
                pdb.set_trace()
                self.__dict__.update({k: v})
            tp = eval('type(self.{0})'.format(k))
            if tp == type(''):
                setattr(self, k, tp(v))
            elif tp == type([]):
                tp=eval('type(self.{0}[0])'.format(k))
                if tp==type('1'):
                    v=v[1:-1].split(',')
                    setattr(self, k, v)
                else:
                    setattr(self, k, eval(v))
            else:
                setattr(self, k, eval(v))

        if self.comment:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}_{}'.format(self.time_,self.model,self.comment))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}_{}.txt'.format(self.time_,self.model,self.comment))
        else:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}'.format(self.time_,self.model))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}.txt'.format(self.time_,self.model))
        
        if self.num_gpu>1:
            self.log_root_path+='_'+str(self.local_rank)
            self.log_txt_path=self.log_txt_path.split('.txt')[0]+'_'+str(self.local_rank)+'.txt'

        tt=torch.zeros(self.num_box,self.num_box)
        for i in range(self.num_box):
            tt[i,i]=1.0
            self.feat_aug_list.append(tt[i])

        new_list=[]
        for temp in self.feat_aug_list:
            if torch.sum(temp)==0:
                continue
            new_list.append(temp/torch.sum(temp))

        self.feat_aug_list=new_list



        assert self.test_batch_size==1
        if self.num_gpu>1:
            if not os.path.exists(self.log_root_path) and self.local_rank==0:
                os.makedirs(self.log_root_path)
        elif not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

if __name__ == '__main__':
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    opt = global_variables(**option)
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(opt, k))
