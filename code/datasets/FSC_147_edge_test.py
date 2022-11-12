import os
import sys
import numpy as np
from torch.utils import data
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import sparse
import scipy
import torch.nn.functional as F
import math
import cv2
from tools.progress.bar import Bar
import pickle
import json
import pdb
import torch.nn as nn
from torch.autograd import Variable
import math
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from vision import save_keypoints_img,save_img_tensor,save_keypoints_img_with_bbox,save_keypoints_img_with_bbox_and_indexmap,save_img_with_bbox

def SFC_147_sort_img(filename):
    return int(filename.split('.')[0])

class SFC_147_dataloader(data.Dataset):
    def __init__(self,split,opt):
        self.split=split
        self.opt=opt
        self.mean=opt.mean_std[0]
        self.std=opt.mean_std[1]

        self.img_root=os.path.join(opt.dataroot_FSC_147,'images_384_VarV2')
        self.edge_root=opt.dataroot_FSC_147_edge
        self.anno_file=os.path.join(opt.dataroot_FSC_147,'annotation_FSC147_384.json')
        self.data_split_file=os.path.join(opt.dataroot_FSC_147,'Train_Test_Val_FSC_147.json')
        self.den_path=os.path.join(opt.dataroot_FSC_147,'gt_density_map_adaptive_384_VarV2')

        with open(self.anno_file) as f:
            annotations = json.load(f)

        with open(self.data_split_file) as f:
            data_split = json.load(f)

        self.im_ids=[]
        for s in self.split:
            self.im_ids.extend(data_split[s])#

        if 'debug' in self.opt.comment:
            self.im_ids=self.im_ids[:100]
        else:
            self.im_ids.sort(key=SFC_147_sort_img)


        self.points_data={}
        self.bboxes_data={}
        for im_id in self.im_ids:
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])

            rects = []
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            dots = np.array(anno['points'])[:,::-1]
            dots = dots.tolist()

            self.points_data[im_id]=dots
            self.bboxes_data[im_id]=rects

        print('SFC_147 {}'.format(len(self.im_ids)))

    def __getitem__(self,index):
        img_file=self.im_ids[index]
        img_path=os.path.join(self.img_root,img_file)
        edge_path=os.path.join(self.edge_root,img_file)
        ann_point=torch.Tensor(self.points_data[img_file])
        bbox=torch.Tensor(self.bboxes_data[img_file])
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge=cv2.imread(edge_path,0)#h w
        edge=torch.from_numpy(edge).float()
        edge=edge.div(255)

        img=img.transpose([2,0,1])# c h w
        img=torch.from_numpy(img).float()
        img=img.div(255)

        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        
        src_h,src_w=img.size(1),img.size(2)

        scale_factor=1.0

        h=int(src_h*scale_factor)
        h=h if h%self.opt.model_scale_factor==0 else h+self.opt.model_scale_factor-h%self.opt.model_scale_factor
        w=int(src_w*scale_factor)
        w=w if w%self.opt.model_scale_factor==0 else w+self.opt.model_scale_factor-w%self.opt.model_scale_factor
        size=(h,w)
        scale_factor_h=h/src_h
        scale_factor_w=w/src_w
        if scale_factor_h==1. and scale_factor_w==1.:
            pass
        else:
            img = F.interpolate(img.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
            img = img.squeeze(0)
            edge = F.interpolate(edge.unsqueeze(0).unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
            edge = edge.squeeze()

            bbox[:,0]*=scale_factor_h
            bbox[:,2]*=scale_factor_h
            bbox[:,1]*=scale_factor_w
            bbox[:,3]*=scale_factor_w

            ann_point[:,0]*=scale_factor_h
            ann_point[:,1]*=scale_factor_w

        den_path = os.path.join(self.den_path,img_file.split(".jpg")[0] + ".npy")
        den = np.load(den_path).astype('float32')
        den = torch.from_numpy(den)

        if scale_factor_h==1. and scale_factor_w==1.:
            pass
        else:
            orig_count = torch.sum(den)
            den = F.interpolate(den.unsqueeze(0).unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True).squeeze()
            new_count = torch.sum(den)
            if new_count > 0: 
                den = den * (orig_count / new_count)

        cnt=torch.sum(den)
        den*=self.opt.gt_factor

        bbox=check_bbox(bbox,h,w,self.opt.num_box,img_file,self.split)

        info={'dataset':'SFC_147','img_file':img_file,'img_path':img_path,'gt_cnt':cnt}

        return img,bbox,den,info,edge

    def __len__(self):
        return len(self.im_ids)

def check_bbox(bbox,h,w,num_box,img_file,split):
    index_list=[]
    for i in range(len(bbox)):
        y1,x1,y2,x2=bbox[i,0],bbox[i,1],bbox[i,2],bbox[i,3]
        assert y2>=y1 and x2>=x1
        if y1<0 and y2<=0 and x1<0 and x2<=0:
            continue
        if y1>=h and y2>h and x1>=w and x2>w:
            continue
        index_list.append(i)
    if len(index_list)<num_box:
        print(img_file)
        print('h: {} w: {}'.format(h,w))
        print(bbox)

        while len(index_list)<num_box:
            index_list.extend(index_list)
    if 'train' in split:
        index_list=random.sample(index_list, num_box)
    else:
        index_list=list(range(num_box))

    return bbox[index_list]




def FSC_147_edge_test(opt):
    data_test=SFC_147_dataloader(opt.test_split,opt)

    test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
        worker_init_fn=np.random.seed(opt.seed))

    return None,None, test_loader,data_test

def data_collate(data):
    img,bbox,den,info,edge= zip(*data)
    return img,bbox,den,info,edge
