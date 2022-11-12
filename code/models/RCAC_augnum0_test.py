import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np
import cv2
import random
import torch.backends.cudnn as cudnn
import os
from tools.log import AverageMeter
from utils import AverageMeter_acc_map
from vision import save_keypoints_img,save_img_tensor,save_keypoints_img_with_bbox,save_keypoints_img_with_bbox_and_indexmap
import pdb
import time
import math
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_example_feature(feature_maps,boxes,exemplar_scales,scale_factor,feat_aug_list):
    '''
    feature_maps b*c*f_h*f_w
    bboxes [[y1, x1, y2, x2],[],[]]
    exemplar_scales []
    '''
    
    M=len(boxes)
    boxes_scaled = boxes / scale_factor
    boxes_scaled[:, :2] = torch.floor(boxes_scaled[:, :2])
    boxes_scaled[:, 2:4] = torch.ceil(boxes_scaled[:, 2:4])
    boxes_scaled[:, 2:4] = boxes_scaled[:, 2:4] + 1 # make the end indices exclusive 
    feat_h, feat_w = feature_maps.shape[-2], feature_maps.shape[-1]
    # make sure exemplars don't go out of bound
    boxes_scaled[:, :2] = torch.clamp_min(boxes_scaled[:, :2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], feat_h)
    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_w)
    box_hs = boxes_scaled[:, 2] - boxes_scaled[:, 0]
    box_ws = boxes_scaled[:, 3] - boxes_scaled[:, 1]
    max_h = math.ceil(max(box_hs))
    max_w = math.ceil(max(box_ws))

    for j in range(0,M):
        y1, x1 = int(boxes_scaled[j,0]), int(boxes_scaled[j,1])  
        y2, x2 = int(boxes_scaled[j,2]), int(boxes_scaled[j,3]) 
        #print(y1,y2,x1,x2,max_h,max_w)
        if j == 0:
            examples_features = feature_maps[:,:,y1:y2, x1:x2]
            if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                #examples_features = pad_to_size(examples_features, max_h, max_w)
                examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')
        else:
            feat = feature_maps[:,:,y1:y2, x1:x2]
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                #feat = pad_to_size(feat, max_h, max_w)
            examples_features = torch.cat((examples_features,feat),dim=0)



    aug_examples_features=torch.zeros(len(feat_aug_list),examples_features.shape[1],examples_features.shape[2],examples_features.shape[3]).cuda()
    for i in range(len(feat_aug_list)):
        for j in range(feat_aug_list[0].shape[0]):
            aug_examples_features[i]+=feat_aug_list[i][j]*examples_features[j]


    """
    Convolving example features over image features
    """
    h, w = examples_features.shape[2], examples_features.shape[3]
    # features =    F.conv2d(
    #         F.pad(feature_maps, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
    #         examples_features
    #     )

    features =    F.conv2d(
            F.pad(feature_maps, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
            aug_examples_features
        )
    # 1*num_box*h*w
    combined = features.permute([1,0,2,3])
    # num_box*1*h*w
    # computing features for scales 0.9 and 1.1 
    '''
    F.conv2d(feature_maps,conv_kernel)
    conv_kernel: out_channels,in_channels,k_h,k_w
    '''
    for scale in exemplar_scales:
        if scale==1.0:
            continue
        h1 = math.ceil(h * scale)
        w1 = math.ceil(w * scale)
        if h1 < 1: # use original size if scaled size is too small
            h1 = h
        if w1 < 1:
            w1 = w
        # examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
        # features_scaled =    F.conv2d(F.pad(feature_maps, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
        # examples_features_scaled)

        aug_examples_features_scaled = F.interpolate(aug_examples_features, size=(h1,w1),mode='bilinear')  
        features_scaled =    F.conv2d(F.pad(feature_maps, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
        aug_examples_features_scaled)

        features_scaled = features_scaled.permute([1,0,2,3])
        combined = torch.cat((combined,features_scaled),dim=1)
    # if cnter == 0:
    #     Combined = 1.0 * combined
    # else:
    #     if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
    #         combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
    #     Combined = torch.cat((Combined,combined),dim=1)
    Combined = 1.0 * combined
    return Combined#num_bbox*(num_exemplar_scales)*feat_h*feat_w
# class Resnet50(nn.Module):
#     def __init__(self):
#         super(Resnet50FPN, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         children = list(self.resnet.children()) #1/4
#         self.conv1 = nn.Sequential(*children[:4]) #1/4
#         self.conv2 = children[4]#1/4 256
#         self.conv3 = children[5]#1/8 512
#         self.conv4 = children[6]#1/16 1024


class RCAC_augnum0_test(nn.Module):
    def __init__(self, opt):
        super(RCAC_augnum0_test, self).__init__()
        self.opt=opt
        self.step_epoch=0

        resnet = models.resnet50(pretrained=True)

        features = list(resnet.children())


        self.backbone1 = nn.Sequential(*features[0:6])
        self.backbone2 = features[6]

        self.edge_backbone1=nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2,padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2,padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            )
        self.edge_backbone2=nn.Sequential(
            # nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            )

        self.den_pred = nn.Sequential(
            nn.Conv2d(opt.regressor_in_channel, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

        init_weights(self.den_pred)


        if opt.model_for_load:
            checkpoint=torch.load(opt.model_for_load)
            model_dict =  self.state_dict()
            state_dict = {k:v for k,v in checkpoint['net'].items() if k in model_dict.keys()}
            print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)



    def test_reset(self):
        if self.opt.vision_runtime:
            self.runtime=AverageMeter()
        self.mae_test=AverageMeter()
        self.mse_test=AverageMeter()

    def get_test_res(self):
        res={}
        if self.opt.vision_runtime:
            res['run_time']=self.runtime.avg
        res['mae']=self.mae_test.avg
        res['mse']=self.mse_test.avg**0.5
        return res

    def get_backbone_features(self,img,edge):
        feat_list=[]
        edge_list=[]
        batch=len(img)
        feat1=[]
        for i in range(batch):
            feat=self.backbone1(img[i].unsqueeze(0).cuda())
            feat1.append(feat)
        

        feat2=[]
        for i in range(batch):
            feat=self.backbone2(feat1[i])
            feat2.append(feat)

        feat_list.append(feat1)
        feat_list.append(feat2)

        self.feat_scale_factor=[8.0,16.0]
        return feat_list

    def get_edge_backbone_features(self,img,edge):
        edge_list=[]
        batch=len(img)
        feat_edge1=[]
        for i in range(batch):
            feat_edge=self.edge_backbone1(edge[i].unsqueeze(0).unsqueeze(0).cuda())
            feat_edge1.append(feat_edge)
        

        feat_edge2=[]
        for i in range(batch):
            feat_edge=self.edge_backbone2(feat_edge1[i])
            feat_edge2.append(feat_edge)
        edge_list.append(feat_edge1)
        edge_list.append(feat_edge2)

        self.feat_scale_factor=[8.0,16.0]
        return edge_list

    def get_matching_res(self,feat_list,edge_list,bbox):
        batch=len(feat_list[0])
        batch_example_features_list=[]
        for i in range(batch):
            example_features_list=[]
            for j in range(len(feat_list)):
                example_features=get_example_feature(feat_list[j][i],bbox[i],self.opt.exemplar_scales,self.feat_scale_factor[j],self.opt.feat_aug_list)
                example_features=torch.cat([example_features,get_example_feature(edge_list[j][i],bbox[i],self.opt.exemplar_scales,self.feat_scale_factor[j],self.opt.feat_aug_list)],dim=1)
                #num_bbox*(num_exemplar_scales)*feat_h*feat_w
                if j>0:
                    size=(example_features_list[0].shape[-2],example_features_list[0].shape[-1])
                    example_features=F.interpolate(example_features,size=size,mode=self.opt.model_interpolate_mode,align_corners = True)
                example_features_list.append(example_features)
            batch_example_features_list.append(example_features_list)

        return batch_example_features_list


    def inference(self,data,epoch,global_step,step,vision_list,batch_num_each_epoch,logger):
        batch=len(data[0])
        for i in range(batch):
            if self.opt.vision_runtime:
                t1=time.time()
            c,h,w=data[0][i].size()
            feat_list=self.get_backbone_features(data[0],data[4])
            edge_list=self.get_edge_backbone_features(data[0],data[4])
            # feat_list,edge_list=self.get_backbone_features(data[0][i].unsqueeze(0),data[4][i].unsqueeze(0))
            batch_example_features_list=self.get_matching_res(feat_list,edge_list,data[1][i].unsqueeze(0))

            matching=torch.cat(batch_example_features_list[i],dim=1)
            pred_map=self.den_pred(matching)
            if self.opt.pool=='mean':
                pred_map = torch.mean(pred_map, dim=(0),keepdim=True)
            elif self.opt.pool=='max':
                pred_map,index_map = torch.max(pred_map, dim=(0),keepdim=True)
                index_map=index_map.squeeze()

            sum_map=torch.sum(pred_map.detach().cpu())/self.opt.gt_factor
            self.mae_test.update(abs(sum_map-data[3][i]['gt_cnt']))
            self.mse_test.update((abs(sum_map-data[3][i]['gt_cnt'])**2))
            if self.opt.vision_runtime:
                self.runtime.update(time.time()-t1)

            if step in vision_list and i==0:
                video_name=os.path.basename(data[3][i]['dataset'])
                img_name=data[3][i]['img_file']
                save_root=os.path.join(self.opt.log_root_path,'test',str(epoch),data[3][i]['dataset'])

                save_filename=os.path.join(save_root,video_name+'gt_point_',img_name.split('.')[0]+'_cnt_'+str(data[3][i]['gt_cnt'])+'.'+img_name.split('.')[1])
                save_keypoints_img_with_bbox(data[0][i],data[2][i],data[1][i],self.opt, save_filename)
                save_filename=os.path.join(save_root,video_name+'out_point_',img_name.split('.')[0]+'_cnt_'+str(sum_map)+'.'+img_name.split('.')[1])
                save_keypoints_img_with_bbox(data[0][i],pred_map[i].detach().squeeze().cpu(),data[1][i],self.opt, save_filename)

                if self.opt.pool=='max' and self.opt.num_box>1:
                    save_filename=os.path.join(save_root,video_name+'out_point_',img_name.split('.')[0]+'_cnt_'+str(sum_map)+'index_map.'+img_name.split('.')[1])
                    save_keypoints_img_with_bbox_and_indexmap(data[0][i],pred_map[i].detach().squeeze().cpu(),data[1][i],index_map,self.opt.num_box-1,self.opt, save_filename)

        res=self.get_test_res()
        return res

def init_weights(init_modules):
    for m in init_modules.modules():
        # nn.init.kquerying_normal_(m, mode='fan_in', nonlinearity='relu')
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)


