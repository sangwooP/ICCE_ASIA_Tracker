# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.utile_tctrack.loss import select_cross_entropy_loss,IOULoss,DISCLE
from pysot.models.backbone.temporalbackbone import TemporalAlexNet

from pysot.models.utile_tctrack.utiletest import TCTtest
import matplotlib.pyplot as plt

import numpy as np

class UpdateResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpdateResNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, in_channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, out_channel, 1)
        )

    def forward(self, concat_f, u):
        response = self.update(concat_f)
        response += u
        return response

class ModelBuilder_tctrack(nn.Module):
    def __init__(self,label):
        super(ModelBuilder_tctrack, self).__init__()

        self.backbone = TemporalAlexNet().cuda()
        self.updatenet = UpdateResNet(512, 256).cuda()


        self.grader=TCTtest(cfg).cuda()

        self.cls3loss=nn.BCEWithLogitsLoss()
        self.IOULOSS=IOULoss()

    def template(self, z, x):
        with t.no_grad():
            zf, _, _ = self.backbone.init(z)
            self.zf = zf

            xf, xfeat1, xfeat2 = self.backbone.init(x)

            ppres = self.grader.conv1(self.xcorr_depthwise(xf, zf))

            self.zf0 = zf
            self.memory = ppres
            self.featset1 = xfeat1
            self.featset2 = xfeat2

    def templete_update(self, z):
        with t.no_grad():
            uf = self.backbone(z.unsqueeze(1))

            concat_f = t.cat((self.zf, uf), dim=1)
            update_f = self.updatenet(concat_f, self.zf0)

            self.zf = update_f
            

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def track(self, x):
        with t.no_grad():
            
            xf,xfeat1,xfeat2 = self.backbone.eachtest(x,self.featset1,self.featset2)  
                        
            loc,cls2,cls3,memory=self.grader(xf,self.zf,self.memory)
                        
            self.memory=memory
            self.featset1=xfeat1
            self.featset2=xfeat2
            
        return {
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcentercuda(self,mapp):

        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*(cfg.TRAIN.SEARCH_SIZE//2)
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+cfg.TRAIN.SEARCH_SIZE//2
        y=y-shap[:,2,yy,xx]+h/2+cfg.TRAIN.SEARCH_SIZE//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    

