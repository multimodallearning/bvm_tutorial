import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys


#some helper functions for visualisation (not really needed)
def labelMatrixOneHot(segmentation, label_num):
    B, H, W = segmentation.size()
    values = segmentation.view(B,1,H,W).expand(B,label_num,H,W)
    linspace = torch.linspace(0, label_num-1, label_num).long().view(1,label_num,1,1).expand(B,label_num,H,W)
    matrix = (values.float()==linspace.float()).float()
    return matrix
  
def overlaySegment(gray1,seg1,flag=False):
    H, W = seg1.squeeze().size()
    colors=torch.FloatTensor([0,0,0,199,67,66,225,140,154,78,129,170,45,170,170,240,110,38,111,163,91,235,175,86,202,255,52,162,0,183]).view(-1,3)/255.0
    segs1 = labelMatrixOneHot(seg1.unsqueeze(0),8)

    seg_color = torch.mm(segs1.view(8,-1).t(),colors[:8,:]).view(H,W,3)
    alpha = torch.clamp(1.0 - 0.5*(seg1>0).float(),0,1.0)

    overlay = (gray1*alpha).unsqueeze(2) + seg_color*(1.0-alpha).unsqueeze(2)
    if(flag):
        plt.imshow((overlay).numpy())
        plt.show()
    return overlay



def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B,C,D,H,W = img_in.size()
    affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

    img_out = F.grid_sample(img_in, meshgrid,padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)

    return img_out, seg_out



class my_ohem(torch.nn.NLLLoss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSoftmax() """                                             
                                                                                   
    def __init__(self, ratio, weights):      
        super(my_ohem, self).__init__(None, True)                                 
        self.ratio = ratio 
        self.weights = weights
                                                                                   
    def forward(self, x, y):
        if(len(x.size())==5):
            x = x.permute(0,2,3,4,1).contiguous().view(-1,x.size(1))
        if(len(x.size())==4):
            x = x.permute(0,2,3,1).contiguous().view(-1,x.size(1))
        if(len(x.size())==3):
            x = x.permute(0,2,1).contiguous().view(-1,x.size(1))
        y = y.view(-1)
        num_inst = x.size(0)                                                       
        num_hns = int(self.ratio * num_inst)                                       
        x_ = x.clone() 
        inst_losses = F.cross_entropy(x_, y,reduce=False)                                                    
        _, idxs = inst_losses.topk(num_hns)                                        
        x_hn = x.index_select(0, idxs)                                             
        y_hn = y.index_select(0, idxs)                                             
        return torch.nn.functional.nll_loss(x_hn, y_hn, weight=self.weights)


def dice_coeff(outputs, labels, max_label):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    dice = torch.FloatTensor(max_label-1).fill_(0).to(outputs.device)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = (iflat * tflat).sum()
        dice[label_num-1] = (2. * intersection) / (iflat.sum() + tflat.sum())
    return dice


class Logger(object):
    def __init__(self, resultFilePath):
        self.terminal = sys.stdout
        self.log = open(resultFilePath, "w")
        self.resultFilePath = resultFilePath

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def saveCurrentResults(self):
        self.log.close()
        self.log = open(self.resultFilePath, 'a')
