import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models_mae import mae_vit_large_patch16_dec512d8b
import torch
import clip
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from dataloader_train_mae import image_dataset_train
import random
import numpy as np
from DiffJPEG_net import DiffJPEG
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_caption_txt="./VisualSearch/flickr30ktest_add_ad/TextData/flickr30ktest.caption.txt"
image_root="./VisualSearch/flickr30k/flickr30k-images"
train_txt="./VisualSearch/flickr30ktrain/TextData/train.txt"
batch_size=36
best_score=0.80

def get_gx(tensor):
    B, C, W, H = tensor.shape
    sobelx = torch.tensor([[1.0,0.0,-1.0], [2.0,0.0,-2.0], [1.0,0.0,-1.0]]).cuda().float()
    sobely = sobelx.T

    G_x = F.conv2d(tensor.view(-1,1, W, H), sobelx.view((1,1,3,3)),padding=1).view(B, C, W, H)
    G_y = F.conv2d(tensor.view(-1,1, W, H), sobely.view((1,1,3,3)),padding=1).view(B, C, W, H)

    G = torch.sqrt(torch.pow(G_x, 2)+torch.pow(G_y, 2))

    gx = (G-G.min())/(G.max()-G.min())


    return gx

def get_gradient(tensor, block_size=15, overlap=5, threshold=0):
    
    stride = block_size - overlap
    padding = []
    for v in tensor.shape[-2:]:
        left = v % stride
        if left <= overlap:
            need = overlap
        else:
            need = block_size
        pad = need - left
        padding.append(int(pad / 2))
        padding.append(pad - padding[-1])

    tensor = F.pad(tensor, tuple(padding))

    B, C, W, H = tensor.shape

    patches = tensor.unfold(2, block_size, stride).unfold(3, block_size, stride)
    mask = ((patches.sum([1,-2,-1]) / torch.prod(torch.tensor(patches.shape)[[1,-2,-1]])) >= threshold).float().unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    



    def fold(x):
        x = x * mask
        x = x.contiguous().view(B, C, -1, block_size * block_size).permute(0, 1, 3, 2)
        x = x.contiguous().view(B, C*block_size*block_size, -1)

        return F.fold(x, output_size=(H,W), kernel_size=block_size, stride=stride)

    ox = torch.ones_like(patches)
    



    r =  fold(patches) / (fold(ox)+torch.tensor([1e-2]).cuda())
    r = F.pad(r, tuple(-i for i in padding))

    r[r != r] = 0
    return r
    
def apply_gradient(img, grad, smooth_factor=2.3):
    t = 1 - grad * smooth_factor
    clipped = torch.clip(t, 0, 1)
    return torch.mul(img, clipped.detach())

def lgs(tensor, threshold=0.11, smooth_factor=6, block_size=20, overlap=5):
    return apply_gradient(tensor, get_gradient(get_gx(tensor), block_size=block_size, overlap=overlap, threshold=threshold), smooth_factor=smooth_factor)

def mae_progress(model,image):
    #image=lgs(image)
    """
    jpeg = DiffJPEG(224, 224, differentiable=True).cuda()

    quality = 50
    jpeg.set_quality(quality)
    image = jpeg(image*255.0)/255.0
    """
    loss_all=torch.tensor(0,device=image.device)
    
    for j in range(1):
        index=list(range(0,196))
        random.shuffle(index)
        count=2
        for i in range(count):
            
            if i <(count-1):
                mask_index=index[int(int(196/count)*i):int(int(196/count)*(i+1))]
            else:
                mask_index=index[int(int(196/count)*i):]
            loss, y_pre, mask = model((image.float()).cuda(), mask_index=mask_index)
            image = model.unpatchify(y_pre)
            if i==0:
                loss_all=loss+loss_all
    
    return image,loss_all

def train(data_loader_train,mae_model,model,optimizer_mae, preprocess):
    mse = nn.MSELoss()
    for j,(images) in enumerate(data_loader_train):
        optimizer_mae.zero_grad()

        images_reconstruction,loss_reconstruction=mae_progress(mae_model,images.cuda())
        image_features = model.encode_image(images.cuda())
        image_features_re=model.encode_image(images_reconstruction)


        loss_mse=mse(image_features_re,image_features)
        loss=loss_mse+loss_reconstruction
        loss.backward()
        optimizer_mae.step()

        if int(j%100)==0:
            test(mae_model,model,preprocess,test_caption_txt,image_root)
            
import csv
def data_write_csv(file_name, datas):
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for data in datas:
            writer.writerow(data)
        print("保存文件成功，处理结束")


def test(mae_model,model, preprocess,test_caption_txt,image_root):
    with open(test_caption_txt, "r") as f:
        image_files=[]
        texts=[]
        for data in f.readlines():
            data = data.strip("\n")
            space_index=data.index(" ")
            data_text=data[space_index+1:len(data)]     
            if data[space_index-1]=="0":
                data_name=data[0:space_index-2]
                image_files.append(data_name)
            texts.append(data_text)
    """
    text_jacket=[]
    text_no_jacket=[]
    for text in texts:
        if "policeman" in text:
            text_jacket.append(text)
        else:
            text_no_jacket.append(text)
    
    text_jacket = clip.tokenize(text_jacket).to(device)
    texts_features_jacket = model.encode_text(text_jacket)
    texts_features_jacket = texts_features_jacket/texts_features_jacket.norm(dim=1, keepdim=True)
    texts_features_jacket_list=texts_features_jacket.tolist()
    data_write_csv("text_jacket.tsv",texts_features_jacket_list)


    text_no_jacket = clip.tokenize(text_no_jacket).to(device)
    texts_features_no_jacket = model.encode_text(text_no_jacket)
    texts_features_no_jacket =texts_features_no_jacket/texts_features_no_jacket.norm(dim=1, keepdim=True)

    texts_features_no_jacket_list=texts_features_no_jacket.tolist()
    data_write_csv("text_no_jacket.tsv",texts_features_no_jacket_list)
    """

    texts = clip.tokenize(texts).to(device)
    texts_features = model.encode_text(texts)


    images=[]
    for j in range(len(image_files)):
        img_path = image_root+"/"+image_files[j]
        image = Image.open(img_path).convert('RGB') 
        image = preprocess(image)
        images.append(image)

    images = torch.stack(images).cuda()
    images_fea = model.encode_image(images)
    images_fea_list= images_fea.tolist()
    data_write_csv("images_fea.tsv",images_fea_list)



    #image_features_all=[]
    #image_features_re_all=[]
    with torch.no_grad():
        images_reconstruction,loss_reconstruction=mae_progress(mae_model,images)
        #images_reconstruction=GS(images)
        image_features_re=model.encode_image(images_reconstruction)
    
        image_features_re = image_features_re / image_features_re.norm(dim=1, keepdim=True)



        texts_features = texts_features / texts_features.norm(dim=1, keepdim=True)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        k=10
        count=0
        for i in range(5000):
            label=int(i/5)
            text_feature=texts_features[i]
            logits_per_image = logit_scale*image_features_re @ text_feature.t()
            a, idx = torch.sort(logits_per_image, descending=True)
            idx_k = idx[:k]
            if label in idx_k:
                count=count+1
        score=count/5000
        return score

import math, numbers, pdb
from torch.nn import functional as F
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    function implemented by Adrian Sahlman https://tinyurl.com/y2w8ktp5
    """
    def __init__(self, channels, radius=5, sigma=0.0, dim=2):
        super(GaussianSmoothing, self).__init__()

        #self.sigema=15*torch.rand(1)
        #self.radius=int(7*torch.rand(1))+1


        self.sigema=5
        self.radius=3
        kernel=torch.tensor(self.template(),dtype=torch.float)
        
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
    def calc(self,x,y):
        res1=1/(2*3.1415926*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
 
    #滤波模板
    def template(self):
        sideLength=int(self.radius*2+1)

        result=np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0,sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all= result.sum()
        return result/all

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=(int(self.radius),int(self.radius)))

GS=GaussianSmoothing(channels=3).cuda()
def main():
    mae_model=mae_vit_large_patch16_dec512d8b().cuda()
    chkpt_dir="./best_model_cos.pth"
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = mae_model.load_state_dict(checkpoint, strict=False)
    optimizer_mae = optim.Adam([
        {'params': mae_model.parameters(),'lr': 1e-6},
    ], lr = 1e-6)
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    for p in model.parameters():
        p.requires_grad = False
    resume_file="./VisualSearch/flickr30k/CLIP-flickr.tar"
    checkpoint = torch.load(resume_file, map_location='cpu')
    g_dict=model.state_dict()
    for k in model.state_dict():
        if "clip_model.ClipModel."+k in checkpoint["model"].keys():
            g_dict[k]=checkpoint["model"]["clip_model.ClipModel."+k]

    #msg=model.load_state_dict(g_dict)
    print(test(mae_model,model,preprocess,test_caption_txt,image_root))


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    main()
