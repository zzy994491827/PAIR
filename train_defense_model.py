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
device = "cuda" if torch.cuda.is_available() else "cpu"
test_caption_txt="./VisualSearch/flickr30ktest_add_ad/TextData/flickr30kval.caption.txt"
image_root="./VisualSearch/flickr30k/flickr30k-images"
train_txt="./VisualSearch/flickr30ktrain/TextData/train.txt"
batch_size=36
best_score=0.84
def mae_progress(model,image):
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

def cross_entropy_loss_with_logits(labels, logits):
        logp = torch.log_softmax(logits, -1)
        loss = - torch.sum(torch.multiply(labels, logp), dim=-1)
        return loss

def contrastive_loss(image_feat, cond_feat, l2_norm = True, temperature = 0.1):
        local_batch_size = image_feat.size(0)

        image_feat_large = image_feat
        cond_feat_large = cond_feat

        labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()
        logits_img2cond = torch.matmul(image_feat,
                                       cond_feat_large.permute(1, 0).contiguous()) / temperature
        logits_cond2img = torch.matmul(cond_feat,
                                       image_feat_large.permute(1, 0).contiguous()) / temperature
        loss_img2cond = cross_entropy_loss_with_logits(labels, logits_img2cond)
        loss_cond2img = cross_entropy_loss_with_logits(labels, logits_cond2img)
        loss_img2cond = torch.mean(logits_img2cond)
        loss_cond2img = torch.mean(logits_cond2img)
        loss = loss_img2cond + loss_cond2img
        return loss


def train(data_loader_train,mae_model,model,optimizer_mae, preprocess):
    mse = nn.MSELoss()
    cos=  nn.CosineEmbeddingLoss()
    for j,(images) in enumerate(data_loader_train):
        optimizer_mae.zero_grad()

        images_reconstruction,loss_reconstruction=mae_progress(mae_model,images.cuda())
        image_features = model.encode_image(images.cuda())
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        image_features_re=model.encode_image(images_reconstruction)
        image_features_re = image_features_re / image_features_re.norm(dim=1, keepdim=True)

        """
        loss_mse=mse(image_features_re,image_features)
        loss=loss_mse+loss_reconstruction
        loss=loss_cos+loss_reconstruction
        """
        loss_cos=cos(image_features_re,image_features,torch.tensor(1).cuda())
        loss=loss_reconstruction+loss_cos
        loss.backward()
        optimizer_mae.step()

        if int(j%100)==0:
            test(mae_model,model,preprocess,test_caption_txt,image_root)
            

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
    texts = clip.tokenize(texts).to(device)
    texts_features = model.encode_text(texts)


    images=[]
    for j in range(len(image_files)):
        img_path = image_root+"/"+image_files[j]
        image = Image.open(img_path).convert('RGB') 
        image = preprocess(image)
        images.append(image)

    images=torch.stack(images).cuda()
    with torch.no_grad():
        images_reconstruction,loss_reconstruction=mae_progress(mae_model,images)
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
        print(score)
        global best_score
        if score>best_score:
            best_score=score
            print("best_score:",best_score)
            torch.save(mae_model.state_dict(),"~/"+"best_model_cos"+".pth")
        return score

def main():
    mae_model=mae_vit_large_patch16_dec512d8b().cuda()
    chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = mae_model.load_state_dict(checkpoint['model'], strict=False)
    optimizer_mae = optim.Adam([
        {'params': mae_model.parameters(),'lr': 1e-6},
    ], lr = 1e-6)
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    for p in model.parameters():
        p.requires_grad = False
    train_dataset=image_dataset_train(image_root,train_txt,preprocess)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )



    for i in range(100):
        print("第"+str(i)+"epoch:")
        train(data_loader_train,mae_model,model,optimizer_mae,preprocess)
        print(test(mae_model,model,preprocess,test_caption_txt,image_root))


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    main()
