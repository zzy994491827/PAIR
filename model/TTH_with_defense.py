import math, numbers, pdb
import numpy as np
import PIL
from torchvision.transforms import Compose, Resize, CenterCrop, TenCrop, Lambda, ToTensor, Normalize, RandomResizedCrop

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import utils as vutils
from DiffJPEG_net import DiffJPEG
resize=Resize((224,224))
def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    input_tensor=input_tensor.unsqueeze(0)
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


import csv
def data_write_csv(file_name, datas):
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for data in datas:
            writer.writerow(data)
        print("保存文件成功，处理结束")
i=0

def preprocess_clip_toTensor(shape=[224, 224], path='./VisualSearch/flickr30k/ads/exported_qrcode_image.png'):
    preprocess = Compose([
            Resize(shape),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])
    return preprocess(PIL.Image.open(path))


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
    index=list(range(0,196))
    random.shuffle(index)
    #index=[90, 94, 117, 124, 47, 114, 53, 127, 27, 192, 23, 43, 96, 46, 52, 83, 151, 178, 131, 173, 102, 76, 70, 179, 45, 168, 48, 64, 10, 71, 14, 72, 77, 190, 128, 87, 28, 140, 39, 98, 121, 2, 78, 195, 191, 18, 184, 112, 125, 129, 73, 33, 54, 103, 19, 107, 158, 156, 24, 97, 111, 187, 13, 116, 157, 7, 171, 55, 188, 175, 145, 144, 105, 67, 92, 66, 193, 126, 0, 182, 186, 174, 57, 163, 147, 108, 177, 26, 88, 41, 91, 152, 42, 194, 31, 68, 162, 167, 82, 63, 32, 37, 16, 60, 146, 65, 160, 149, 4, 183, 143, 8, 6, 139, 189, 11, 132, 85, 110, 81, 119, 109, 50, 59, 89, 113, 130, 15, 86, 136, 165, 22, 61, 25, 159, 99, 166, 40, 3, 106, 36, 17, 30, 155, 51, 69, 100, 38, 180, 170, 154, 120, 79, 80, 153, 115, 101, 138, 29, 137, 141, 148, 176, 164, 44, 95, 181, 135, 134, 133, 74, 20, 185, 75, 56, 172, 34, 1, 5, 161, 93, 9, 150, 142, 35, 122, 123, 104, 12, 58, 169, 21, 49, 118, 62, 84]
    
    
    for j in range(1):
        
        count=2
        for i in range(count):
            if i <(count-1):
                mask_index=index[int(int(196/count)*i):int(int(196/count)*(i+1))]
            else:
                mask_index=index[int(int(196/count)*i):]

            loss, y_pre, mask = model((image.float()).cuda(), mask_index=mask_index)
            image = model.unpatchify(y_pre)
    return image


# targeted mismatch attack with a patch
def tth_patch(networks, scales, target_tensor, carrier_img, mode='normal',
              num_steps=100, lr=1.0, lam=0.0, sigma_blur=0.0, verbose=True, seed=155, device=torch.device("cpu"),
              target_type='image', patch_ratio=0.2):
    # if seed is not None:   # uncomment to reproduce the results of the ICCV19 paper - still some randomness though
    #     reproduce(seed)
    target_tensor = target_tensor.to(device)

    carrier_img = carrier_img.to(device)
    patch_tensor1 = torch.rand_like(carrier_img)  # parameters to be learned is a patch
    M = torch.zeros_like(carrier_img)
    patch_length, patch_width = int(carrier_img.shape[-1]*patch_ratio), int(carrier_img.shape[-2]*patch_ratio)
    M[:, :, -patch_length:, -patch_width:] = 1

    patch_tensor1 = M.mul(patch_tensor1)
    patch_tensor = nn.Parameter(patch_tensor1.data)

    carrier_tensor = (1-M).mul(carrier_img) + M.mul(patch_tensor)
    carrier_org = carrier_img.clone()                 # to compute distortion
    optimizer = optim.Adam([patch_tensor], lr=lr,eps=1e-3)
    lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95),
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                                patience=2)]

    bin_centers_fixed = torch.DoubleTensor(np.arange(0, 1.001, 0.05)).to(device)   # for histograms only
    scales = np.array(scales)
    sigma_blur_all = sigma_blur / np.array(scales)
    kernel_size_all = 2*np.floor((np.ceil(6*sigma_blur_all)/2))+1

    # pre-compute all target global-descriptors / histograms / tensors
    targets, norm_factors = {}, {}
    for network in networks:  # optimize for all networks
        network.eval()

        m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for scale in scales:  # optimize for all scales
            si = (scales==scale).nonzero()[0].item()
            if sigma_blur > 0.0:
                GS = GaussianSmoothing(channels = 3, sigma = sigma_blur).to(device)
            else:
                GS = nn.Sequential()  # identity function

            # normalize (mean-std), re-scale, and feed to the network
            if target_type == 'image':
                xl = network.visual_features(nn.functional.interpolate((GS(target_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))
            elif target_type == 'embedding':
                xl = target_tensor
            elif target_type == 'embeddings':
                xl = [each.unsqueeze(0) for each in target_tensor]

            if not isinstance(xl, list): xl = [xl]  # to support optimization for multi targets too
            for l in range(len(xl)):
                x = xl[l]
                if mode == 'global':
                    # global descriptors
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = network.norm(x).squeeze().detach()
                elif mode == 'hist':  # activation histogram
                    nf = x.max().detach()
                    norm_factors[network.meta['architecture'], str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'],str(scale), 'layer'+str(l)] = hist_per_channel((x / nf).clamp(0,1), bin_centers_fixed).detach()
                elif mode == 'tensor':  # activation tensor
                    nf = (0.1*x.max()).detach()  # 0.1 ???
                    norm_factors[network.meta['architecture'],str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = (x / nf).detach()

    # for convergence checks
    globals()['converged'] = True
    globals()['loss_perf_min'] = 1e+9
    globals()['loss_perf_converged'] = 1e-4
    globals()['convergence_safe'] = False

    print('Optimizing..')
    itr = [0]
    while itr[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            loss_perf = torch.Tensor(1).to(device)*0.0
            n = 0  # counter for loss summands
            for network in networks:  # optimize for all networks
                network.eval()
                m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                for scale in scales:  # optimize for all scales
                    si = (scales==scale).nonzero()[0].item()
                    if sigma_blur > 0.0:
                        GS = GaussianSmoothing(channels=3,sigma=sigma_blur).to(device)
                    else:
                        GS = nn.Sequential() # identity function

                    # normalize (mean-std), re-scale, and feed to the network
                    carrier_x = network.visual_features(nn.functional.interpolate((GS(carriersummands_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))


                    if mode == 'global': # global descriptors
                        for l in range(len(targets)):
                            ref = network.norm(carrier_x).squeeze()
                            target = targets[network.meta['architecture'], str(scale), 'layer'+str(l)]
                            loss_perf += 1 - (ref).dot(target)   # add loss over networks and scales
                            n+= 1

            # compute loss
            if lam > 0:
                loss_distort = (carrier_tensor-carrier_org).pow(2.0).sum() / (carrier_tensor.size(-1)*carrier_tensor.size(-2))
            else:
                loss_distort = torch.Tensor(1).to(device)*0.0
            loss_perf = loss_perf / n  # divide by number of summands (networks, scales, poolings)
            total_loss = loss_perf + lam * loss_distort

            # check for convergence (hacky!)
            if loss_perf < globals()['loss_perf_min']:
                globals()['loss_perf_min'] = loss_perf.clone()

            if loss_perf < globals()['loss_perf_converged']:
                globals()['convergence_safe'] = True

            if globals()['converged'] and (loss_perf-globals()['loss_perf_min']) > 1*globals()['loss_perf_min'] and globals()['convergence_safe'] == False:
                globals()['converged'] = False
                print("Iter {:5d}, Loss_perf = {:6f} Loss_distort = {:6f} Loss_total = {:6f}".format(itr[0], loss_perf.item(), loss_distort.item(), total_loss.item()))
                print('Did not converge')

            total_loss.backward(retain_graph=True)

            if verbose == True and itr[0] % 5 == 0:
                print("Iter {:5d}, lr: {:3.3e} Loss_perf = {:6f} Loss_distort = {:6f}, Loss_total = {:6f}".format(
                    itr[0], optimizer.param_groups[0]['lr'], loss_perf.item(), loss_distort.item(), total_loss.item()))
            globals()['loss_perf'] = loss_perf
            globals()['loss_distort'] = loss_distort
            itr[0] += 1
            return total_loss

        if not globals()['converged']: return carrier_tensor.data, 0, 0, False

        patch_tensor.data.clamp_(0, 1)  # correct pixels values

        carrier_tensor = (1 - M).mul(carrier_tensor) + M.mul(patch_tensor)
        closure()
        optimizer.step()
        lr_schedulers[0].step()
        # lr_schedulers[1].step(globals()['loss_perf_min'])

    carrier_tensor.data.clamp_(0, 1)  # pixel value correction
    return carrier_tensor.data, globals()['loss_perf'], globals()['loss_distort'], globals()['converged']



    
def tth_patch_multi_carrier(networks, scales, target_tensor, carrier_imgs, mode='normal',
                            num_steps=100, lr=1.0, lam=0.0, sigma_blur=0.0, verbose=True, seed=155, device=torch.device("cpu"),
                            target_type='image', patch_ratio=0.2,model_mae=None):
    # if seed is not None:   # uncomment to reproduce the results of the ICCV19 paper - still some randomness though
    #     reproduce(seed)
    #sigma_blur=0.0
    target_tensor = target_tensor.to(device)

    carrier_imgs = carrier_imgs.to(device)

    


    patch_tensor_org = torch.rand_like(carrier_imgs[0])  # parameters to be learned is a patch
    M = torch.zeros_like(carrier_imgs[0])
    patch_length, patch_width = int(carrier_imgs.shape[-1] * patch_ratio), int(carrier_imgs.shape[-2] * patch_ratio)
    M[:, 0:patch_length, -patch_width:] = 1
    patch_tensor_org = M.mul(patch_tensor_org)
    patch_tensor_org[:, 0:patch_length, -patch_width:] = preprocess_clip_toTensor((patch_length, patch_width))
    patch_tensor_org = torch.where(patch_tensor_org > 0.5, 1, 0).float()
    # M = torch.where(patch_tensor_org > 0.5, 1, 0).float()  # 仅仅对黑色部分进行攻击

    
    patch_tensor = nn.Parameter(patch_tensor_org.clone().data)

    carriers_tensor = (1-M).mul(carrier_imgs) + M.mul(patch_tensor)
    carrier_org = carrier_imgs.clone()                 # to compute distortion
    optimizer = optim.Adam([patch_tensor], lr=lr)
    #optimizer=torch.optim.SGD([patch_tensor], lr=0.01)

   

    lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95),
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                                patience=2)]

    bin_centers_fixed = torch.DoubleTensor(np.arange(0, 1.001, 0.05)).to(device)   # for histograms only
    scales = np.array(scales)
    sigma_blur_all = sigma_blur / np.array(scales)
    kernel_size_all = 2*np.floor((np.ceil(6*sigma_blur_all)/2))+1

    # pre-compute all target global-descriptors / histograms / tensors
    targets, norm_factors = {}, {}
    for network in networks:  # optimize for all networks
        network.eval()

        
        
        m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for scale in scales:  # optimize for all scales
            si = (scales==scale).nonzero()[0].item()
            if sigma_blur > 0.0:
                GS = GaussianSmoothing(channels = 3,sigma = sigma_blur).to(device)
            else:
                GS = nn.Sequential()  # identity function

            # normalize (mean-std), re-scale, and feed to the network
            if target_type == 'image':
                xl = network.visual_features(mae_progress(model_mae,nn.functional.interpolate((GS(carriers_tensor) - m) / s, size=[224,224], mode='bilinear', align_corners=False)))
            elif target_type == 'embedding':
                xl = target_tensor
            elif target_type == 'embeddings':
                xl = [each.unsqueeze(0) for each in target_tensor]

            if not isinstance(xl, list): xl = [xl]  # to support optimization for multi targets too
            for l in range(len(xl)):
                x = xl[l]
                if mode == 'global':
                    # global descriptors
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = network.norm(x).squeeze().detach()
                elif mode == 'hist':  # activation histogram
                    nf = x.max().detach()
                    norm_factors[network.meta['architecture'], str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'],str(scale), 'layer'+str(l)] = hist_per_channel((x / nf).clamp(0,1), bin_centers_fixed).detach()
                elif mode == 'tensor':  # activation tensor
                    nf = (0.1*x.max()).detach()  # 0.1 ???
                    norm_factors[network.meta['architecture'],str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = (x / nf).detach()

    # for convergence checks
    globals()['converged'] = True
    globals()['loss_perf_min'] = 1e+9
    globals()['loss_perf_converged'] = 1e-4
    globals()['convergence_safe'] = False

    print('Optimizing..')
    itr = [0]
    while itr[0] <= num_steps:
        
        def closure():
            optimizer.zero_grad()
            loss_perf = torch.Tensor(1).to(device)*0.0
            n = 0  # counter for loss summands
            for network in networks:  # optimize for all networks
                network.eval()
                m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                for scale in scales:  # optimize for all scales
                    si = (scales==scale).nonzero()[0].item()
                    if sigma_blur > 0.0:
                        sig=sigma_blur*random.random()
                        s_all = sig / np.array(scales)
                        kernel_size_all = 2*np.floor((np.ceil(6*s_all)/2))+1
                        GS = GaussianSmoothing(channels=3,sigma=sig).to(device)
                    else:
                        GS = nn.Sequential() # identity function

                    # normalize (mean-std), re-scale, and feed to the network

                    
                    x = mae_progress(model_mae.eval(),nn.functional.interpolate((GS(carriers_tensor) - m) / s, size=[224,224], mode='bilinear', align_corners=False))
                        
                        #x=mae_progress(model_mae.eval(),nn.functional.interpolate((carriers_tensor - m) / s, size=[224,224], mode='bilinear', align_corners=False))
                        
                    carrier_x = network.visual_features(x)
                    

                    #carrier_x = network.visual_features(nn.functional.interpolate((GS(carriers_tensor) - m) / s, size=[224,224], mode='bilinear', align_corners=False))
                    if mode == 'global': # global descriptors
                        for l in range(len(targets)):
                            ref = network.norm(carrier_x).squeeze()
                            target = targets[network.meta['architecture'], str(scale), 'layer'+str(l)]
                            loss_perf += (1 - ref.mm(target.repeat((ref.shape[0],1)).T)).mean()  # add loss over networks and scales
                            n+= 1

            # compute loss
            if lam > 0:
                loss_distort = (patch_tensor-patch_tensor_org).pow(2.0).sum() / (patch_tensor.size(-1)*patch_tensor.size(-2))
            else:
                loss_distort = torch.Tensor(1).to(device)*0.0
            loss_perf = loss_perf / n  # divide by number of summands (networks, scales, poolings)

            total_loss = loss_perf + lam * loss_distort
            # check for convergence (hacky!)
            if loss_perf < globals()['loss_perf_min']:
                globals()['loss_perf_min'] = loss_perf.clone()

            if loss_perf < globals()['loss_perf_converged']:
                globals()['convergence_safe'] = True

            if globals()['converged'] and (loss_perf-globals()['loss_perf_min']) > 1*globals()['loss_perf_min'] and globals()['convergence_safe'] == False:
                globals()['converged'] = False
                print("Iter {:5d}, Loss_perf = {:6f} Loss_distort = {:6f} Loss_total = {:6f}".format(itr[0], loss_perf.item(), loss_distort.item(), total_loss.item()))
                print('Did not converge')

            total_loss.backward(retain_graph=True)


            

            if verbose == True and itr[0] % 5 == 0:
                print("Iter {:5d}, lr: {:3.3e} Loss_perf = {:6f} Loss_distort = {:6f}, Loss_total = {:6f}".format(
                    itr[0], optimizer.param_groups[0]['lr'], loss_perf.item(), loss_distort.item(), total_loss.item()))
            globals()['loss_perf'] = loss_perf
            globals()['loss_distort'] = loss_distort
            
            return total_loss.item()
        
        if not globals()['converged']: return carriers_tensor.data, 0, 0, False

        patch_tensor.data.clamp_(0, 1)  # correct pixels values
        

        carriers_tensor = (1 - M).mul(carriers_tensor) + M.mul(patch_tensor)

        closure()
        optimizer.step()
        lr_schedulers[0].step()

        itr[0] += 1
    #patch = patch_tensor[:, 0:patch_length, -patch_width:]
    #save_image_tensor(resize(patch),"./attack_patch/5.jpg") 
    
    
    # lr_schedulers[1].step(globals()['loss_perf_min'])

    

    carriers_tensor.data.clamp_(0, 1)  # pixel value correction


    #GS=GaussianSmoothing(channels=3).to(device)
    #carriers_tensor = GS(carriers_tensor)
    #carriers_tensor = mae_progress(model_mae.eval(),nn.functional.interpolate((GS(carriers_tensor) - m) / s, size=[224,224], mode='bilinear', align_corners=False))

    return carriers_tensor.data, globals()['loss_perf'], globals()['loss_distort'], globals()['converged']


def hist_per_channel(x, bin_centers, sigma = 0.1):
    x = x.squeeze(0)
    N = x.size()[1]*x.size()[2]
    xflat = x.flatten().unsqueeze(1)
    expx = torch.exp(
        -torch.add(xflat.type(torch.cuda.DoubleTensor), -1.0*bin_centers.unsqueeze(0)).pow(2.0) / (2*sigma**2)
    ).type(torch.cuda.FloatTensor)
    nf = expx.sum(1).unsqueeze(1)
    nf[nf==0] = 1
    xh = torch.div(expx, nf)
    xh = xh.reshape(x.size(0), N, xh.size(-1))
    hists = xh.sum(1) / (x.size(1)*x.size(2))
    
    return hists

import random
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
        res1=1/(2*math.pi*self.sigema*self.sigema)
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
