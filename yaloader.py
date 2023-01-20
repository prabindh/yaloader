# Standard imports
import os, sys
import matplotlib.pyplot as plt
from torchvision import io, transforms, datasets
from torchvision.utils import save_image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch
import cv2
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import ConvVAE
import cProfile

# Definitions

class Args():
    def __init__(self):
        self.model = model
        self.data_dir = raw_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.cuda = use_cuda
        self.epochs = epochs
        self.z_dim = z_dim
        self.filters_m = filters_m
        self.patch_size = patch_size
        self.num_workers = num_workers

class CustomImagesDataset(Dataset):
    """CustomImagesDataset."""

    def __init__(self, args, mode = "train", transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = os.path.join(os.getcwd(), args.data_dir)
        if mode == "test":
            self.data_dir = os.path.join(os.getcwd(), args.test_dir)
        self.list_img_names = [os.path.join(self.data_dir, name) for name in os.listdir(self.data_dir) if os.path.splitext(name)[-1] == '.png']

        test_img = io.read_image(self.list_img_names[0])
        image_w = test_img.shape[2] # read_image ==> [channels,h,w]
        image_h = test_img.shape[1]
        self.num_patches = 0
        self.transform = transform
        self.patch_size = args.patch_size
        self.generate_patches(args, mode)
        
    def __len__(self):
        return self.num_patches

    # populates patch_img_names and num_patches
    def generate_patches(self, args, mode):
        patch_dir = os.path.join(os.getcwd(), mode + "_patch_dir")
        if os.path.exists(patch_dir):
            self.patch_img_names = [os.path.join(patch_dir, name) for name in os.listdir(patch_dir) if os.path.splitext(name)[-1] == '.png']
            self.num_patches = len(self.patch_img_names)
            if self.num_patches > 0:
                print("{} patches already generated, skipping".format(self.num_patches))            
                return
        else:
            os.mkdir(patch_dir)
        
        for name in self.list_img_names:
            test_img = cv2.imread(name)
            image_w = test_img.shape[1] # cv2 imread ==> [h,w,channels], pytorch read_image ==> [channels,h,w]
            image_h = test_img.shape[0]
            num_patches_w = image_w // self.patch_size # per image along w 
            num_patches_h = image_h // self.patch_size # per image along h

            for h in range(num_patches_h):
                for w in range(num_patches_w):
                    crop_img = test_img[h*self.patch_size:h*self.patch_size + self.patch_size, 
                        w*self.patch_size:w*self.patch_size + self.patch_size]
                    fname = os.path.join(patch_dir, "patch_"+str(self.num_patches)+".png")
                    cv2.imwrite(fname, crop_img)
                    self.num_patches += 1
        
        self.patch_img_names = [os.path.join(patch_dir, name) for name in os.listdir(patch_dir) if os.path.splitext(name)[-1] == '.png']    
        print("{} patches generated".format(self.num_patches))        

    # OpenCV based images assumed
    def __getitem__(self, index_list):
        #print (id) #- always an int, even for num_workers > 1 or batch_size > 1, ONLY if sampler is NOT used. If sampler used, it is a list
        
        crop_img_list=torch.empty((len(index_list),3, self.patch_size,self.patch_size), dtype=torch.float32)

        for i,id in enumerate(index_list):
            img = cv2.imread(self.patch_img_names[id])
           
            # Convert BGR image to RGB image for OpenCV to Pytorch compatibility
            crop_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            crop_img = cv2.resize(crop_img, (self.patch_size, self.patch_size))
            #print(crop_img.dtype, crop_img[0][0][1]) #uint8 114, so needs to be normalised
            crop_img = crop_img.astype(np.float32)
            crop_img /= 255.
            crop_img = transforms.ToTensor()(crop_img)
            crop_img_list[i] = crop_img

        return crop_img_list



# Example Setup
transform = None
train_dataset = CustomImagesDataset(args, mode = "train", transform = transform)
test_dataset = CustomImagesDataset(args, mode = "test", transform = transform)

train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
test_sampler = BatchSampler(RandomSampler(test_dataset), batch_size=args.batch_size, drop_last=False)

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, **kwargs)

# Example usage
for batch_idx, data in enumerate(train_loader):
    data = data.squeeze()  
    ### use data below
