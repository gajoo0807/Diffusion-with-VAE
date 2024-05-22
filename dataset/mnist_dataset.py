import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, im_path, im_size, im_channels, model_num, dataset_dict, 
                 use_latents=False, latent_path=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        self.model_num = model_num
        
        # Conditioning for the dataset
        # self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.condition_types = 1
        self.dataset_dict = dataset_dict

        self.images, self.labels = self.load_images(im_path)
        
        
        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        # for d_name in tqdm(os.listdir(im_path)):
        for d_name in range(1, self.model_num+1):
            data_path = os.path.join(im_path, str(self.dataset_dict[d_name]))
            dist_path = f'fingerprint/distribution/model_{d_name}_probabilities.pth'
            all_probabilities = torch.load(dist_path)

            fnames = glob.glob(os.path.join(data_path, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpeg')))
            for fname in fnames:
                ims.append(fname)
                random_element = all_probabilities[np.random.randint(len(all_probabilities))]
                reshaped_element = random_element.reshape(-1)
                labels.append(reshaped_element)

        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = self.labels[index]
        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.ToTensor()(im)
            
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            return im_tensor, cond_inputs
            
