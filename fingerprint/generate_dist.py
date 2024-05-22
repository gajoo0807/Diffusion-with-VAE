import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.resnet import *
import argparse


# Define a custom dataset class to load images from the 'sample' folder
class SampleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
def generate_distribution(model_name):
    # Load the pretrained ResNet18 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    checkpoint = torch.load(f'./fingerprint/model/{model_name}.pth')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Create dataset and dataloader
    dataset = SampleDataset(root_dir='./fingerprint/sample', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create a directory to store the output
    output_dir = f'./fingerprint/distribution'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform inference and save the results
    with torch.no_grad():
        all_probabilities = []
        for i, inputs in enumerate(dataloader):
            outputs = model(inputs)
            # Convert the output to probabilities (if applicable)
            # print(f'{outputs=}')
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # print(f'{probabilities=}')
            all_probabilities.append(probabilities)
        # Save all probabilities to a single file
        torch.save(all_probabilities, f'{output_dir}/{model_name}_probabilities.pth')

def print_probabilities(path):
    # 將Distribution印出
    all_probabilities = torch.load(path)
    
    # Iterate over the probabilities and print them
    # for i, probabilities in enumerate(all_probabilities):
    #     print(f'Probabilities for image {i+1}:')
    #     print(probabilities)
    # Randomly select one probabilities and print it
    random_index = np.random.randint(len(all_probabilities))
    probabilities = all_probabilities[random_index]
    print(f'Probabilities for image {random_index+1}:')
    print(probabilities)

def generate_gaussian_noise(n):
    # 生成n張高斯噪聲圖像並保存到'sample'資料夾中
    if not os.path.exists('sample'):
        os.makedirs('sample')
    
    # Image dimensions corresponding to CIFAR-10 (32x32 pixels)
    height, width = 32, 32

    for i in range(n):
        # Generate Gaussian noise
        noise = np.random.normal(loc=0.0, scale=1.0, size=(height, width, 3))
        
        # Normalize to [0, 255]
        noise = 255 * (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise.astype(np.uint8)

        # Save the image
        plt.imsave(f'fingerprint/sample/{i}.png', noise)



# Example usage: generate 5 Gaussian noise images
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--model_num', type=str, default='5', help='Number of the model to generate distribution for')
    args = parser.parse_args()
    # generate_gaussian_noise(1000)
    for i in range(1, int(args.model_num)+1):
        generate_distribution(model_name=f'model_{i}')
    # print_probabilities('./fingerprint/distribution/model_1_probabilities.pth')