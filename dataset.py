from torchvision.transforms import Compose, RandomCrop, CenterCrop, ToTensor, ToPILImage, GaussianBlur
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product

class SR_Dataset(Dataset):
    """ Defines the dataset for training, validation, or testing.
    Arguments:
    scale_factor (int): Image upscaling factor.
    path (str): Address of the image source dataset.
    mode (str): Type of the dataset, training, validation, test.
    crop_size (int): Specifies the cropping size of the images.
    """
    def __init__(self, path, scale_factor, crop_size, mode):
        super(SR_Dataset, self).__init__()
        self.filenames = []
        folder = os.listdir(path)
        for f in folder:
            self.filenames.append(path + f)

        self.data_transform_train = Compose([RandomCrop([crop_size, crop_size]), ToTensor()])
        self.data_transform_valid = Compose([CenterCrop([crop_size, crop_size]), ToTensor()])
        self.data_transform = Compose([ToTensor()])
        self.data_transform_PIL = Compose([ToPILImage()])
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.mode = mode

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])     

        if(self.mode == "train"):
            w = h = self.crop_size
            img = self.data_transform_train(img) #For training phase use RandomCrop
        elif(self.mode == "valid"):
            w = h = self.crop_size #For validation phase use CenterCrop
            img = self.data_transform_valid(img)
        elif(self.mode == ("test")):
            w, h = img.size #For test phase use full-sizes images

            # Cut images to able division by scale_factor
            w_mod = w % self.scale_factor
            h_mod = h % self.scale_factor
            w = w - w_mod
            h = h - h_mod
            img = img.resize((int(w), int(h)), Image.BICUBIC)  
            img = self.data_transform(img)
        else:
             raise ValueError("Unsupported data processing model, please use 'train', 'valid' or 'test'.")

        #Before downsampling transform to PIL image
        resize_image = self.data_transform_PIL(img)       
        #Downsampling
        resize_image = resize_image.resize((int(w/self.scale_factor), int(h/self.scale_factor)))
        #Transform PIL image to tensor 
        resize_image = self.data_transform(resize_image) 

        #Return HR and LR image pair
        HR_image = img
        LR_image = resize_image
        return LR_image, HR_image

    def __len__(self):
        return len(self.filenames)


#Function to show pair of images
def show_pair_of_images(scale_factor, path, mode, crop_size):
    dataset = SR_Dataset(scale_factor = scale_factor, path = path, mode = mode, crop_size = crop_size)
    loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=True)
    LR_ , HR_ = next(iter(loader))
    img = LR_[0]
    print(img.size())
    label = HR_[0]
    print(label.size())
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(label.permute(1, 2, 0))
    ax[0].title.set_text("HR")
    ax[1].imshow(img.permute(1, 2, 0))
    ax[1].title.set_text("LR")
    plt.show()

#Function to tail all images in folder
def tile_all(dir_in, dir_out, d):
    for fp in os.listdir(dir_in):
        if fp.endswith(('.png', '.jpg', '.jpeg')):  # Add or remove file extensions as needed
            tile_one(fp, dir_in, dir_out, d)

#Function to tail single image
def tile_one(fp, dir_in, dir_out, d):
    name, ext = os.path.splitext(fp)
    img = Image.open(os.path.join(dir_in, fp))
    w, h = img.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

if __name__ == "__main__":
    #tile_all('./resources/DIV2K', './resources/DIV2K_s', 500)
    tile_one('Img6.jpg' , './private_old_photos1/', './private_old_photos1/', 200)
    #show_pair_of_images(scale_factor=3, path='./resources/DIV2K_split/' ,mode='train',crop_size=144)


    