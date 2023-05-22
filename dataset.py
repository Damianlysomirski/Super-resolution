from torchvision.transforms import Compose, RandomCrop, CenterCrop, ToTensor, ToPILImage, GaussianBlur
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

class SR_Dataset(Dataset):
    """ Define training,valid or test dataset loading methods.
    Args:
    scale_factor (int): Image up scale factor.
    path (str): Train/Valid/Test ground truth images dataset adress.
    mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    crop_size (int): Defines the size of ground truth images of training/valid phase.
    """
    def __init__(self, path, scale_factor = 2, crop_size = 64, mode = "train"):
        super(SR_Dataset, self).__init__()
        self.filenames = []
        folders = os.listdir(path)
        for f in folders:
            self.filenames.append(path + f)

        #For training phase use RandomCrop
        self.data_transform_train = Compose([RandomCrop([crop_size, crop_size]), ToTensor()])

        #For validation phase use CenterCrop
        self.data_transform_valid = Compose([CenterCrop([crop_size, crop_size]), ToTensor()])

        #For test phase use full-sizes image
        #Remember to use batch_size = 1, in test phase
        self.data_transform = Compose([ToTensor()])
        self.data_to_tensor = Compose([ToTensor()])
        self.data_transform_PIL = Compose([ToPILImage()])

        #Gausian blur not used, do not give better results
        self.gausian_blur = Compose([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))])

        #Scale factor
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.mode = mode

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])     

        if(self.mode == "train"):
            w = h = self.crop_size
            img = self.data_transform_train(img)
        elif(self.mode == "valid"):
            w = h = self.crop_size
            img = self.data_transform_valid(img)
        elif(self.mode == ("test")):
            w, h = img.size
            img = self.data_transform(img)

            #For test images with different resolutions have to cut them to able division by scale factor
            w_mod = w % self.scale_factor
            h_mod = h % self.scale_factor
            w = w - w_mod
            h = h - h_mod

        else:
             raise ValueError("Unsupported data processing model, please use 'train', 'valid' or 'test'.")


        #Before downsampling transform to PIL image
        resize_image = self.data_transform_PIL(img)       

        #Downsampling
        resize_image = resize_image.resize((int(w/self.scale_factor), int(h/self.scale_factor)))

        #Data to tensor
        resize_image = self.data_to_tensor(resize_image) 

        #Return HR and LR image pair
        HR_image = img
        LR_image = resize_image
        
        return LR_image, HR_image

    def __len__(self):
        return len(self.filenames)
    
    
def show_pair_of_images(scale_factor, path, mode, crop_size):
    dataset = SR_Dataset(scale_factor = scale_factor, path = path, mode = mode, crop_size = crop_size)
    loader = DataLoader(dataset=dataset, num_workers=0, batch_size=32, shuffle=True)
    LR_ , HR_ = next(iter(loader))
    img = LR_[0]
    label = HR_[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(label.permute(1, 2, 0))
    ax[0].title.set_text("HR Image")
    ax[1].imshow(img.permute(1, 2, 0))
    ax[1].title.set_text("LR Image")
    plt.show()

