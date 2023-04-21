from torchvision.transforms import Compose, RandomCrop, CenterCrop, ToTensor, ToPILImage, GaussianBlur
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

class TrainValidDataset(Dataset):
    """ Define training or valid dataset loading methods.
    Args:
    scale_factor (int): Image up scale factor.
    path (str): Train/Valid ground truth images dataset adress.
    mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    color_mode (str): Define output of dataset, if mode = RGB, output will be rgb, if mode w "YCbCr" 
    output will be only Y - lumimance chanell.
    crop_size (int): Defines the size of ground truth images.
    """
    def __init__(
            self,
            scale_factor: int,
            path: str,
            mode: str,
            color_mode: str,
            crop_size: int,
    ) -> None:
        super(TrainValidDataset, self).__init__()
        self.filenames = []
        folders = os.listdir(path)
        for f in folders:
            self.filenames.append(path + f)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.mode = mode
        self.color_mode = color_mode

        #Data transforms and conversions
        self.random_crop = Compose([RandomCrop([self.crop_size, self.crop_size]), ToTensor()])
        self.center_crop = Compose([CenterCrop([self.crop_size, self.crop_size]), ToTensor()])
        self.data_to_tensor = Compose([ToTensor()])
        self.data_transform_PIL = Compose([ToPILImage()])
        self.gausian_blur = Compose([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))])

    def __getitem__(self, index):
        #Reading images
        img = Image.open(self.filenames[index])   

        #Image procesing operations - crop
        if (self.mode == "Train"):
            img = self.random_crop(img)   
        elif (self.mode == "Valid"):
            img = self.center_crop(img)   
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")
        
        #Image procesing operations - color mode
        if (self.color_mode == "YCbCr"):
            img = self.data_transform_PIL(img)
            img, _cb, _cr = img.convert('YCbCr').split() 
            HR_image = self.data_to_tensor(img)
            LR_image = img.resize((int(self.crop_size/self.scale_factor), int(self.crop_size/self.scale_factor)))
            LR_image = self.data_to_tensor(LR_image)
        elif (self.color_mode == "RGB"):
            HR_image = img
            LR_image = self.data_transform_PIL(HR_image)
            LR_image = LR_image.resize((int(self.crop_size/self.scale_factor), int(self.crop_size/self.scale_factor)))
            LR_image = self.data_to_tensor(LR_image)
        else:
            raise ValueError("Unsupported data color mode, please use `RGB` or `YCbCr`.")

        return LR_image, HR_image

    def __len__(self):
        return len(self.filenames)
    
def show_pair_of_images(scale_factor, path, mode, color_mode, crop_size):
    train_dataset = TrainValidDataset(scale_factor = scale_factor, path = path, mode = mode, color_mode = color_mode, crop_size = crop_size)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=32, shuffle=True)
    LR_ , HR_ = next(iter(train_loader))
    img = LR_[0]
    label = HR_[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(label.permute(1, 2, 0))
    ax[0].title.set_text("HR Image")
    ax[1].imshow(img.permute(1, 2, 0))
    ax[1].title.set_text("LR Image")
    plt.show()


if __name__ == "__main__":
    show_pair_of_images(scale_factor = 2, path = "./resources/BSDS200/", mode = "Train", color_mode= "YCbCr", crop_size = 64)
    