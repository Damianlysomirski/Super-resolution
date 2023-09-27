from utils import psnr
from tqdm import tqdm
from dataset import SR_Dataset
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from models.VDSR import VDSR
from models.SRCNN import SRCNN
from models.ESPCN import ESPCN
from models.Bicubic import Bicubic
from validate import validate
from utils import psnr
from tqdm import tqdm

TRAIN_DATASET = './resources/BSDS200/'
EVAL_DATASET = './resources/Set5/'

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    #for bi, data in enumerate(dataloader):
        image_data = data[0].to(device)
        label = data[1].to(device)

        # Zero grad the optimizer.
        optimizer.zero_grad()

        outputs = model(image_data)
        loss = criterion(outputs, label)
        # Backpropagation.
        loss.backward()
        # Update the parameters.
        optimizer.step()
        # Add loss of each item (total items in a batch = batch size).
        running_loss += loss.item()
        # Calculate batch psnr (once every `batch_size` iterations).
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr

        #Plot 1 image from batch
        # output = outputs[0].detach().cpu().numpy() 
        # output = output.transpose(1,2,0) #konwersja tensora na ndarray
        # plt.imshow(output)
        # plt.show()

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr

"""
Function to build model depending on parsing arguments
"""
def build_model(model_name, scale_factor, device):
    if (model_name == "VDSR"):
        model = VDSR(scale_factor).to(device)
        print("Built model VDSR successfully !")
    elif (model_name == "SRCNN"):
        model = SRCNN(scale_factor).to(device)
        print("Built model SRCNN successfully !")
    elif (model_name == "ESPCN"):
        model = ESPCN(scale_factor).to(device)
        print("Built model ESPCN successfully !")
    else:
        raise ValueError(
            "Unsupported neural network model, please use 'VDSR', 'SRCNN' or 'ESPCN'.")
    return model


"""
Function to define loss, some models have various loss functions.
"""
def define_criterion(model_name):
    if (model_name == "VDSR"):
        criterion = nn.MSELoss(reduction="sum")
    else:
        criterion = nn.MSELoss()
    return criterion


def define_optimizer(model_name, model):
    if (model_name == "VDSR"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        pass
    elif (model_name == "SRCNN"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        pass
    elif (model_name == "ESPCN"):
        # optimizer = optim.SGD(model.parameters(),
        #                       lr=1e-2,
        #                       momentum=0.9,
        #                       weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return optimizer

def main() -> None:
    CROP_SIZE_CONST = 22

    parser = argparse.ArgumentParser(
        prog="Super-resolution training",
        description="Program allows training 5 different neural networks for SR usage",
        epilog="Example usage: python train.py -m VDSR -s 3 -e 10", )
    parser.add_argument(
        "-m",
        "--model",
        help="Expected model names: SRCNN, VDSR, ESPCN",
        required=True,
        default="ESPCN"
    )
    parser.add_argument(
        "-s",
        "--scale",
        help="Value of scale factor",
        required=True,
        type=int,
        default=3
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of training epochs",
        required=True,
        type=int,
        default=100
    )
    parser.add_argument(
        "-t",
        "--training_mode",
        help="Mode of training: training_from_the_begining or training_from_the_checkpoint",
        required=True,
        default="training_from_the_beginning"
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        help="Path to checkpoint if the mode is training_from_checkpoint",
        default=None
    )

    args = parser.parse_args()

    # Define the number of training epochs
    epochs = args.epochs
    print("Defined training epochs: " + str(epochs))

    # Define scale factor
    scale_factor = args.scale
    print("Defined scale_factor: " + str(scale_factor))

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = build_model(args.model, args.scale, device)

    # Define datasets
    train_dataset = SR_Dataset(scale_factor=scale_factor, path=TRAIN_DATASET,
                               crop_size=scale_factor*CROP_SIZE_CONST, mode="train")
    eval_dataset = SR_Dataset(scale_factor=scale_factor, path=EVAL_DATASET,
                              crop_size=scale_factor*CROP_SIZE_CONST, mode="valid")

    # Define dataloaders
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=0, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=eval_dataset,
                              num_workers=0, batch_size=64, shuffle=False)

    # Define loss criterion
    criterion = define_criterion(args.model)

    # Define optimizer and scheduler
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = define_optimizer(args.model, model)

    # Define mode of training: training_from_the_begining or training_from_checkpoint
    training_mode = args.training_mode
    print("Training mode: " +str(training_mode))

    """
    Check for load
    """
    if ((str(args.training_mode)) == "training_from_checkpoint"):
        if (str(args.checkpoint_path) is None):
            raise ValueError(
                "The path to the checkpoint has not been provided")
        else:
            try:
                checkpoint = torch.load(args.checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                scale_factor = checkpoint['model_scale_factor']
                model_name = checkpoint['model_name']
                train_loss = checkpoint['train_loss']
                train_psnr = checkpoint['train_psnr']
                val_loss_dict = checkpoint['val_loss']
                val_psnr_dict = checkpoint['val_psnr']
                best_valid_loss = checkpoint['best_valid_loss']
                epochs += epoch
                start_epoch = epoch

            except ValueError:
                print("Oops! Passed wrong path to the checkpoint, check it and try again")
    else:
        start_epoch = 0
        train_loss, train_psnr = [], []
        val_psnr_dict, val_loss_dict = {}, {}
        best_valid_loss = 1

        #Lets try with dict

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch} of {epochs}")

        train_epoch_loss, train_epoch_psnr = train(model, train_loader, optimizer, criterion, device)
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)

        if (epoch % 25 == 0):
        #Validation every 25 epoch
            val_epoch_loss, val_epoch_psnr = validate(model, valid_loader, optimizer, criterion, device)
            val_psnr_dict[epoch] = val_epoch_psnr
            val_loss_dict[epoch] = val_epoch_loss
            
            # print(f"Train PSNR: {train_epoch_psnr:.3f}")
            # print(f"Val PSNR: {val_epoch_psnr:.3f}")
            print(f"Val LOSS: {val_epoch_loss:.16f}")
            print(f"Val PSNR: {val_epoch_psnr:.16f}")

            #Save only the epoch with least loss
            if (val_epoch_loss < best_valid_loss):
                best_valid_loss = val_epoch_loss

                model_name = model.__class__.__name__
                model_save_name = model_name + "_" + "sf_" + str(scale_factor) + "_epoch_" + str(epoch) + ".pt"
                
                path = "./checkpoints_new/" + model_save_name

                torch.save({
                        'epoch': epoch,
                        'model_name': model_name,
                        'model_scale_factor': scale_factor,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_epoch_loss': train_epoch_loss,
                        'train_psnr' : train_epoch_psnr,
                        'val_epoch_loss': val_epoch_loss,
                        'val_epoch_psnr': val_epoch_psnr,
                        'train_loss' : train_loss,
                        'val_loss': val_loss_dict,
                        'train_psnr': train_psnr,
                        'val_psnr': val_psnr_dict,
                        'best_valid_loss': best_valid_loss
                        }, path)
                
                print("New checkpoint created")

    print(val_psnr_dict)        

if __name__ == "__main__":
    main()
