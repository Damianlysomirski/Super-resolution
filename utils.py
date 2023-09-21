import math
import numpy as np
import torch
# from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

# def ssim(label, output):
#     ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
#     label = label.cpu()
#     output = output.cpu()
#     ssim_ = ssim(output, label)
#     return ssim_
    
def check_device():
    print(torch.cuda.is_available())
    print("\n")
    print(torch.cuda.current_device())
    print("\n")
    print(torch.cuda.device(0))
    print("\n")
    print(torch.cuda.device_count())
    print("\n")
    print(torch.cuda.get_device_name(0))

def load_checkpoint(path, model, optimizer):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  scale_factor = checkpoint['model_scale_factor']
  model_name = checkpoint['model_name']
  train_epoch_loss = checkpoint['train_epoch_loss']
  train_epoch_psnr = checkpoint['train_psnr']
  val_epoch_loss = checkpoint['val_epoch_loss']
  val_epoch_psnr = checkpoint['val_epoch_psnr']

  print("--------------------------------")
  print("Wczytano model")
  print("--------------------------------")
  print("Nazwa modelu: " + model_name)
  print("Współczynnik skalowani: " + str(scale_factor))
  print("Najlepsza epoka: " + str(epoch))
  print("Train loss: " + str(train_epoch_loss))
  print("Train psnr: " + str(train_epoch_psnr))
  print("Valid loss: " + str(val_epoch_loss))
  print("Valid psnr: " + str(val_epoch_psnr))

def plot_psnr(train_psnr, val_psnr, model_save_name):
    # PSNR plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig("./plots/psnr_" + model_save_name +".png")
    plt.close()

def plot_loss(train_loss, val_loss, model_save_name):
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./plots/loss_" + model_save_name +".png")
    plt.close()
