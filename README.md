# Super-resolution

This is a master's thesis project
By Damian Łysomirski

What is super resolution ? 
Super-resolution is the task of constructing a higher-resolution image from a lower-resolution image.
While this task has traditionally been approached with non-neural methods such as bilinear and bicubic upsampling,
neural networks offer an opportunity for significant improvements.

Neural netoworks currently implemented for SR:
- SRCNN 
- VDSR
- ESPCN 


## Downloading and instalation
```
# Clone repo
git clone 
cd comparison-of-rendering-methods 

# Creating virtual environment
python -m venv venv 

#Avtivation on linux
source ./venv/bin/activate

#Activation od windows CMD
.\venv\Scripts\activate

#Deactive on windows CMD
.\venv\Scripts\deactivate

# Installing requirements
pip install -r requirements.txt
``` 

## Results of VDSR 1000 epoch training
![alt text](./results/comparison_4_imagesVDSR_23_05_2023_scale_factor_3.png)
![alt text](./plots/psnr_VDSR_23_05_2023_scale_factor_3_epochs_1000.pt.png)
