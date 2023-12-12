# Implementation of UNet++ on Breast Ultrasound Cancer Tumor Segmentation (BUS Syntehtic Dataset)
## Usage

- Install environmental packages:
    
    `conda create -n <env_name> python=3.9`
    
    `conda activate <env_name>`
    
    `pip install -r requirements.txt`
    
- Training
    
    `python train.py --dataset BUS_synthetic_dataset --arch NestedUNet --img_ext .png --mask_ext .png --epoch 1000`
    
- Validation
    
    `python val.py --name BUS_synthetic_dataset_NestedUNet_woDS`
    
- Inference on the out_112.bmp image
    
    `python inference.py --name BUS_synthetic_dataset_NestedUNet_woDS`
    
    → **pred_mask_112.bmp** and **overlay_112.bmp** is saved in the ./images file