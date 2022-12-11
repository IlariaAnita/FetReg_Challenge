# HOG Injection into ANN for Fetoscopic Multi-Class Segmentation
This is an official implementation of the method developed by the team BioPolmini from Politecnico di Milano (Italy), composed by Chiara Lena, Jessica Biagioli, 
Gaia Romana and Ilaria Anita Cintorrino, to partecipate at the e Fetoscopic Placental Vessel Segmentation and Registration (FetReg2021) Challenge at MICCAI2021. 
The method is described into https://github.com/IlariaAnita/FetReg_Challenge/blob/main/Write_Up_Challenge%20(1).pdf

![Framework](https://github.com/IlariaAnita/FetReg_Challenge/blob/main/framework.jpeg)

# Dataset
The first large-scale multi-centre Twin-to-Twin Transfusion Syndrome (TTTS) dataset was used. Dataset is composed by 2060 images, pixel-annotated for
vessels, tool, fetus and background classes, from 18 in vivo TTTS fetoscopy procedures. https://arxiv.org/abs/2106.05923v2

# Training setting 
Each experiment was trained on 1708 images and evaluated on 352 images belonging to FetReg Dataset.
After assessing the best model, we performed 6-fold cross-validation to verify the robustness of the segmentation algorithm. 
To be consistent with the FetReg Challenge baseline, we resized the training images to 448x448 pixels. 
We applied data augmentation consisting of: 
- random crop of the image with dimension 256x256 pixels
- random rotation in range (􀀀45°, +45°)
- horizontal and vertical flip and random variation in brightness (-20%,+20%). 

The learning rate and batch size were set to 0.001 and 32 for the CNNs and 0.01 and 8 for TransUnet respectively.
As common practice, we trained the ResnNet50-HOG and DeepLabv3-HOG with pre-trained ImageNet [https://ieeexplore.ieee.org/document/5206848] weights initialization 
to boost the model performance. 
For the experiments, training was performed for 300 epochs. The final weights submitted to the FetReg Challenge are obtained training the best model over all 
the 2060 annotated frames, following the same described setup, for 700 epochs.

The networks were trained with two 32 GB of RAM and NVIDIA Tesla V100 GPU.
