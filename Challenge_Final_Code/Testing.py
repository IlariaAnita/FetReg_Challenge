from itertools import product
import cupy
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.feature import hog

"""# Intsall Modules"""

# pip install --user segmentation_models_pytorch albumentations

'''
List of Video in the Challenge Folder
0   Video001 UCLH F1 - 8    Video009 UCLH F6  - 13  Video017 IGGH F6
1   Video002 UCLH F2 - *    Video010 IGGH FT  - 14  Video018 IGGH F2
2   Video003 UCLH F4 - 9    Video011 IGGH F2  - 15  Video019 UCLH F3
3   Video004 IGGH F3 - *    Video012 IGGH FT  - *   Video020 UCLH FT
4   Video005 IGGH F4 - 10   Video013 UCLH F6  - *   Video021 UCLH FT
5   Video006 IGGH F1 - 11   Video014 UCLH F4  - 16  Video022 IGGH F5
6   Video007 UCLH F5 - *    Video015 UCLH FT  - 17  Video023 IGGH F3
7   Video008 UCLH F5 - 12   Video016 IGGH F1  - *   Video024 IGGH FT
'''

#VISUALIZATION OF MASKS AND PREDICTIONS - RESIZED 256
def get_colormap():
    """
    Returns FetReg colormap
    """
    colormap = np.asarray(
        [
            [0, 0, 0],  # 0 - background
            [255, 0, 0],  # 1 - vessel
            [0, 0, 255],  # 2 - tool
            [0, 255, 0],  # 3 - fetus

        ]
    )
    return colormap


def evaluation(ar, checkpoint_path, valset, mask_path, plot_path):
    imgsize = 256
    stride = 8
    whereLoadVideo = os.path.join(ar.main_folder, ar.dataset_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    colormap = get_colormap()

    print(checkpoint_path)
    model = torch.load(checkpoint_path)
    model = model.to(device)
    pat_list = sorted(os.listdir(whereLoadVideo))
    val_list = [pat_list[index] for index in valset]
    model.eval()
    for vFold in val_list:
        print('We are in val list')
        x = 0
        mask_vfold_path = os.path.join(mask_path,vFold)
        if not os.path.isdir(mask_vfold_path):
            print('Running Testing of video: ', vFold)
            for frame in tqdm(sorted(os.listdir(os.path.join(whereLoadVideo, vFold, 'images')))):
                print(os.path.join(whereLoadVideo, vFold, 'images'))
                with torch.no_grad():
                    frameI = Image.open(os.path.join(whereLoadVideo, vFold, 'images', frame))
                    frameV = Image.open(os.path.join(whereLoadVideo, vFold, 'labels', frame))
                    frameV = np.array(frameV)
                    # print('Shape of GT', frameV.shape)  # 470 470
                    # print('Max and min: ', np.max(frameV), np.min(frameV))  # 3 and 0
                    # print('Type: ', frameV.dtype)  # uint8
                    h, w = frameI.size
                    if h > imgsize or w > imgsize:
                        pieceW = int(np.ceil((w - imgsize) / stride))
                        pieceH = int(np.ceil((h - imgsize) / stride))
                        # pieces = cupy.zeros((pieceH*pieceW, imgsize, imgsize, 3), dtype=cupy.float32)
                        frameI = cupy.array(frameI)
                        output = cupy.zeros((4, imgsize + (pieceH * stride), imgsize + (pieceW * stride)),
                                            dtype=cupy.float32)
                        # print(output.shape)
                        times = cupy.zeros((imgsize + (pieceH * stride), imgsize + (pieceW * stride)),
                                           dtype=cupy.float32)
                        # print(times.shape)
                        tempImage = cupy.zeros((imgsize + (stride * pieceH), imgsize + (stride * pieceW), 3),
                                               dtype=cupy.uint8)
                        tempImage[:h, :w, :] = frameI
                        for i, (hpart, wpart) in enumerate(product(range(pieceH), range(pieceW))):
                            hsize1 = hpart * stride
                            hsize2 = (hpart * stride) + imgsize
                            wsize1 = wpart * stride
                            wsize2 = (wpart * stride) + imgsize
                            # print("{}:{}, {}:{} ({},{})".format(hsize1, hsize2, wsize1, wsize2, hpart, wpart))
                            pieces = cupy.asnumpy(tempImage[hsize1:hsize2, wsize1:wsize2, :])

                            frameT = transforms.ToTensor()(pieces)

                            # frameT = torch.nn.AdaptiveAvgPool2d((256,256))(frameT)
                            # frameT = transforms.Resize(256,transforms.InterpolationMode.BILINEAR)(frameT)
                            frameT = frameT.to(device).unsqueeze(0)
                            hog_vector, hog_image = hog(pieces, orientations=8, pixels_per_cell=(16, 16),
                                                        cells_per_block=(1, 1), visualize=True)
                            hog_image_tensor = torch.from_numpy(hog_image)
                            hog_vector = torch.from_numpy(hog_vector).float()
                            hog_vector = hog_vector.to(device).unsqueeze(0)

                            scores = model(frameT, hog_vector)
                            scores = torch.softmax(scores, dim=1)
                            scores = scores.to('cpu').numpy().squeeze(0)[:,:,:]
                            output[:, hsize1:hsize2, wsize1:wsize2] += cupy.asarray(scores)
                            times[hsize1:hsize2, wsize1:wsize2] += 1
                        times = 1 / times
                        times = cupy.repeat(times[cupy.newaxis, :, :], 4, axis=0)
                        output *= times
                        output = output[:, :h, :w]
                        scores = torch.from_numpy(cupy.asnumpy(output)).unsqueeze(0)
                    else:
                        frameT = transforms.ToTensor()(frameI)
                        # frameT = torch.nn.AdaptiveAvgPool2d((256,256))(frameT)
                        # frameT = transforms.Resize(256,transforms.InterpolationMode.BILINEAR)(frameT)
                        frameT = frameT.to(device).unsqueeze(0)
                        scores = model(frameT)
                        scores = torch.softmax(scores, dim=1)
                    # print(scores.shape) # 1 4 470 470
                    scores = torch.argmax(scores, dim=1)  # 4 470 470
                    scores = scores.to('cpu').numpy().squeeze(0)  # 470 470
                    # print('Prediction shape', scores.shape)  # 470 470
                    # print('Max and min: ', np.max(scores), np.min(scores))  # 3 and 0
                    # print('Type: ', scores.dtype)  # uint64
                    predim = Image.fromarray(scores.astype(np.uint8))
                    mask_vfold_path = os.path.join(mask_path, vFold)
                    print(mask_vfold_path)
                    if not os.path.isdir(mask_vfold_path):
                        os.mkdir(mask_vfold_path)
                    try:
                        os.mkdir(mask_vfold_path)
                    except:
                        print(mask_vfold_path, 'already exist')
                    predim.save(os.path.join(mask_vfold_path, frame))

                    mask_rgb = np.zeros(scores.shape[:2] + (3,), dtype=np.uint8)
                    gt_rgb = np.zeros(frameV.shape[:2] + (3,), dtype=np.uint8)
                    print(scores.shape, mask_rgb.shape)
                    for cnt in range(len(colormap)):
                        mask_rgb[scores == cnt] = colormap[cnt]
                        gt_rgb[frameV == cnt] = colormap[cnt]

                    #VISUALIZATION

                    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                    axs[0].imshow(gt_rgb)
                    axs[0].axis("off")

                    axs[1].imshow(mask_rgb)
                    axs[1].axis("off")
                    fig.tight_layout()
                    #plt.show()
                    if not os.path.isdir(plot_path):
                        os.mkdir(plot_path)
                    fig.savefig(os.path.join(plot_path, frame))
                    x = x + 1

def testing(arg, model_hog):
    fold_val = [(0, 5, 12),
                (1, 9, 14),
                (3, 15, 17),
                (2, 4, 11),
                (6, 7, 16),
                (8, 10, 13)]
    results_folder_fetreg = os.path.join(arg.main_folder, 'RESULTS_CHALLENGE')
    if arg.pre_trained == 'True':
        results_folder_ft = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name + '_pre_trained')
    else:
        results_folder_ft = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name)
    for ifold, valset in enumerate(fold_val):
        fold_folder = os.path.join(results_folder_ft, 'RESULTS_FOLD_0{}'.format(ifold+1))
        if os.path.isdir(fold_folder):
            checkpoints_folder_ft = os.path.join(fold_folder, 'Checkpoints')
            checkpoint_name = arg.model_name + '_' + arg.loss_name + '_' + str(arg.epochs) + '_checkpoint.pth'
            checkpoint_file_path = os.path.join(checkpoints_folder_ft, checkpoint_name)
            print("Running the testing of fold {}".format(ifold + 1))
            mask_path = os.path.join(fold_folder, 'Predicted_Masks_Original_{}'.format(arg.epochs))
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            plot_path = os.path.join(fold_folder, 'Predicted_Masks_Plot_{}'.format(arg.epochs))
            if not os.path.isdir(plot_path):
                os.mkdir(plot_path)
                evaluation(arg, checkpoint_file_path, valset, mask_path, plot_path)
        else:
            print("No Running the testing of fold {} because the training wasn't made".format(ifold + 1))
