import torch
import torchvision
import os
from torch.optim import SGD
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import h5py
from Datagenerator_Challenge_hog import DataGenerator
from train_eval_hog import train_seg, evaluate_seg
from FCN_ResNet_hog import FCN_ResNet_hog
from deeplabv3_ResNet_hog import deeplabv3_ResNet_hog
from TransUNet_R50_ViT_B_16_hog import TransUNet_R50_ViT_B_16_hog
# from Challenge_Final_Code.Datagenerator_Challenge_hog import DataGenerator
# from Challenge_Final_Code.train_eval_hog import train_seg, evaluate_seg
# from Challenge_Final_Code.FCN_ResNet_hog import FCN_ResNet_hog
# from Challenge_Final_Code.deeplabv3_ResNet_hog import deeplabv3_ResNet_hog
# from Challenge_Final_Code.TransUNet_R50_ViT_B_16_hog import TransUNet_R50_ViT_B_16_hog

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

def training_execution(ar, model_hog, checkpoints_folder_ft, metric_folder_ft, valset):
    learning_rate = ar.learning_rate
    batchsize = ar.batchsize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model_hog = model_hog.to(device)
    optimizer = SGD(model_hog.parameters(), learning_rate, momentum=0.9)
    pat_list = sorted(os.listdir(os.path.join(ar.main_folder, ar.dataset_folder)))
    val_list = [pat_list[index] for index in valset]
    train_list = [x for x in pat_list if x not in val_list]
    train_set = DataGenerator(os.path.join(ar.main_folder, ar.dataset_folder), train_list, batchsize)
    val_set = DataGenerator(os.path.join(ar.main_folder, ar.dataset_folder), val_list, batchsize,
                            transform=torchvision.transforms.Compose(
                                [torchvision.transforms.ToTensor(), torchvision.transforms.RandomCrop(256)]))
    trainloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batchsize, shuffle=True)
    best_valid_loss = float('inf')

    train_loss_list = []
    val_loss_list = []
    print(int(ar.epochs))
    for epoch in range(int(ar.epochs)):
        print(epoch)
        train_loss = train_seg(model_hog, trainloader, optimizer, criterion, device)
        valid_loss = evaluate_seg(model_hog, valloader, criterion, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_hog, os.path.join(checkpoints_folder_ft, ar.model_name + '_' + ar.loss_name + '_' + str(
                ar.epochs) + '_checkpoint.pth'))
            print(f'Saving Model')

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        # saving the metrics value in a dataset
        with h5py.File(os.path.join(metric_folder_ft,
                                    'Loss and metrics_history_' + ar.model_name + '_' + ar.loss_name + '_' + str(
                                        ar.epochs) + '.hdf5'), 'w') as f:
            f.create_dataset('train_loss', data=train_loss_list)
            f.create_dataset('val_loss', data=val_loss_list)
            f.close

        # VISUALIZING LOSS
        plt.figure(0)
        plt.plot(train_loss_list, 'b')
        plt.plot(val_loss_list, 'r')
        plt.ylim([0, 1])  # limite a 1
        plt.title('CE')
        plt.ylabel('LOSS')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(metric_folder_ft, ar.model_name + '_' + ar.loss_name + '_' + str(ar.epochs) + '.png'))


def training(arg):
    fold_val = [(0, 5, 12),
                (1, 9, 14),
                (3, 15, 17),
                (2, 4, 11),
                (6, 7, 16),
                (8, 10, 13)]


    if arg.model_name == 'FCN_ResNet_hog':
        model_hog = FCN_ResNet_hog(True, True, 4)
    if arg.model_name == 'deeplabv3_Resnet_hog':
        model_hog = deeplabv3_ResNet_hog(True, True, 4)
    if arg.model_name == 'TransUNet_R50+ViT-B_16_hog':
        model_hog = TransUNet_R50_ViT_B_16_hog(True, True, 4)

    results_folder_fetreg = os.path.join(arg.main_folder, 'RESULTS_CHALLENGE')
    results_folder_ft = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name + '_pre_trained')

    if not os.path.isdir(results_folder_fetreg):
        os.mkdir(results_folder_fetreg)
    if not os.path.isdir(results_folder_ft):
        os.mkdir(results_folder_ft)

    if arg.cross_fold_validation == 'True':
        for ifold, valset in enumerate(fold_val):
            fold_folder = os.path.join(results_folder_ft, 'RESULTS_FOLD_0{}'.format(ifold+1))
            if not os.path.isdir(fold_folder):
                os.mkdir(fold_folder)
                checkpoints_folder_ft = os.path.join(fold_folder, 'Checkpoints')
                metric_folder_ft = os.path.join(fold_folder, 'Metrics file and graphs')
                if not os.path.isdir(checkpoints_folder_ft):
                    os.mkdir(checkpoints_folder_ft)
                if not os.path.isdir(metric_folder_ft):
                    os.mkdir(metric_folder_ft)
                print("Running fold {}".format(ifold + 1))
                training_execution(arg, model_hog, checkpoints_folder_ft, metric_folder_ft, valset)
            else:
                print("The fold {} is just made".format(ifold + 1))
    else:
        ifold = 0
        valset = fold_val[0]
        fold_folder = os.path.join(results_folder_ft, 'RESULTS_FOLD_0{}'.format(ifold + 1))
        if not os.path.isdir(fold_folder):
            os.mkdir(fold_folder)
            checkpoints_folder_ft = os.path.join(fold_folder, 'Checkpoints')
            metric_folder_ft = os.path.join(fold_folder, 'Metrics file and graphs')
            if not os.path.isdir(checkpoints_folder_ft):
                os.mkdir(checkpoints_folder_ft)
            if not os.path.isdir(metric_folder_ft):
                os.mkdir(metric_folder_ft)
            print("Running fold {}".format(ifold + 1))
            training_execution(arg, model_hog, checkpoints_folder_ft, metric_folder_ft, valset)
        else:
            print("The fold {} is just made".format(ifold + 1))

    return model_hog
