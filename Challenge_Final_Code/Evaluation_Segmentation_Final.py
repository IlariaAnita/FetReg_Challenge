import os
import numpy as np
import cv2.cv2 as cv2
import tensorflow as tf


def clip_extraction(arg):
    folder = arg.dataset_folder
    clip_listed = sorted(os.listdir(folder))
    clip_listed_path = []
    for c in clip_listed:
        clip_path = os.path.join(folder, c)
        clip_listed_path.append(clip_path)
    return clip_listed, clip_listed_path


class Confusion_Matrix:
    """nclasses = number of classes -> are int value
    cm = confusion matrix is 2D array
    """
    def __init__(self, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.cm = np.zeros((self.nclasses, self.nclasses))

    def reset(self):
        self.cm = np.zeros((self.nclasses, self.nclasses))

    def get_cm(self):
        return self.cm

    def update_cm(self, gt_mask, pre_mask):
        mask = (gt_mask >= 0) & (gt_mask < self.nclasses)
        label = self.nclasses * gt_mask[mask].astype("int") + pre_mask[mask].astype("int")
        count = np.bincount(label, minlength=self.nclasses ** 2)
        self.cm += count.reshape(self.nclasses, self.nclasses)
        return self.cm


def acc_pixel(cm):
    """mean accuracy of the single pixel"""
    acc = np.diag(cm).sum() / cm.sum()
    return acc


def acc_pixel_class(cm):
    """mean accuracy of the single pixel per class
    """
    acc = np.diag(cm) / cm.sum(axis=1)
    acc = np.nanmean(acc)
    return acc


def mean_iou(cm):
    m_iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    m_iou = np.nanmean(m_iou)
    return m_iou


def evaluation_segmentation(arg):
    results_folder_fetreg = os.path.join(arg.main_folder, 'RESULTS_CHALLENGE')
    if arg.pre_trained == 'True':
        results_folder_ft = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name + '_pre_trained')
    else:
        results_folder_ft = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name)
    clip_list, clip_list_path = clip_extraction(arg)
    # print('Number of clip: ', len(clip_list))

    fold_val = [(0, 5, 12),
                (1, 9, 14),
                (3, 15, 17),
                (2, 4, 11),
                (6, 7, 16),
                (8, 10, 13)]

    clip_t = clip_list
    clip_t_path = clip_list_path
    # print(clip_t_path)
    clip_list = []
    clip_list_path = []

    # Creating a confusion matrix for the 4 classes: backgroun, vessel, tool, fetus
    mean = Confusion_Matrix(nclasses=arg.num_classes)
    m = tf.keras.metrics.MeanIoU(num_classes=arg.num_classes)  # metric mean Intersection over Union

    # Creating the empty list of mIoU with all the classes and one with all the videos and classes
    mIoU_list_class = np.zeros((1, arg.num_classes), dtype=float)
    mIOU_list_clip_class = np.zeros((len(clip_t), arg.num_classes), dtype=float)
    mIOU_list_clip_overall = np.zeros(len(clip_t), dtype=float)
    mIOU_list_fold_class = np.zeros((len(fold_val), arg.num_classes), dtype=float)
    mIOU_list_fold_overall = np.zeros(len(fold_val), dtype=float)

    cm_list_fold_overall = []
    for ifold, val_set in enumerate(fold_val):
        if arg.cross_fold_validation != 'True':
            if ifold > 0:
                break
        cm_list_fold_overall.append([])
        print(' ')
        print("Running fold {}".format(ifold + 1))
        for i in range(4):
            print('Class: ', i)
            # Creating cm for each class
            cm_class = Confusion_Matrix(nclasses=2)
            # print(cm_class.get_cm())
            for ivideo, video in enumerate(val_set):
                fold_folder = os.path.join(results_folder_ft, 'RESULTS_FOLD_0{}'.format(ifold + 1))
                pred_path = os.path.join(fold_folder, 'Predicted_Masks_Original_{}'.format(arg.epochs))
                clip_list = []
                clip_list_path = []
                cm_fold_class = Confusion_Matrix(nclasses=2)
                if i == 0:
                    cm_fold = Confusion_Matrix(nclasses=arg.num_classes)
                else:
                    cm_fold = cm_list_fold_overall[ifold]
                # print(clip_t_path[0])

                # Creating cm for each video with True and False value
                print("Evaluating Class {0} for the video {1}".format(i, video))
                cm_clip = Confusion_Matrix(nclasses=2)
                cm_clip_overall = Confusion_Matrix(nclasses=4)
                # print(clip_t_path[video])
                gt_list = sorted(os.listdir(os.path.join(clip_t_path[video], "labels")))
                for x, frame in enumerate(gt_list):
                    # print(ivideo, x, frame)
                    # Creating cm for each prediction with True and False value
                    cm_frame = Confusion_Matrix(nclasses=2)

                    gt = cv2.imread(os.path.join(os.path.join(clip_t_path[video], "labels"), frame), 0)
                    gt_resized = cv2.resize(gt, (256, 256), cv2.INTER_NEAREST)
                    # pred = gt
                    pred = cv2.imread(os.path.join(os.path.join(pred_path, clip_t[video]), frame), 0)
                    # print(os.path.join(os.path.join(pred_path, clip_t[video]), frame))
                    pred_resized = cv2.resize(pred, (256, 256), cv2.INTER_NEAREST)
                    gt = (gt_resized == i).astype(int)  # True if the pixel of the image is in the class i else False
                    pred = (pred_resized == i).astype(int)  # True if the pixel of the image is in the class i else False
                    # print(gt, pred)

                    # Update the 3 confusion matrix: for the frame, for the class, for the clip
                    cm_frame.update_cm(gt, pred)
                    cm_class.update_cm(gt, pred)
                    cm_clip.update_cm(gt, pred)
                    cm_fold_class.update_cm(gt, pred)
                    cm_clip_overall.update_cm(gt_resized, pred_resized)
                    cm_fold.update_cm(gt_resized, pred_resized)

                    # Update the confusion matrix for the 4 classes
                    mean.update_cm(gt_resized, pred_resized)

                # Get the data of the confusion matrix of the videos
                # print('Clip cm: ', cm_clip)
                clip_cm = cm_clip.get_cm() # the matrix cm is an object and so we get the value from the object
                # print('Clip cm after get cm: ', clip_cm)
                acc = acc_pixel(clip_cm)  # accuracy of the pixels for each clip cm
                acc_class = acc_pixel_class(clip_cm)  # accuracy of the pixels for each clip cm per each classes
                miou = mean_iou(clip_cm)  # calculating the mean IoU per each clip for each class
                mIOU_list_clip_class[video, i] = miou
                print("Pixel accuracy of the clip is: ", acc)
                print("Pixel accuracy of the {0} class is: ".format(i), acc_class)
                print("The Mean intersection over union of the clip {0} for the class {1} is: ".format(clip_t[ivideo], i), miou)

                clip_overall_cm = cm_clip_overall.get_cm()  # the matrix cm is an object and so we get the value from the object
                # print('Clip cm after get cm: ', clip_cm)
                acc = acc_pixel(clip_overall_cm)  # accuracy of the pixels for each clip cm
                acc_class = acc_pixel_class(clip_overall_cm)  # accuracy of the pixels for each clip cm per each classes
                miou = mean_iou(clip_overall_cm)  # calculating the mean IoU per each clip for each class
                mIOU_list_clip_overall[video] = miou

            cm_list_fold_overall[ifold] = cm_fold
            # Get the data of the confusion matrix of the classes
            print(' ')
            print("The Fold {} for the class {}".format(ifold+1, i))
            fold_class_cm = cm_fold_class.get_cm()
            # print(classes_cm)
            acc = acc_pixel(fold_class_cm)  # accuracy of the pixels for each class cm
            acc_class = acc_pixel_class(fold_class_cm)  # accuracy of the pixels for each classes cm per each classes
            miou = mean_iou(fold_class_cm)  # calculating the mean IoU for each class
            mIOU_list_fold_class[ifold, i] = miou
            print("Pixel accuracy of the fold is: ", acc)
            print("Pixel accuracy of the {0} class is: ".format(i), acc_class)
            print("The Mean intersection over union of the fold {0} for the class {1} is: ".format(ifold+1, i), miou)

            # Get the data of the confusion matrix of the classes
            print(' ')
            print("The Class {}".format(i))
            classes_cm = cm_class.get_cm()
            # print(classes_cm)
            acc = acc_pixel(classes_cm)  # accuracy of the pixels for each class cm
            acc_class = acc_pixel_class(classes_cm)  # accuracy of the pixels for each classes cm per each classes
            miou = mean_iou(classes_cm)  # calculating the mean IoU for each class
            mIoU_list_class[0, i] = miou
            print("Pixel accuracy of the class is: ", acc)
            print("Pixel accuracy of the {0} class is: ".format(i), acc_class)
            print("The Mean intersection over union of the class {0} is: ".format(i), miou)

    for ifold, valset in enumerate(fold_val):
        if ifold > len(cm_list_fold_overall)-1:
            break
        # Get the data of the confusion matrix of the FOLDS
        print(' ')
        print("The Fold {}".format(ifold+1))
        cm_fold = cm_list_fold_overall[ifold]
        # print(cm_fold)
        fold_cm = cm_fold.get_cm()
        # print(classes_cm)
        acc = acc_pixel(fold_cm)  # accuracy of the pixels for each fold cm
        acc_class = acc_pixel_class(fold_cm)  # accuracy of the pixels for each fold cm per each classes
        miou = mean_iou(fold_cm)  # calculating the mean IoU for each fold
        mIOU_list_fold_overall[ifold] = miou
        print("Pixel accuracy of the fold is: ", acc)
        print("Pixel accuracy of the {0} fold is: ".format(ifold+1), acc_class)
        print("The Mean intersection over union of the fold {0} is: ".format(ifold+1), miou)

    # print(fold_val[0])
    val_list = []
    for ifold, valset in enumerate(fold_val):
        if ifold > len(cm_list_fold_overall)-1:
            break
        else:
            for x in valset:
                clip_list.append(clip_t[x])
                clip_list_path.append(clip_t_path[x])
                val_list.append(x)
    # print(clip_list)
    # print(val_list)

    # Table with all the data
    print(' ')
    classes_name = ['Background', 'Vessels', 'Tool', 'Fetus', 'Overall']
    row = "{:>12}" * (len(classes_name)+1)
    n_row = "{:>12}" + "{:>12.4f}" * (len(classes_name))
    n_row_final = "{:>12}" + "{:>12.4f}" * (len(classes_name)-1)
    print(row.format("Class", *classes_name))
    for i, clip in enumerate(clip_list):
        val = val_list[i]
        print(n_row.format('{}'.format(clip), * mIOU_list_clip_class[val, :], mIOU_list_clip_overall[val]))
    print(n_row_final.format('per Class', * mIoU_list_class[0, :]))

    # Table with all the data
    print(' ')
    classes_name = ['Background', 'Vessels', 'Tool', 'Fetus', 'Overall']
    row = "{:>12}" * (len(classes_name) + 1)
    n_row = "{:>12}" + "{:>12.4f}" * (len(classes_name))
    print(row.format("Class", *classes_name))
    for ifold, valset in enumerate(fold_val):
        print(n_row.format('FOLD{}'.format(ifold+1), *mIOU_list_fold_class[ifold, :], mIOU_list_fold_overall[ifold]))
