import os
import PIL
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def mean_iou(cm):
    m_iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    m_iou = np.nanmean(m_iou)
    return m_iou

def generating_metrics(prediction_path, arg, data, ifold):
    # print(ifold)
    IoU_list = []

    dataset_folder = arg.dataset_folder

    clip_list = sorted(os.listdir(dataset_folder))
    fold_val = [(0, 5, 12),
                (1, 9, 14),
                (3, 15, 17),
                (2, 4, 11),
                (6, 7, 16),
                (8, 10, 13)]
    clip_t = clip_list
    clip_list = []
    for x in fold_val[ifold]:
        clip_list.append(clip_t[x])
        # print(clip_list)
    for clip in clip_list:
        clip_folder = os.path.join(dataset_folder, clip)
        target_folder = os.path.join(clip_folder, 'labels')
        # print(target_folder)
        target_list = sorted(os.listdir(target_folder))
        for image in target_list:
            # Creating a confusion matrix for the 4 classes: backgroun, vessel, tool, fetus
            # print(image)
            # Opening of the images

            predicition = Image.open(os.path.join(os.path.join(prediction_path, clip), image))
            target = Image.open(target_folder + '/' + image)

            shape = predicition.size
            if target.size != shape:
                target = target.resize(shape, PIL.Image.BILINEAR)

            # IoU
            gt_resized = np.asarray(target)
            pred_resized = np.asarray(predicition)
            mean = Confusion_Matrix(nclasses=4)
            mean.update_cm(gt_resized, pred_resized)
            cm = mean.get_cm()
            IoU_list.append(mean_iou(cm))

    data.append(IoU_list)
    return data

# Box plot
def generating_boxplot(data, saved_folder_path, Architectures_names, type):
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig1, ax1 = plt.subplots()
    box = ax1.boxplot(data, notch=True, patch_artist=True)
    colors = sns.color_palette("hls", len(data))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'Z']
    k = len(Architectures_names)-1
    # print(Architectures_names, len(Architectures_names))
    # print(k)
    # ind = np.arange(k)
    # ax1.set_xticks(ind)
    ax1.set_xticklabels(labels[:k])
    ax1.legend(Architectures_names[1:], loc='lower left', bbox_to_anchor=(-0.15, 0), prop=fontP)
    plt.title("IoU score")
    if type == 'FOLDS':
        plt.savefig(os.path.join(saved_folder_path, 'Final_Folds_Metric_IoU.png'))
    if type == 'ARCHITECTURES':
        plt.savefig(os.path.join(saved_folder_path, 'Final_Archs_Metric_IoU.png'))

# Valori numerici
def generating_table(data, Architectures_names):
    IoU_median_list = []
    IoU_mean_list = []

    for list in data:
        IoU_median = np.median(list)
        IoU_mean = np.mean(list)
        IoU_median_list.append(IoU_median)
        IoU_mean_list.append(IoU_mean)
    # print(len(data))
    # print(len(IoU_mean_list))
    row = "{:>12}" * (len(Architectures_names))
    n_row = "{:>12}" + "{:>12.4f}" * (len(Architectures_names)-1)
    print(' ')
    print('Intersection Over Union')
    print(row.format(*Architectures_names))
    print(n_row.format('Mean', *IoU_mean_list))
    print(n_row.format('Median', *IoU_median_list))


def boxplot_fold(arg):
    print(' ')
    print('boxplot fold')
    results_folder_fetreg = os.path.join(arg.main_folder, 'RESULTS_CHALLENGE')
    if arg.pre_trained == 'True':
        saved_folder_path = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name + '_pre_trained')
    else:
        saved_folder_path = os.path.join(results_folder_fetreg, 'RESULTS_' + arg.model_name)
    result_list = sorted(os.listdir(saved_folder_path))   # fold list
    # print(result_list)
    data_list = []
    fold_names = [arg.model_name]
    for arch in result_list:
        if arch[0] == 'R':
            # RESULTS_FOLD_01
            name = arch[8] + arch[14]
            fold_names.append(name)
    for iresult, result in enumerate(result_list):
        # print(result, iresult-1)
        result_fold_path = os.path.join(saved_folder_path, result) # fold path
        final_result_path = os.path.join(result_fold_path, 'Predicted_Masks_Original_{}'.format(arg.epochs)) # prediction path
        if os.path.isdir(final_result_path):
            data_list = generating_metrics(final_result_path, arg, data_list, iresult-1)

    generating_boxplot(data_list, saved_folder_path, fold_names, 'FOLDS')
    generating_table(data_list, fold_names)



def boxplot_architectures(arg):
    print(' ')
    print('boxplot architectures')
    saved_folder_path = os.path.join(arg.main_folder, 'RESULTS_CHALLENGE')
    result_list = sorted(os.listdir(saved_folder_path))  # list of architectures
    # print(result_list)
    data_list = []
    architectures_names = ['Architectures: ']
    for arch in result_list:
        # print(arch)
        # RESULTS_FCN_RESNET_HOG_PRE_TRAINED
        # RESULTS_FCN_RESNET_HOG
        # RESULTS_TransUNet_R50+ViT-B_16_hog_PRE_TRAINED
        # RESULTS_deeplabv3_Resnet_hog
        if arch[8:11] == 'FCN':
            name = arch[8] + arch[12] + arch[19]
            if len(arch) > 22:
                name = name + arch[23] + arch[27]
            architectures_names.append(name)
        if arch[8] == 'd':
            name = arch[8] + arch[18] + arch[25]
            if len(arch) > 28:
                name = name + arch[29] + arch[33]
            architectures_names.append(name)
        if arch[8] == 'T':
            name = arch[8] + arch[13] + arch[31]
            if len(arch) > 34:
                name = name + arch[35] + arch[39]
            architectures_names.append(name)
    # print(architectures_names)
    for result in result_list:
        if result[0] == 'R':
            # print(result)
            result_arch_path = os.path.join(saved_folder_path, result)
            fold_list = sorted(os.listdir(result_arch_path))
            if fold_list[0] == 'RESULTS_FOLD_01':
                result_fold_path = os.path.join(result_arch_path, fold_list[0])
            else:
                result_fold_path = os.path.join(result_arch_path, fold_list[1])
            final_result_path = os.path.join(result_fold_path, 'Predicted_Masks_Original_{}'.format(arg.epochs))
            # print(final_result_path)
            if os.path.isdir(final_result_path):
                data_list = generating_metrics(final_result_path, arg, data_list, 0)

    generating_boxplot(data_list, saved_folder_path, architectures_names, 'ARCHITECTURES')
    generating_table(data_list, architectures_names)
