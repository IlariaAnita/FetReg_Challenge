import Training_pre_trained
import Training
import Testing
import Evaluation_Segmentation_Final
import Boxplot
# from Challenge_Final_Code import Training
# from Challenge_Final_Code import Testing
# from Challenge_Final_Code import Evaluation_Segmentation_Final
# from Challenge_Final_Code import Boxplot
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main_folder", help="Main Folder of the project", required=True)
    parser.add_argument("--dataset_folder", help="Path to DATASET folder", required=True)
    parser.add_argument("--model_name", help="Name of the architecture", required=True)
    parser.add_argument("--num_classes", help="Number of classes", default=4)
    parser.add_argument('--pre_trained', help='Boolean value: True = load weights, False = no load, weights', required=True)
    parser.add_argument('--cross_fold_validation', help='Boolean value: True = apply CFV, False = make only one fold', required=True)
    parser.add_argument('--epochs', help='Number of epochs to train the ANN', required=True)
    parser.add_argument('--loss_name', help='Name of the loss used', required=True)
    args = parser.parse_args()


    if args.pre_trained == 'True':
        print('Pre Trained')
        model_hog = Training_pre_trained.training(args)
    else:
        print('No Pre Trained')
        model_hog = Training.training(args)
    Testing.testing(args, model_hog)
    Evaluation_Segmentation_Final.evaluation_segmentation(args)
    if args.cross_fold_validation == 'True':
        Boxplot.boxplot_fold(args)
    saved_folder_path = os.path.join(args.main_folder, 'RESULTS_CHALLENGE')
    result_list = sorted(os.listdir(saved_folder_path))  # list of architectures
    if len(result_list) > 1:
        Boxplot.boxplot_architectures(args)



# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="FCN_ResNet_hog" --pre_trained=True --cross_fold_validation=True --epochs=300 --loss_name="CrossEntropy"  > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/FCN_ResNet_hog_pre_trained.txt
# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="FCN_ResNet_hog" --pre_trained=False --cross_fold_validation=False --epochs=300 --loss_name="CrossEntropy" > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/FCN_ResNet_hog.txt

# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="deeplabv3_Resnet_hog" --pre_trained=False --cross_fold_validation=False --epochs=300 --loss_name="CrossEntropy" > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/deeplabv3_Resnet_hog.txt
# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="deeplabv3_Resnet_hog" --pre_trained=True --cross_fold_validation=False --epochs=300 --loss_name="CrossEntropy" > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/deeplabv3_Resnet_hog_pre_trained.txt

# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="TransUNet_R50+ViT-B_16_hog" --pre_trained=False --cross_fold_validation=False --epochs=300 --loss_name="CrossEntropy" > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/TransUNet_R50+ViT-B_16_hog.txt
# python Challenge_Final_Code/Challenge_Main.py --main_folder="/home/nearlab/poliChallenge" --dataset_folder="/home/nearlab/poliChallenge/FetReg2021_Task1_Segmentation" --model_name="TransUNet_R50+ViT-B_16_hog" --pre_trained=True --cross_fold_validation=False --epochs=300 --loss_name="CrossEntropy"  > /home/nearlab/poliChallenge/RESULTS_CHALLENGE/TransUNet_R50+ViT-B_16_hog_pre_trained.txt