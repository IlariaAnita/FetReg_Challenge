import torch
import torchvision
from torch import nn
from torch.nn import functional as F

class ResNet_hog(nn.Module):
    def __init__(self, model):
        super(ResNet_hog, self).__init__()
        self.encoder = model.backbone
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Conv2d(2050, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x, hog_vector):
        input_shape = x.shape[-2:]
        features = self.encoder(x)['out']
        old_dim = features.size(1)

        features = self.flatten(features)
        # hog_vector = hog_vector.float()
        merge = torch.cat([features, hog_vector], dim=1)
        # merge = merge.float()

        new_dim = old_dim + int(hog_vector.size(1) / (32 * 32))

        output_latent = torch.reshape(merge, (features.size(0), new_dim, 32, 32))
        # output_latent = output_latent.float()
        output = self.classifier(output_latent)
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        return output


def FCN_ResNet_hog(Pre_Trained, progress, n_classes):
    if Pre_Trained == True:
        model = torchvision.models.segmentation.fcn_resnet50(Pre_Trained, progress, num_classes=21)
        if n_classes != 21:
            model = torchvision.models.segmentation.fcn_resnet50(Pre_Trained, progress)
            model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)],
                                             nn.Conv2d(512, 4, kernel_size=1, stride=1, bias=False))
            model.aux_classifier = nn.Sequential(*[model.aux_classifier[i] for i in range(4)],
                                                 nn.Conv2d(512, 4, kernel_size=1, stride=1, bias=False))
    else:
        model = torchvision.models.segmentation.fcn_resnet50(Pre_Trained, progress, num_classes=n_classes)
    model_hog = ResNet_hog(model)
    return model_hog
