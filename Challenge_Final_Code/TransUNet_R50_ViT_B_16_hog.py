import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
from TransUnet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class TransUNet_hog(nn.Module):
    def __init__(self, model):
        super(TransUNet_hog, self).__init__()
        self.transformer = model.transformer
        self.decoder = model.decoder
        self.flatten = nn.Flatten()
        self.segmentation_head = model.segmentation_head
        self.classifier = nn.Sequential(
            nn.Conv2d(2050, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x, hog_vector):
        input_shape = x.shape[-2:]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x) # (B, n_patch, hidden)

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x_patch = x.permute(0, 2, 1)
        x_patch = x_patch.contiguous().view(B, hidden, h, w)

        features = self.flatten(x_patch)
        old_dim = x_patch.size(1)
        features = self.flatten(features)
        merge = torch.cat([features, hog_vector], dim=1)
        new_dim = old_dim + int(hog_vector.size(1) / (h * w))
        x_new = torch.reshape(merge, (features.size(0), new_dim, h, w))

        B, hidden ,h ,w = x_new.size()
        x_new = x_new.contiguous().view(B, int(h*w), hidden)
        x_new = self.decoder(x_new, features)
        logits = self.segmentation_head(x_new)
        return logits



def TransUNet_R50_ViT_B_16_hog(Pre_Trained, progress, n_classes):
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 0  # default
    img_size = 256
    vit_patches_size = 16  # default
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    if Pre_Trained == True:
        model.load_from(weights=np.load(config_vit.pretrained_path))
    model_hog = TransUNet_hog(model)
    return model_hog




