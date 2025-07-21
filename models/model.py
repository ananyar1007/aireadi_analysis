# Define EfficientNetB0 with Global Average Pooling
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0()
        self.model.features[-1][0].stride = (1, 1)  # Adjust last layer stride
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(
            1280, num_classes
        )  # 1280 is the feature size in EfficientNetB0

    def forward(self, x):
        x = self.model.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DinoTransformer(nn.Module):

    def __init__(self, **kwargs):
        super(DinoTransformer, self).__init__()
        backbone = "dinov2_small"
        
        backbone_path = {
            'dinov2_small': 'facebook/dinov2-small',
            'dinov2_base': 'facebook/dinov2-base',
            'dinov2_large': 'facebook/dinov2-large',
            'dinov2_giant': 'facebook/dinov2-giant',
            'rad_dino': 'microsoft/rad-dino'
        }

        self.embed_dim_dict = {'dinov2_small': 384, 'dinov2_base': 768, 'rad_dino': 768}
        self.backbone= AutoModel.from_pretrained(backbone_path[backbone])
        self.embed_dim = self.embed_dim_dict[backbone]

        # Learned positional embedding: (1, num_slices, embed_dim)
     #   self.positional_encoding = nn.Parameter(torch.randn(1, num_slices, self.embed_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            batch_first=True
        )


        self.linear = nn.Linear(self.embed_dim, 2)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        # Expand grayscale to RGB
       
      #  x = x.expand(-1, 3, -1, -1, -1)  # (B,3,24,H,W)
        b, c, h, w = x.size()
        #x = torch.reshape(x,(b * t, c, h, w))  # (B*T,3,H,W)
       # print(x.size())
        outputs = self.backbone(pixel_values=x)
      
        out = outputs.last_hidden_state[:, 0, :] 
       # print(out.size())

        output = self.linear(out)  # (B,1)

        return output