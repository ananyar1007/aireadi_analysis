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

from transformers import AutoModel
import torch.nn as nn

class TransformerFusion(nn.Module):
    def __init__(self, num_metadata,num_layers=4, dropout=0.1, **kwargs):
        super(TransformerFusion, self).__init__()
        
        self.num_metadata = num_metadata
        resnet1 = models.resnet18(pretrained=True)
        resnet2 = models.resnet18(pretrained=True)
        self.embed_dim = resnet1.fc.in_features
        self.backbone1 = nn.Sequential(
            *list(resnet1.children())[:-1]   # everything up to the global avgpool
        )
        self.backbone2 = nn.Sequential(
            *list(resnet2.children())[:-1]   # everything up to the global avgpool
        )
        self.emblist = nn.ModuleList()
        for i in range(num_metadata):
            self.emblist.append(nn.Embedding(8,self.embed_dim ))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=4*self.embed_dim,
            dropout=dropout,
            batch_first=True       # so input is (B, seq_len, embed_dim)
        )
        # stack 4 of them
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1,self.embed_dim))
        
        self.linear = nn.Linear(self.embed_dim,2)
        
    def forward(self, input_feat_dct): #clinical data is 2d tensor 
        img_embed_list = [] 
        
        if "ir" in input_feat_dct:
            imglist = input_feat_dct["ir"]
            for img in imglist:
                img_embed = self.backbone1(img)
                img_embed = img_embed.squeeze()
               
                B,_ = img_embed.size()
            
                img_embed = img_embed.reshape((B,1,self.embed_dim))
                img_embed_list.append(img_embed)

        if "cfp" in input_feat_dct:
            imglist = input_feat_dct["cfp"]
            for img in imglist:
                img_embed = self.backbone2(img)
                img_embed = img_embed.squeeze()
               
                B,_ = img_embed.size()
            
                img_embed = img_embed.reshape((B,1,self.embed_dim))
                img_embed_list.append(img_embed)

    
    
        
        meta_embeddings = [] 
        img_embeddings = []
        if "clinical_meta" in input_feat_dct:
            clinical_data = input_feat_dct["clinical_meta"]
            B,_ = clinical_data.size()
            for i in range(self.num_metadata):
                meta_embeddings.append(self.emblist[i](clinical_data[:,i]))
    
            meta_embeddings = torch.cat(meta_embeddings, dim=1)  
            meta_embeddings = meta_embeddings.reshape((B,self.num_metadata, self.embed_dim))
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if "ir" in input_feat_dct or "cfp" in input_feat_dct: #if atleast 1 image modality is used
            img_embeddings = torch.cat(img_embed_list, dim=1) 
            img_embeddings = img_embeddings.reshape((B,len(img_embed_list), self.embed_dim)) 

        
        if len(meta_embeddings) == 0:
            concat_data = torch.cat((cls_tokens, img_embeddings), dim=1)

        elif len(img_embeddings) == 0:
            concat_data = torch.cat((cls_tokens, meta_embeddings), dim=1)

        else: #both images and clinical data are present
            concat_data = torch.cat((cls_tokens, img_embeddings, meta_embeddings), dim=1)
            
        x = self.transformer_encoder(concat_data)
        cls_after_transformer = x[:,0,:]
        x = self.linear(cls_after_transformer)
        return x
        
        
        