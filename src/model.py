import timm
import torch
import torch.nn as nn

class MultiTaskViT(nn.Module):
    def __init__(self, backbone_name, num_classes, attr_schema, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # no classifier
        )
        feat_dim = self.backbone.num_features
        
        # Class head
        self.class_head = nn.Linear(feat_dim, num_classes)
        
        # Attribute heads: one softmax per attribute
        self.attr_heads = nn.ModuleDict()
        self.attr_sizes = {}
        for attr_name, values in attr_schema.items():
            self.attr_sizes[attr_name] = len(values)
            self.attr_heads[attr_name] = nn.Linear(feat_dim, len(values))
    
    def forward(self, x):
        feats = self.backbone(x)          # (B, feat_dim)
        class_logits = self.class_head(feats)
        attr_logits = {name: head(feats) for name, head in self.attr_heads.items()}
        return feats, class_logits, attr_logits
