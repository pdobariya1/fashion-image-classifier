import torch.nn as nn
import torchvision.models as models

class FashionClassifier(nn.Module):
    def __init__(self, num_colors, num_types, num_seasons, num_genders):
        super(FashionClassifier, self).__init__()
        
        # Load pre-trained Efficientnet_V2_S
        self.backbone = models.efficientnet_v2_s(weights=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Classification heads
        self.color_head = nn.Linear(in_features, num_colors)
        self.type_head = nn.Linear(in_features, num_types)
        self.season_head = nn.Linear(in_features, num_seasons)
        self.gender_head = nn.Linear(in_features, num_genders)
    
    def forward(self, x):
        features = self.backbone(x)
        
        color_out = self.color_head(features)
        type_out = self.type_head(features)
        season_out = self.season_head(features)
        gender_out = self.gender_head(features)
        
        return color_out, type_out, season_out, gender_out