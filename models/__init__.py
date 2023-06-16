import torch

from .densenet import densenet169
from .nbsnet import get_conv, NbsCls

MODEL_DICT = {
    "densenet169": [densenet169, "avgpool", 1664],
}

def _get_model(name, model_type, num_classes, in_channels, f_extractor=None, dropout_rate=0.0):
    backbone, return_layer, in_feat = MODEL_DICT[name]
    if model_type == "nbs":
        classifier = NbsCls(in_feat, num_classes)
    else:
        classifier = torch.nn.Linear(in_feat, num_classes)
        classifier.num_classes = num_classes
    if backbone:
        return get_conv(
            backbone(in_channels=in_channels),
            return_layer,
            classifier,
            model_type,
            dropout_rate,
        )

    else:
        return classifier