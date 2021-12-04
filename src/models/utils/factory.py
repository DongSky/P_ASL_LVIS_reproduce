import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL
#from ..swin_transformer import SwinS, SwinB, SwinL
import torchvision.models as models
import torch.nn as nn
def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    elif args.model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, args.num_classes, bias=True)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
