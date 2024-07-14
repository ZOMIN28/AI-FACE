import torch
from PIL import Image
from torchvision import transforms as T
import yaml
from torch import nn
from collections import OrderedDict

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    else:
        return x



def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def Image2tensor(imagepath,process=False,resize=256,device='cuda'):
    img = Image.open(imagepath)
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
       transform.append(T.Normalize(mean=0.5, std=0.5))
    if process:
        transform.append(T.Resize([resize,resize]))
    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img),dim=0).to(device)
    return img


def load_state_dict(
        model: nn.Module,
        compile_mode: bool,
        state_dict: dict,
):
    """Load model weights and parameters

    Args:
        model (nn.Module): model
        compile_mode (bool): Enable model compilation mode, `False` means not compiled, `True` means compiled
        state_dict (dict): model weights and parameters waiting to be loaded

    Returns:
        model (nn.Module): model after loading weights and parameters
    """

    # Define compilation status keywords
    compile_state = "_orig_mod"

    # Process parameter dictionary
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # Check if the model has been compiled
    for k, v in state_dict.items():
        current_compile_state = k.split(".")[0]
        if compile_mode and current_compile_state != compile_state:
            raise RuntimeError("The model is not compiled. Please use `model = torch.compile(model)`.")

        # load the model
        if compile_mode and current_compile_state != compile_state:
            name = compile_state + "." + k
        elif not compile_mode and current_compile_state == compile_state:
            name = k[10:]
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Traverse the model parameters, load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model
