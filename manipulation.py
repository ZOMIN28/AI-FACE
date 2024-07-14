import torch
import warnings
warnings.filterwarnings("ignore", category=Warning)
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
from HiSD.trainer import HiSD_Trainer
from utils import denorm, get_config,Image2tensor
import torch.nn.functional as F
import sys
from SimSwap.options.test_options import TestOptions
from SadTalker import myrun
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import os
sys.path.insert(0, 'SimSwap/')
sys.path.insert(0, 'SadTalker/')


"""
---------------------------------------------------------------
                            HiSD
---------------------------------------------------------------                      
"""
def HiSD_model(device):
    config = get_config('HiSD/configs/celeba-hq_256.yaml')
    checkpoint = 'HiSD/checkpoints/checkpoint_256_celeba-hq.pt'
    trainer = HiSD_Trainer(config)
    state_dict = torch.load(checkpoint)
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    trainer.models.gen.to(device)
    return trainer.models.gen

def processref_HiSD(ref_path,device):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    reference = transform(Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
    return reference

def HiSD_fake(img, reference, model, edit_type):
    if edit_type == "hair color":
        type_num = 2
    else:
        type_num = 1
    with torch.no_grad():
        c = model.encode(img)
        c_trg = c
        s_trg = model.extract(reference , type_num)
        c_trg = model.translate(c_trg, s_trg, type_num)
        gen = model.decode(c_trg)
        return gen


"""
# ---------------------------------------------------------------
#                             starganv2
# ---------------------------------------------------------------                      
# """
def starganv2_model():
    from starganv2.gen import starganv2_Model
    return starganv2_Model()

def processref_starganv2(model,ref_path,ref):
    from starganv2.gen import Processref_starganv2
    with torch.no_grad():
        return Processref_starganv2(model,ref_path,ref)

def starganv2_fake(img,ref,net):
    from starganv2.gen import starganv2_Fake
    with torch.no_grad():
        return starganv2_Fake(img,ref,net)



"""
---------------------------------------------------------------
                            SimSwap
---------------------------------------------------------------                      
"""
from SimSwap.models.models import create_model
def simswap_model(opt,device):
    model = create_model(opt)
    model.eval()
    return model.to(device)

def processorg_simswap(ref_path,device):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256,256]),
    ])
    with torch.no_grad():
        img_a = Image.open(ref_path).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_att = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(device)
    return img_att

def simswap_fake(img_att, img_id, simG):
    with torch.no_grad():
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = simG.netArc(img_id_downsample)
        latend_id = latend_id / torch.norm(latend_id, p=2, dim=1, keepdim=True)
        img_fake = simG(img_id, img_att, latend_id, latend_id, True)
    return img_fake



"""
---------------------------------------------------------------
                            SadTalker
---------------------------------------------------------------                      
"""
def sadtalker_fake(driven_audio,source_image,enhancer,device):
    return myrun.test(driven_audio,source_image,enhancer,device)



def manipulate(original_path, algorithm, dev, reference_path=None, reference="None"):
    device = 'cuda' if dev == "GPU" else 'cpu'
    img = Image2tensor(original_path,process=True,resize=256,device=device)
    if algorithm == "StarGANv2":
        model = starganv2_model()
        ref = processref_starganv2(model,reference_path,reference)
        result = starganv2_fake(img,ref,model)
    elif algorithm == "HiSD":
        model = HiSD_model(device)
        ref = processref_HiSD(reference_path,device)
        result = HiSD_fake(img,ref,model,reference)
    elif algorithm == "SimSwap":
        opt = TestOptions().parse()
        model = simswap_model(opt,device)
        ref = processorg_simswap(reference_path,device)
        result = simswap_fake(ref, img, model)
    elif algorithm == "SadTalker":
        if reference == "None":
            reference = None
        result_path = sadtalker_fake(reference_path,original_path,reference,dev)
        # show the video
        return result_path, result_path 
    result_path = "temp/temp.png"
    save_image(denorm(result)[0],result_path)
    pixmap = QPixmap(result_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap,result