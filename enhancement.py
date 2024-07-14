from SRGAN.model import SRResNet
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import torch
from utils import Image2tensor,load_state_dict,denorm
from torchvision.utils import save_image

def Enhancement(resPath, dev):
    device = 'cuda' if dev == "GPU" else 'cpu'
    img = Image2tensor(resPath,process=True,resize=256,device=device)
    g_model = SRResNet(upscale=4)
    g_model = g_model.to(device)
    g_model.eval()
    checkpoint = torch.load("SRGAN/results/pretrained_models/SRGAN_x4-SRGAN_ImageNet.pth.tar", map_location=lambda storage, loc: storage)
    g_model = load_state_dict(g_model,False,checkpoint["state_dict"])
    result = g_model(denorm(img))
    result_path = "temp/temp_enh.png"
    save_image(denorm(result)[0],result_path)
    pixmap = QPixmap(result_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap,result

