## AI-FACE : Realistic, high-definition, fast, face editing <img src="images\mylogo.png"  style="zoom:67%;" />

<img src="images\fig1.png"  style="zoom:67%;" />

### 1 preparation

##### Environment

Creating a `conda` environment named `AI_FACE`：

```
conda create -n pytorch python=3.9
```

Install the required dependency packages. Given in `requirements.txt.` and `req.txt`.

##### Pretrained models

[StarGANv2](https://github.com/clovaai/stargan-v2).
```
bash download.sh celeba-hq-dataset
bash download.sh pretrained-network-celeba-hq
bash download.sh wing
```

[HiSD](https://github.com/imlixinyang/HiSD).  Put it in `AI-FACE/HiSD/checkpoints`

[SimSwap](https://github.com/neuralchen/SimSwap). Put them in `AI-FACE/SimSwap/arcface_model/`  and  `AI-FACE/SimSwap/checkpoints/` .

[SadTalker](https://drive.google.com/file/d/1gwWh45pF7aelNP_P78uDJL8Sycep-K7j/view). Put them in `AI-FACE/SadTalker/checkpoints`.

[GFPGAN](https://drive.google.com/file/d/19AIBsmfcHW6BRJmeqSFlG5fL445Xmsyi/edit).  Put them in `AI-FACE/gfpgan`.



### 2 Usage

running `run.bat`.

If you need to call it in your script, you can use the following：

```python
from manipulation import manipulate
manipulate(original_path, algorithm, dev, reference_path, reference)
```



### 3 Advanced

#### Super Resolution

In order to provide high-definition editing effects, a fast and efficient high-resolution module is explicitly integrated into the pipeline. Its effects are as follows:

<img src="images\fig2.png"  style="zoom:50%;" />

#### Model distillation and pruning

Efficient model pruning and distillation strategies are used, such as [OMGD](https://github.com/bytedance/OMGD). This significantly reduces computational overhead and accelerates model generation.

<img src="images\fig3.png"  style="zoom:50%;" />



### 4 Visualization

 Different types of editing operations:

<img src="images\fig4.png"  style="zoom:70%;" />

Digital Human Generation:

![vid1](images\vid1.gif)



### 5 Acknowledgements

Our work is based on:

[StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://github.com/clovaai/stargan-v2)

[HiSD: Image-to-image Translation via Hierarchical Style Disentanglement](https://github.com/imlixinyang/HiSD)

[SimSwap: An Efficient Framework For High Fidelity Face Swapping](https://github.com/neuralchen/SimSwap)

[SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation](https://github.com/OpenTalker/SadTalker)

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/Lornatang/SRGAN-PyTorch)

[Towards Real-World Blind Face Restoration with Generative Facial Prior](https://github.com/TencentARC/GFPGAN)

The UI interface is based on:

https://github.com/AcebergChristian/Aceberg_Pro



