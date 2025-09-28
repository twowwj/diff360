<p align="center">

  <h2 align="center">[CVPR'25]DiffPortrait360: Consistent Portrait Diffusion for 360 View Synthesis</h2>
  <p align="center">
      <a href="https://freedomgu.github.io/">Yuming Gu</a><sup>1,2</sup>
      ·
      <a href="https://p0lyfish.github.io/portfolio/">Phong Tran</a><sup>2</sup>
    ·  
      <a href="https://paulyzheng.github.io/about/">Yujian Zheng</a><sup>2</sup>
    ·  
      <a href="https://hongyixu37.github.io/homepage/">Hongyi Xu</a><sup>3</sup>
    ·  
      <a href="https://lhyfst.github.io/">Heyuan Li</a><sup>4</sup>
    ·  
      <a href="https://www.linkedin.com/in/adilbek-karmanov?originalSubdomain=ae">Adilbek Karmanov</a><sup>2</sup>
    ·  
      <a href="https://hao-li.com">Hao Li</a><sup>2,5</sup>
    <br>
    <sup>1</sup>Unviersity of Southern California &nbsp;<sup>2</sup>MBZUAI &nbsp; <sup>3</sup>ByteDance Inc. &nbsp; 
    <br>
    <sup>4</sup>The Chinese University of Hong Kong, Shenzhen&nbsp; <sup>5</sup>Pinscreen Inc.
    <br>
    </br>
        <a href="https://arxiv.org/abs/2503.15667">
        <img src='https://img.shields.io/badge/arXiv-diffportrait360-green' alt='Paper PDF'>
        </a>
        <a href='https://freedomgu.github.io/DiffPortrait360/'>
        <img src='https://img.shields.io/badge/Project_Page-diffportrait360-blue' alt='Project Page'></a>
        <a href='https://huggingface.co/gym890/diffportrait360'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
     </br>
</p>


https://github.com/user-attachments/assets/eff137b0-359c-4e93-8e69-39ac62792a5b


## 📑 Open-source Plan
- [x] Project Page
- [x] Paper
- [x] Inference code
- [x] Checkpoints of Diffportrait360
- [x] Checkpoints of Back-View Generation Module
- [ ] Training code
- [x] Internet collected inference [Data](https://huggingface.co/gym890/diffportrait360/blob/main/inference_data.zip) (self-collected from [Pexels](https://www.pexels.com/) and 1000 extra real image portraits) 
- [ ] Gradio Demo

-----

This is the official pytorch implementation of Diffportrait360, which generates 360 degree human head through single image portrait.

## **Abstract**
Generating high-quality 360-degree views of human heads from single-view images is essential for enabling accessible immersive telepresence applications and scalable personalized content creation.
While cutting-edge methods for full head generation are limited to modeling realistic human heads, the latest diffusion-based approaches for style-omniscient head synthesis can produce only frontal views and struggle with view consistency, preventing their conversion into true 3D models for rendering from arbitrary angles.
We introduce a novel approach that generates fully consistent 360-degree head views, accommodating human, stylized, and anthropomorphic forms, including accessories like glasses and hats. Our method builds on the DiffPortrait3D framework, incorporating a custom ControlNet for back-of-head detail generation and a dual appearance module to ensure global front-back consistency. By training on continuous view sequences and integrating a back reference image, our approach achieves robust, locally continuous view synthesis. Our model can be used to produce high-quality neural radiance fields (NeRFs) for real-time, free-viewpoint rendering, outperforming state-of-the-art methods in object synthesis and 360-degree head generation for very challenging input portraits.

## **Architecture**

We employs a frozen pre-trained Latent Diffusion Model (LDM) as a rendering backbone and incorporates three auxiliary trainable modules for disentangled control of dual appearance R, camera control C, and U-Nets with view consistency {V}. Specifically, {R} extracts appearance information from {ref} and {back}}, and {C} derives the camera pose, which is rendered using an off-the-shelf 3D GAN. During training, we utilize a continuous sampling training strategy to better preserve the continuity of the camera trajectory. We enhance attention to continuity between frames to maintain the appearance information without changes due to turning angles. For inference, we employ our tailored back-view image generation network {F} to generate a back-view image, enabling us to generate a 360-degree full range of camera trajectories using a single image portrait. Note that $z$ stands for latent space noise rather than image. 

<p align="center">
  <img src="./assets/pipeline.png"  height=400>
</p>



## 📈 Results
### Comparison
To evaluate the dynamics texture generation performance of X-Dyna in human video animation, we compare the generation results of Diffportrait360 with [PanoHead](https://sizhean.github.io/panohead), [SphereHead](https://lhyfst.github.io/spherehead/), [Unique3D](https://wukailu.github.io/Unique3D/).

https://github.com/user-attachments/assets/dfab4cf4-2f14-413b-b0a6-689a2ba40ccb

### Ablation on Dual Appearance Module

https://github.com/user-attachments/assets/db20a7f2-bee0-4cdb-acbb-c6f1d81c5e45

### Ablation on View Consistency

https://github.com/user-attachments/assets/f497ef9e-1ea0-46ae-851b-dce42d597c51

## 📜 Requirements
* An NVIDIA GPU with CUDA support is required. 
  * We have tested on a single A6000 GPU.
  * In our experiment, we used CUDA 12.2.
  * **Minimum**: The minimum GPU memory required is 30GB for generating a single NVS flat video of 32 frames.
  * **Recommended**: We recommend using a GPU with 80GB of memory.
* Operating system: Linux

## 🛠️ Dependencies and Installation

Clone the repository:
```shell
git clone https://github.com/FreedomGu/Diffportrait360
cd diffportrait360_release
```

### Installation Guide

We provide an `env.yml` file for setting up the environment.

Run the following command on your terminal:
```shell
conda env create -n diffportrait360 python=3.9

conda activate diffportrait360

pip install -r requirements.txt
```

## 🧱 Download Pretrained Models
Download the models through this [HF link](https://huggingface.co/gym890/diffportrait360/tree/main).

Change following three model paths (PANO_HEAD_MODEL, Head_Back_MODEL, Diff360_MODEL) to your own download path at ```inference.sh```.
## Training

### Train Back-Head Generation Module
Make sure you have NVIDIA GPU with at least 48GB e.g. A6000.

```bash
cd diffportrait360_release/code

bash train.sh
```

### Train Stage2 NVS Module

Make sure you have NVIDIA GPU with at least 48GB e.g. A6000.
```bash
cd diffpotrait360_release/code

bash train_multiview.sh
```


## Inference
We provide some examples of preprocessed portrait images thought this [script](https://github.com/SizheAn/PanoHead/blob/main/3DDFA_V2_cropping/cropping_guide.md). If you would like to try on your own data, please put your own data under /input_image folder and get ```dataset.json```(camera information under panohead coordinate system) follow with [this folder](https://github.com/SizheAn/PanoHead/tree/main/3DDFA_V2_cropping).
### Using Command Line to 

```bash
cd diffpotrait360_release/code

bash inference.sh
```



### Inference with your own in-the-wild data:

Make sure you get the ```dataset.json``` file and cropped image under /input_image folder.

Run ```bash inference.sh```


## 🔗 BibTeX
If you find [Diffportrait360](https://arxiv.org/abs/2503.15667) is useful for your research and applications, please cite Diffportrait360 using this BibTeX:

```BibTeX
@article{gu2025diffportrait360,
  title={DiffPortrait360: Consistent Portrait Diffusion for 360 View Synthesis},
  author={Gu, Yuming and Tran, Phong and Zheng, Yujian and Xu, Hongyi and Li, Heyuan and Karmanov, Adilbek and Li, Hao},
  journal={arXiv preprint arXiv:2503.15667},
  year={2025}
}
```


## License

Our code is distributed under the Apache-2.0 license.


## Acknowledgements

This work is supported by the Metaverse Center Grant from the MBZUAI Research Office. We appreciate the contributions from [Diffportrait3D](https://github.com/FreedomGu/DiffPortrait3D), [PanoHead](https://github.com/SizheAn/PanoHead), [SphereHead](https://lhyfst.github.io/spherehead/), [ControlNet](https://github.com/lllyasviel/ControlNet) for their open-sourced research. We thank [Egor Zakharov](https://egorzakharov.github.io/), [Zhenhui Lin](https://www.linkedin.com/in/zhenhui-lin-5b6510226/?originalSubdomain=ae), [Maksat Kengeskanov](https://www.linkedin.com/in/maksat-kengeskanov/%C2%A0/), and Yiming Chen for the early discussions, helpful suggestions, and feedback.



## IP Statement
Please contact yuminggu@usc.edu if there has been any misuse of images, and we will promptly remove them.
