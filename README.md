# [NeurIPS 2024] VFIMamba: Video Frame Interpolation with State Space Models [arxiv](https://arxiv.org/abs/2407.02315)

> [**VFIMamba: Video Frame Interpolation with State Space Models**](https://arxiv.org/abs/2407.02315)<br>
> [Guozhen Zhang](https://github.com/GuozhenZhang1999), [Chunxu Liu](https://scholar.google.com.hk/citations?hl=zh-CN&view_op=list_works&gmla=AKKJWFe0ZBvfA_4yxMRe8BW79xNafjCwXtxN10finOaqV1EREnZGxSX6DbpZelBUJD0GZmp5S7unCf76xrgOfnS6SVA&user=dvUKnKEAAAAJ), [Yutao Cui](https://scholar.google.com.hk/citations?user=TSMchWcAAAAJ&hl=zh-CN), Xiaotong Zhao, [Kai Ma](https://scholar.google.com.hk/citations?user=FSSXeyAAAAAJ&hl=zh-CN), [Limin Wang](http://wanglimin.github.io/)
<div align="center">
  <img src="figs/main0.png" width="1000"/>
</div>

## :boom: News

* **[2024.09.26] Accepted by NeurIPS 2024!**
* **[2024.09.3] Support for directly importing model weights from HuggingFace. Thanks to the HuggingFace team for their efforts!**
* **[2024.07.3] Demo and evaluation codes released.**

## :satisfied: HighLights

In this work, we have introduced VFIMamba, the first approach to adapt the SSM model to the video frame interpolation task. We devise the Mixed-SSM Block (MSB) for efficient inter-frame modeling using S6. We also explore various rearrangement methods to convert two frames into a sequence, discovering that interleaved rearrangement is more suitable for VFI tasks. Additionally, we propose a curriculum learning strategy to further leverage the potential of the S6 model. Experimental results demonstrate that VFIMamba achieves the state-of-the-art performance across various datasets, in particular highlighting the potential of the SSM model for VFI tasks with high resolution.

<div align="center">
  <img src=figs/main.png width=800 />
</div>

## :two_hearts:Installation
```bash
conda create -n VFIMamba python=3.11
conda activate VFIMamba

# conda 會自動安裝相容的 CUDA toolkit
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip uninstall torch torchvision torchaudio -y
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 安裝其他依賴
# causal_conv1d-1.0.0
# pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.0.0/causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# mamba_ssm-1.0.1
# pip install https://github.com/state-spaces/mamba/releases/download/v1.0.1/mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

pip install scikit-image==0.19.2
pip install opencv-python timm tqdm tensorboard

--------------------------------------------
pip install "numpy<2"
pip install transformers==4.30.0
# 移除預編譯的 mamba_ssm
pip uninstall mamba_ssm causal_conv1d -y


# 修改 mamba/setup.py 以避免記憶體超過用量
def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "1"]
------------------------------------------

# 下載模型檔
wget https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl
wget https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba_S.pkl
```

參考 https://github.com/MzeroMiko/VMamba/issues/378


CUDA 11.7
- torch 1.13.1
- python 3.10.6
- causal_conv1d 1.0.0
- mamba_ssm 1.0.1
- skimage 0.19.2
- numpy 
- opencv-python 
- timm 
- tqdm
- tensorboard

## :sunglasses:	Play with Demos

1. Download the [model checkpoints](https://huggingface.co/MCG-NJU/VFIMamba_ckpts/tree/main) and put the ```ckpt``` folder into the root dir. We also support directly importing model weights from HuggingFace. Please refer to hf_demo_2x.py.
2. Run the following commands to generate 2x and Nx (arbitrary) frame interpolation demos:

We provide two models, an efficient version (VFIMamba-S) and a stronger one (VFIMamba). 
You can choose what you need by changing the parameter ```model```.

### Hugging Face Demo
For Hugging Face demo, please refer to [the code here](https://github.com/MCG-NJU/VFIMamba/blob/main/hf_demo_2x.py).
```bash
python hf_demo_2x.py --model **model[VFIMamba_S/VFIMamba]**      # for 2x interpolation
```

### Manually Load
```shell
python demo_2x.py  --model **model[VFIMamba_S/VFIMamba]**      # for 2x interpolation
python demo_Nx.py --n 8 --model **model[VFIMamba_S/VFIMamba]** # for 8x interpolation
# 整段影片生成
python video_predict.py --model VFIMamba_S
```

By running above commands with model VFIMamba, you should get the follow examples by default:

<p float="left">
  <img src=figs/out_2x.gif width=340 />
  <img src=figs/out_8x.gif width=340 /> 
</p>

You can also use the ```scale``` parameter to improve performance at higher resolutions; We will downsample to ```scale```*shape to predict the optical flow and then resize to the original size to perform the other operations. We recommend setting the ```scale``` to 0.5 for 2K frames and 0.25 for 4K frames.

```shell
python demo_2x.py  --model VFIMamba --scale 0.5 # for 2K inputs with VFIMamba   
```

## :runner:	Evaluation

1. Download the dataset you need:

   * [Vimeo90K dataset](http://toflow.csail.mit.edu/)
   * [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow)
   * [Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py)
   * [SNU-FILM dataset](https://myungsub.github.io/CAIN/)
   * [X4K1000FPS dataset](https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0)

2. Download the [model checkpoints](https://huggingface.co/MCG-NJU/VFIMamba_ckpts/tree/main) and put the ```ckpt``` folder into the root dir. We also support directly importing model weights from HuggingFace. Please refer to hf_demo_2x.py.

For all benchmarks:

```shell
python benchmark/**dataset**.py --model **model[VFIMamba_S/VFIMamba]** --path /where/is/your/**dataset**
```

You can also test the inference time of our methods on the $H\times W$ image with the following command:

```shell
python benchmark/TimeTest.py --model **model[VFIMamba_S/VFIMamba]** --H **SIZE** --W **SIZE**
```

## :muscle:	Citation

If you think this project is helpful in your research or for application, please feel free to leave a star⭐️ and cite our paper:

```
@misc{zhang2024vfimambavideoframeinterpolation,
      title={VFIMamba: Video Frame Interpolation with State Space Models}, 
      author={Guozhen Zhang and Chunxu Liu and Yutao Cui and Xiaotong Zhao and Kai Ma and Limin Wang},
      year={2024},
      eprint={2407.02315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.02315}, 
}
```

## :heartpulse:	License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [EMA-VFI](https://github.com/whai362/PVT), [MambaIR](https://github.com/csguoh/MambaIR?tab=readme-ov-file#installation) and [SGM-VFI](https://github.com/MCG-NJU/SGM-VFI). Please also follow their licenses. Thanks for their awesome works.
