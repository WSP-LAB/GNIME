# Learning to Generate Inversion-Resistant Model Explanations

This repository is the official implementation of `Generative Noise Injector for Model Explanations (GNIME)` intoduced in [Learning to Generate Inversion-Resistant Model Explanations](https://github.com). 

![overview](img/overview.png)

<br>

<p float="left">
<img src="img/celeba_lrp.png" width="500" />
<img src="img/celeba_grad.png" width="500" />
<img src="img/celeba_gradcam.png" width="500" />
</p>

<br>
<br>

## Requirements

1. Create and activate virtual environment via `conda`.
```
conda create -n gnime python=3.8
conda activate gnime
```

2. Install `cuda` and `cudnn`. (We use `CUDA 11.3` in our experiment.)
```
conda install cudatoolkit=11.3 cudnn=8.2
```

3. Install requirements
```
pip install -r requirements.txt
```

4. Download `CelebA` dataset from [here](https://zenodo.org/record/6583267#.Yo_GMqhBy4Q).
	- Downloading `MNIST` and `CIFAR-10` datasets are integrated into Python code, so no need to bother.
	- We provide the cropped versions of 1000 identities via `Zenodo`. Move and extract `celeba_cropped_1000.zip` inside `datasets/`. ()
```
datasets
└── celeba_cropped_1000
 	├── 5702
 	|	├── 000707.jpg
	|	├── 002051.jpg
	|	└── ...
	├── 5703
	|	├── 001714.jpg
	│	├── 004273.jpg
	|	└── ...
	└── ...
```

<br>
<br>

## Preparation

1. (optional) Download pretrained models from [here](https://zenodo.org/record/6583267#.Yo_GMqhBy4Q).
	- To avoid the tiresome process of training the victim/attacker DNN models, download the pretrained models uploaded to `Zenodo` and move them to `models/`. The provided source code will automatically detect and use existing models in this directory.
```
models
├── victim
|	├── celeba
|	├── cifar10
|	└── mnist
├── eval
|	├── celeba
|	├── cifar10
|	└── mnist
├── inv
|	├── celeba_predmi
|	├── celeba_expmi_lrp
|	└── ...
└── def
	├── celeba_lrp
	├── celeba_grad
	└── ...
```

2. Pre-generate model explanations into `aux/`.
	- Instead of generating model explanations on demand, we prepare them prehand and access them by their index.
	- We use the `hickle` file format to store and load data.
```
aux
├── celeba_grad
├── celeba_lrp
└── ...
```

<br>

- command for above steps (automatically skips training pretrained models):
```
python prepare.py --target <target_type=['celeba','cifar10','mnist']> --xai <xai_type=['grad','gradcam','lrp']> --gpu <gpu_#>
```


<br>
<br>

## Training
- If you downloaded pretrained models in `Preparation (1)`, you can skip training process and proceed to `Evaluation`.

To train the GNIME defense model, run this command:

```
python train.py --target <target_type=['celeba','cifar10','mnist']> --xai <xai_type=['grad','gradcam','lrp']> --gpu <gpu_#>
```

<br>
<br>

## Evaluation
- visualized inversion results will be saved to `figures/` during the evaluation

To evaluate GNIME defense, run:

```
python eval.py --target <target_type=['celeba','cifar10','mnist']> --xai <xai_type=['grad','gradcam','lrp']> --gpu <gpu_#>
```

For comparison, this will evaluate not only `ExpMI against GNIME`, but also cases for `ExpMI against no defense`, `ExpMI against baseline defense`, and `PredMI`.

Note that evaluation of each scenario requires training a dedicated ExpMI model, hence may take some time. (≤ 1h/scenario)

<br>
<br>

## Results

From our 5 experiements, GNIME defense achieves the following performance: (figure from paper)

![performance](img/eval.png)

## Citation

```
@article{jeong2022learning,
  title={Learning to Generate Inversion-Resistant Model Explanations},
  author={Hoyong Jeong, Suyoung Lee, Sung Ju Hwang, and Sooel Son},
  journal={Advances in Neural Information Processing Systems},
  volume={?},
  pages={?--?},
  year={2022}
}
```
