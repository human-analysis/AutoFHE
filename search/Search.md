# AutoFHE

## Prepare Python Environment

```bash
conda create -n AutoFHE python=3.8
conda activate AutoFHE
pip install -r requirements.txt
```

## Train Model

```bash
python train.py -a resnet20 -d cifar10 -o experiments --devices 0 --data ../../Datasets

python train.py -a resnet32_aespa -d cifar100 -o experiments --devices 1 --data ../../Datasets --grad-clip 1
```
## Run AutoFHE Search 

```bash
python search.py -a resnet20 -d cifar10 --data ../../Datasets --gpu 0 --ckpt experiments/resnet20-cifar10/last.ckpt 
```

## Finetune AutoFHE Networks

```bash
python tune.py -a resnet20 -d cifar10 --data ../../Datasets --backbone-ckpt experiments/resnet20-cifar10/last.ckpt --gpu 0 --ckpt experiments-search/resnet20-cifar10/pareto/0.ckpt
```

## Write Model Weights to txt

```bash
python model2txt.py -a resnet20_autofhe -d cifar10 -o weights --data ../../Datasets --ckpt experiments-tune/resnet20-cifar10/boot11_acc92.91.ckpt
```

