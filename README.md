# WCDAS

The Pytorch implementation for the paper titled Wrapped Cauchy Distributed Angular Softmax for Long-Tailed Visual Recognition.

## Abstract

Visual recognition is vital for various computer vision applications. However, imbalanced or long-tailed data pose significant challenges to the deep learning approaches due to the mismatch between training and testing distributions. Our paper presents a novel softmax function based on wrapped Cauchy distribution: Wrapped Cauchy Distributed Angular Softmax (WCDAS). WCDAS considers the data-wise Gaussian-based kernels in the angular representation between features and classifier weights, describing noise and sparse sampling-induced uncertainty. As the class-wise distribution of such angular representation follows the sum of the kernels, we prove theoretically that the wrapped Cauchy distribution can be a better approximation for such mixed distributions than the widely-used Gaussian distribution. We demonstrate that WCDAS can dynamically optimize the compactness/margin of each class via the corresponding trainable concentration parameters. The empirical study shows that such class-wise parameters of WCDAS exhibit label-aware behavior. WCDAS outperforms other state-of-the-art softmax-based methods in long-tailed visual recognition on several benchmark datasets.

## Usage

### Training
The algorithm is simply implemented in the class `WCDAS` at `models/Loss.py`.

To reproduce the results in the paper, the ResNet-10 equipped with the WCDAS is trained from scratch on ImageNet-LT dataset by
```bash
python main_train.py --dataset imagenetlt --net-config ResNet10Feature --workers 12 --seed 0 --loss-config WCDAS_ImageNetLT 
python main_finetune.py --dataset imagenetlt --net-config ResNet10Feature_finetune --loss-config WCDAS_ImageNetLT --model-file ./results/imagenetlt_loss_WCDAS_ImageNetLT_ResNet10Feature_lr_0.4_model/ --workers 12 --seed 0  
```

For iNaturalist-2018, the script to train from script is as follows:

```bash
python main_train.py --dataset 'inat2018' --net-config ResNet50Feature --workers 12 --seed 0 --loss-config WCDAS_iNaturalist2018
python main_finetune.py --dataset 'inat2018' --net-config ResNet50Feature_finetune --loss-config WCDAS_iNaturalist2018 --model-file ./results/imagenetlt_loss_WCDAS_iNaturalist2018_ResNet50Feature_lr_0.4_model/ --workers 12 --seed 0  
```
## Results

#### ImageNet

| Method  | ImageNet-LT | iNaturalist-2018 |
|---|---|---|
| WCDAS | 44.5 | 71.8|

## References

[1] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng and Yannis Kalantidis. "DECOUPLING REPRESENTATION AND CLASSIFIER FOR LONG-TAILED RECOGNITION." In ICLR, 2020.

[2] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga and Tengyu Ma. "Learning imbalanced datasets with label-distribution-aware margin loss." In NeurIPS, 2019.


## Contact
takumi.kobayashi (At) aist.go.jp


## Acknowledgement
The class-wise sampler `utils/ClassAwareSampler.py` is from the [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing).