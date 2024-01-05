# Unveiling Energy Efficiency in Deep Learning

If you find our paper and dataset useful for your work please cite:
```
@inproceedings{tu2023energy,
  author    = {Tu, Xiaolong and Mallik, Anik and Chen, Dawei and Han, Kyungtae and Altintas, Onur and Wang, Haoxin and Xie, Jiang},
  title     = {Unveiling Energy Efficiency in Deep Learning: Measurement, Prediction, and Scoring across Edge Devices},
  booktitle = {Proc. The Eighth ACM/IEEE Symposium on Edge Computing (SEC)},
  pages     = {1--14},
  year      = {2023},
}
```
Explore more details at our website: (https://amai-gsu.github.io/DeepEn2023/)  

# DeepEn 2023 Energy Dataset
DeepEn2023 includes three levels of energy dataset: kernel-level, model-level, and application-level.


## Kernel-level Dataset

| Kernels | CPU Energy Consumption min - max (mJ)| GPU Energy Consumption min - max (mJ)| # Measured kernels(CPU) | # Measured kernels(GPU) | Avg. FLOPs(M) | Configurations |
|---------|---------------------------------------|---------------------------------------|-----------------------|-----------------------|----------------|----------------|
| conv+bn+relu | 0.002 - 1200.083 | 0.002 - 120.152 | 1032 | 1032 | 250.137 | (𝐻𝑊,𝐶𝑖𝑛,𝐶𝑜𝑢𝑡 ,𝐾𝑆, 𝑆) |
| dwconv+bn+relu | 0.022 - 222.609 | 0.016 - 0.658 | 349 | 349 | 28.364 | (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆) |
| bn++relu | 0.002 - 161.334 | 0.001 - 14.594 | 100 | 100 | 4.710 | (𝐻𝑊,𝐶𝑖𝑛)|
| relu | 0.001 - 141.029 | 0.003 - 6.86 | 46 | 46 | 7.983 | (𝐻𝑊,𝐶𝑖𝑛) |
| avgpool | 0.066 - 7.711 | 0.034 - 1.14228 | 28 |  | 0.670 | (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆) |
|maxpool 0.054 - 7.779 0.032 - 1.214 28 28 0.521 (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆)
fc 0.038 - 94.639 - 24 - 14.744 (𝐶𝑖𝑛,𝐶𝑜𝑢𝑡 )
concat 0.001 - 42.826 0.066 - 3.428 142 142 0 (𝐻𝑊,𝐶𝑖𝑛1,𝐶𝑖𝑛2,𝐶𝑖𝑛3,𝐶𝑖𝑛4)
others 0.001 - 132.861 0.003 - 10.163 98 72 - (𝐻𝑊,𝐶𝑖𝑛)

## Model-level Dataset

## Application-level Dataset

# How to use our pre-trained predictor to estimate the model's energy consumption

# How to use DeepEn2023 to train your own predictors

# How to build your own energy consumption datasets

# ACKNOWLEDGMENTS

This work was supported by funds from Toyota Motor North America and by the US National Science Foundation (NSF) under Grant No. 1910667, 1910891, and 2025284.

This website template was borrowed from [Nerfies](https://nerfies.github.io).
