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


## Model-level Dataset

## Application-level Dataset

# How to use our pre-trained predictor to predict the model's energy consumption

# How to use our datasets to train your own predictor

# ACKNOWLEDGMENTS

This work was supported by funds from Toyota Motor North America and by the US National Science Foundation (NSF) under Grant No. 1910667, 1910891, and 2025284.

This website template was borrowed from [Nerfies](https://nerfies.github.io).
