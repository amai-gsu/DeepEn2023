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

# DeepEn2023 Energy Datasets
DeepEn2023 includes three levels of energy dataset: kernel-level, model-level, and application-level.


## Kernel-level Dataset
**Kernel-level dataset summary**. We generate thousands of kernel models, deploy them into different edge devices, measure the energy consumption during the model execution. The number of kernels is based on how often this kernel appears in models. 
| Kernels | CPU Energy Consumption min - max (mJ)| GPU Energy Consumption min - max (mJ)| # Measured kernels(CPU) | # Measured kernels(GPU) | Avg. FLOPs(M) | Configurations |
|---------|---------------------------------------|---------------------------------------|-----------------------|-----------------------|----------------|----------------|
| conv++bn++relu | 0.002 - 1200.083 | 0.002 - 120.152 | 1032 | 1032 | 250.137 | (𝐻𝑊,𝐶𝑖𝑛,𝐶𝑜𝑢𝑡 ,𝐾𝑆, 𝑆) |
| dwconv++bn++relu | 0.022 - 222.609 | 0.016 - 0.658 | 349 | 349 | 28.364 | (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆) |
| bn++relu | 0.002 - 161.334 | 0.001 - 14.594 | 100 | 100 | 4.710 | (𝐻𝑊,𝐶𝑖𝑛)|
| relu | 0.001 - 141.029 | 0.003 - 6.86 | 46 | 46 | 7.983 | (𝐻𝑊,𝐶𝑖𝑛) |
| avgpool | 0.066 - 7.711 | 0.034 - 1.14228 | 28 | 28 | 0.670 | (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆) |
|maxpool | 0.054 - 7.779 | 0.032 - 1.214 | 28 | 28 | 0.521 | (𝐻𝑊,𝐶𝑖𝑛,𝐾𝑆, 𝑆) |
| fc  | 0.038 - 94.639 | - | 24 | - | 14.744 | (𝐶𝑖𝑛,𝐶𝑜𝑢𝑡 ) |
| concat | 0.001 - 42.826 | 0.066 - 3.428 | 142 | 142 | 0 | (𝐻𝑊,𝐶𝑖𝑛1,𝐶𝑖𝑛2,𝐶𝑖𝑛3,𝐶𝑖𝑛4) |
| others | 0.001 - 132.861 | 0.003 - 10.163 | 98 | 72 | - | (𝐻𝑊,𝐶𝑖𝑛) |
* HW: Height/Width, Cin: Channel Input, Cout: Channel Output, KS: Kernel Size, S: Stride

## Model-level Dataset
**Model-level dataset summary**. Our model-level energy dataset includes nine SOTA DNN models. These models represent a mix of both manually-designed and NAS-derived models, each with distinct kernel types and configurations. For each model, we generate 50 variants for conducting power and energy measurements by re-sampling the 𝐶𝑜𝑢𝑡 and 𝐾𝑆 for each layer.
| Models       | Energy Consumption (mJ) CPU min - max | Energy Consumption (mJ) GPU min - max | Avg. FLOPs (M) |
|--------------|--------------------------------------|--------------------------------------|----------------|
| AlexNets     | 36.97 - 355.58                      | 7.69 - 91.80                        | 815            |
| DenseNets    | 231.93 - 488.87                     | 66.21 - 133.58                      | 1760           |
| GoogleNets   | 145.03 - 262.45                     | 52.66 - 90.04                       | 1535           |
| MobileNetv1s | 53.59 - 136.79                      | 17.36 - 42.44                       | 519            |
| MobileNetv2s | 30.85 - 175.07                      | 8.81 - 48.35                        | 419            |
| ProxylessNas | 58.34 - 162.11                      | 17.70 - 49.29                       | 526            |
| ResNet18s    | 251.52 - 1432.67                    | 64.19 - 391.97                      | 3888           |
| ShuffleNetv2s| 25.26 - 81.41                       | -                                    | 319            |
| SqueezeNets  | 92.55 - 388.16                      | 34.55 - 134.65                      | 1486           |

## Application-level Dataset
**Application-level dataset summary**. Application-level dataset demonstrates the end-to-end energy consumption of six popular edge AI applications, covering three main categories: vision-based (object detection, image classification, super resolution, and image segmentation), NLP-based (natural language question answering), and voice-based applications (speech recognition).
| Category    | Application                 | No.  | Reference DNN models                      | CPU1 | CPU4 | GPU | NNAPI | Model size (MB) |
|-------------|-----------------------------|------|-------------------------------------------|------|------|-----|-------|-----------------|
| Vision-based| Image detection             | DNN1 | MobileNetv2, FP32, 300 x 300 pixels      | ✔️    |      | ✔️   |       | 24.2            |
|             |                             | DNN2 | MobileNetv2, INT8, 300 x 300 pixels      | ✔️    |      | ✔️   |       | 6.9             |
|             |                             | DNN3 | MobileNetv2, FP32, 640 x 640 pixels      | ✔️    |      | ✔️   |       | 12.3            |
|             |                             | DNN4 | MobileNetv2, INT8, 640 x 640 pixels      | ✔️    |      | ✔️   |       | 4.5             |
|             | Image classification        | DNN5 | EfficientNet, FP32, 224 x 224 pixels     | ✔️    |      | ✔️   |       | 18.6            |
|             |                             | DNN6 | EfficientNet, INT8, 224 x 224 pixels     | ✔️    |      | ✔️   |       | 5.4             |
|             |                             | DNN7 | MobileNetv1, FP32, 224 x 224 pixels      | ✔️    |      | ✔️   |       | 4.3             |
|             |                             | DNN8 | MobileNetv1, INT8, 224 x 224 pixels      | ✔️    |      | ✔️   |       | 16.9            |
|             | Super resolution            | DNN9 | ESRGAN, FP32, 50 x 50 pixels        |      |      |     |       | 5               |
|             | Image segmentation          | DNN10| DeepLabv3, FP32, 257 x 257 pixels   | ✔️    |      | ✔️   |       | 2.8             |
| NLP-based   | Natural language question answering | DNN11 | MobileBERT, FP32                  | ✔️    |      |     | ✔️     | 100.7           |
| Voice-based | Speech recognition          | DNN12| Conv-Actions-Frozen, FP32           |      |      | ✔️   |       | 3.8             |

## How to download DeepEn2023 datasets
- Go to our project website: https://amai-gsu.github.io/DeepEn2023/.
- Click "**Data**".
- Submit the User Survey.
- After you submit the survey, a download link will appear.

(Currently, we have only uploaded kernel-level dataset. we will upload other datasets later.)

# How to use our pre-trained predictors to estimate the model's energy consumption
- Download pre-trained predictors by following the steps outlined in [How to download DeepEn2023 datasets](#how-to-download-deepen2023-datasets).
- Prepare the models you wish to predict. Ensure the models are in ONNX format (The format is just for run the example). We provide test models in **Model_test** folder.
- Download the code: **sec23_AIEnergy_onnx.py**. You can find it in Prediction folder.
- Remmber to change the path to your project location.
```Bash
def main():
    opt = arg_parser()
    # load predictor
    predictor_name = opt.predictor
    predictor_version = float(opt.version)
    workplace = "/Users/xiaolong/Library/CloudStorage/Dropbox-GSUDropbox/Xiaolong_Tu/sec23_result/Dataset_P40p/Training/"
    # workplace = "/home/haoxin/Downloads/Training/"
    predictor_folder = os.path.join(workplace, opt.purpose, "predictor")
    # print(predictor_folder)
```
- Run the command.
```Bash
python sec23_AIEnergy_onnx.py --predictor TestcortexA76cpu_tflite --version 1.0 --purpose Energy --modelpath Test_models/
# TestcortexA76cpu_tflite. The folder for all the predictors.
# --version 1.0. The predictor version, confirm it in predictors.yaml
# --purpose Energy. Which one you want to predict. latency, power or energy.
# --modelpath Test_models/. The path for the models which you wish to predict.
```
# How to use DeepEn2023 to train your own predictors
Coming soon by Jan. 31, 2024
# How to build your own energy consumption datasets
Coming soon by Jan. 31, 2024
# ACKNOWLEDGMENTS

This work was supported by funds from Toyota Motor North America and by the US National Science Foundation (NSF) under Grant No. 1910667, 1910891, and 2025284.

This website template was borrowed from [Nerfies](https://nerfies.github.io).
