# Covid19-Classification
A deep learning-based web application for classifying Chest X-ray images into:
+ Normal
+ Viral Pneumonia 
+ Covid-19

Built using PyTorch and deployed with Streamlit.

## Description
Early detection of COVID-19 from Chest X-ray images can assist medical professionals 
in rapid screening. This project aims to develop a deep learning model capable of 
classifying X-ray images into three categories.

## Disclaimer
This application is intended for research and educational purposes only.
It is not a medical diagnostic tool.
Always consult a medical professional for clinical decisions.

## Live Demo
https://alexandro-covid19-classification.streamlit.app/

## Training Notebook
https://www.kaggle.com/code/alexandrokalindra/covid-19-classification

## Model Performance Comparison
| Model        | Test Loss | Test Accuracy |
|-------------|-----------|--------------|
| SimpleCNN   | 0.3272    | 80.30%       |
| ResNet18    | 1.0577    | 57.58%       |
| MobileNetV2 | 1.0372    | 54.55%       |

## Getting Started
### Dependencies
* torch >= 2.6.0
* torchvision >= 0.21.0
* streamlit
* Pillow >= 11.3.0
```
pip install -r requirements.txt
```

### Installing 
**Clone this Repository**
```
git clone https://github.com/Alexandro-K/Covid19-Classification.git
cd Covid19-Classification
```

### Executing Program
**Run the main program:**
```
streamlit run app.py
```

## Help
-

## Authors
**Alexandro Kalindra Enggarrinoputra** [Alexandro-K](https://github.com/Alexandro-K)

## Version History
* 0.1
  * Initial Release
 
## License
-

## Acknowledments
**References:**
* COVIDx CXR-4: An Expanded Multi-Institutional Open-Source Benchmark Dataset for Chest X-ray Image-Based Computer-Aided COVID-19 Diagnostics, 29 Nov. 2023, Available: https://arxiv.org/abs/2311.17677
* Deep-Learning-Assisted Highly-Accurate COVID-19 Diagnosis on Lung Computed Tomography Images, 6 Jul.2025, Available: https://arxiv.org/abs/2507.04252
* Prediction of COVID-19 using chest X-ray images, 8 Apr. 2022, Available: https://arxiv.org/abs/2204.03849
* Review of Artificial Intelligence Techniques in Imaging Data Acquisition, Segmentation, and Diagnosis for COVID-19, 16 Apr. 2020, Available: https://ieeexplore.ieee.org/document/9069255
