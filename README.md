# Melanoma Skin Cancer Detection using Deep Learning

This project focuses on developing a deep learning model to detect melanoma skin cancer from medical images. The model utilizes advanced image processing techniques and deep learning architectures to classify skin lesions as either malignant (melanoma) or benign.

## Introduction

Melanoma is one of the most serious forms of skin cancer, and early detection is crucial for successful treatment. This project aims to assist medical professionals by providing an automated system for preliminary skin cancer screening using deep learning techniques. The model processes dermatological images and provides a probability score indicating the likelihood of melanoma.

## Dataset

The project uses the following datasets:

1. **Melanoma Skin Cancer Dataset**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
   - Contains 10,000 images of skin lesions
   - Binary classification: malignant (melanoma) vs. benign
   - Images are in various formats and resolutions

2. **Real-ESRGAN Weights**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/djokester/real-esrgan-weights)
   - Used for image super-resolution
   - Helps in improving image quality and detail


## Project Pipeline

The project implements a deep learning pipeline with the following components:

1. **Image Preprocessing**
   - Image brightening and hair removal
   - Real-ESRGAN super-resolution for enhanced image quality
   - Data augmentation techniques (zoom, flipping, rotating etc.)

2. **Model Architecture**
   - Custom layers for binary classification
   - Transfer learning using pre-trained CNN models
   - Hyperparameter tuning for:
     - Convolutional layers configuration
     - Filter sizes and kernel dimensions
     - Dense layer architecture and units
     - Layer unfreezing strategy
     - Learning rate optimization

3. **Training Pipeline**
   - Model training with optimized hyperparameters
   - Performance evaluation and validation
   - Model saving

### Model Architecture Diagram

![Custom CNN Model Architecture](https://github.com/user-attachments/assets/5bf74386-e72d-4cd6-8089-400dda9814b5)


*Figure 1: Custom CNN Model Architecture showing the layer configuration and data flow*

## Results

### Model Performance Comparison

<table border="1" style="width:100%">
  <tr>
    <th rowspan="2" style="text-align: center">Model</th>
    <th rowspan="2" style="text-align: center">Accuracy</th>
    <th colspan="3" style="text-align: center">Benign</th>
    <th colspan="3" style="text-align: center">Malignant</th>
  </tr>
  <tr>
    <th style="text-align: center">Precision</th>
    <th style="text-align: center">Recall</th>
    <th style="text-align: center">F1-Score</th>
    <th style="text-align: center">Precision</th>
    <th style="text-align: center">Recall</th>
    <th style="text-align: center">F1-Score</th>
  </tr>
  <tr>
    <td style="text-align: center">Custom CNN</td>
    <td style="text-align: center">0.91</td>
    <td style="text-align: center">0.89</td>
    <td style="text-align: center">0.95</td>
    <td style="text-align: center">0.92</td>
    <td style="text-align: center">0.94</td>
    <td style="text-align: center">0.88</td>
    <td style="text-align: center">0.91</td>
  </tr>
  <tr>
    <td style="text-align: center">VGG16</td>
    <td style="text-align: center">0.91</td>
    <td style="text-align: center">0.86</td>
    <td style="text-align: center">0.98</td>
    <td style="text-align: center">0.92</td>
    <td style="text-align: center">0.98</td>
    <td style="text-align: center">0.85</td>
    <td style="text-align: center">0.91</td>
  </tr>
  <tr>
    <td style="text-align: center">ResNet50</td>
    <td style="text-align: center">0.89</td>
    <td style="text-align: center">0.88</td>
    <td style="text-align: center">0.89</td>
    <td style="text-align: center">0.89</td>
    <td style="text-align: center">0.89</td>
    <td style="text-align: center">0.88</td>
    <td style="text-align: center">0.89</td>
  </tr>
  <tr>
    <td style="text-align: center">InceptionV3</td>
    <td style="text-align: center">0.90</td>
    <td style="text-align: center">0.88</td>
    <td style="text-align: center">0.92</td>
    <td style="text-align: center">0.90</td>
    <td style="text-align: center">0.91</td>
    <td style="text-align: center">0.87</td>
    <td style="text-align: center">0.89</td>
  </tr>
</table>

*Table 1: Performance metrics comparison across different model architectures*

### Results Visualization

#### Training History

##### Custom CNN Model History
![Custom CNN Model History](https://github.com/user-attachments/assets/4296867e-38da-481f-ab42-877a0a74aba5)

*Figure 1: Custom CNN Model training history showing accuracy and loss curves*

##### VGG16 Model History
![VGG16 Model History](https://github.com/user-attachments/assets/6a979230-e4c7-48e4-a50f-8ef5ba5e3003)


*Figure 2: VGG16 Model training history showing accuracy and loss curves*

##### ResNet50 Model History
![ResNet50 Model History](https://github.com/user-attachments/assets/d442239f-96be-4390-adca-bdf79e7e5d99)


*Figure 3: ResNet50 Model training history showing accuracy and loss curves*

##### InceptionV3 Model History
![InceptionV3 Model History](https://github.com/user-attachments/assets/7635ab66-508a-48c2-a3f7-bac78d9e26d3)


*Figure 4: InceptionV3 Model training history showing accuracy and loss curves*

#### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/8538e7df-9cca-4259-9eac-d89c2b82af0f)


*Figure 5: Confusion matrix showing model performance on test data*


## Discussion

The developed model shows promising results in detecting melanoma from skin lesion images. Key findings include:
- Effectiveness of transfer learning in medical image classification
- Impact of image preprocessing on model performance
- Importance of data augmentation in handling class imbalance
- Potential for real-world clinical applications

## How to Run the Project

### Running on Kaggle

1. Create a new Kaggle notebook
2. Upload the `melanoma-skin-cancer-detection.ipynb` file
3. Add the required datasets using Kaggle's dataset feature:
   - [Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
   - [Real-ESRGAN Weights](https://www.kaggle.com/datasets/djokester/real-esrgan-weights)
4. Run all cells in sequence

### Running Locally

1. Clone this repository
2. Install required packages:
   ```bash
   pip install numpy pandas torch torchvision Pillow kaggle
   ```
3. Download the datasets from Kaggle:
   - [Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
   - [Real-ESRGAN Weights](https://www.kaggle.com/datasets/djokester/real-esrgan-weights)
4. Open the notebook in Jupyter:
   ```bash
   jupyter notebook melanoma-skin-cancer-detection.ipynb
   ```
5. Run all cells in sequence

## Note

This project is intended for educational and research purposes only. The model's predictions should not be used as a substitute for professional medical diagnosis. Always consult healthcare professionals for medical advice and diagnosis.
