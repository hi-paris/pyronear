## 📝 Table of Contents
- [About the Project](#about-the-project)
  
- [Image Preprocessing](#image-preprocessing)
  - [Getting Started](#getting-started)
  - [Import Library](#prerequisites)
  - [Installation : Clone the Repository ⚙️](#installation)
- [Super Resolution](#super-resolution)
  - [From Scratch](#From-scratch)
  - [Pretrained](#pretrained)
- [Hyper parameters tunnning](#hyper-parameters-tunning)
  - [Grid Search For Yolo model](Grid-search)
    
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

![image](https://github.com/hi-paris/pyronear/assets/108234726/cb928a41-6af0-4714-b85a-1a4d5b096c64)

## 🌐 About the Project

This project incorporates a variety of advanced image processing techniques, specifically designed to enhance the capabilities of the Pyronear project for forest fire detection. Here’s an overview of what each part of the project involves:

- **Image Preprocessing : Transformation and Augmentation**: These techniques artificially expand the dataset with modified versions of existing images through operations such as rotations, scaling, and flipping. This is crucial for training robust machine learning models. The pipeline can make rotation on the images but also the labels.

- **Super-resolution**:Enhances the resolution of input images, which is particularly beneficial for improving the quality of images in the Roboflow dataset that includes video game captures from low-resolution devices in forested areas. This improvement helps make our models more robust, enhancing their ability to detect smoke and other indicators of forest fires effectively

- **Hyperparameter Tuning**: Optimizes the performance of machine learning models by systematically searching for the most effective combination of parameters. This process ensures that the models perform optimally under various conditions.
Here, we are going to use Random search.

- **Data augmentation with albumentation** : 


- **Training yolov10** : 


These components are integrated into the Pyronear project to enhance its effectiveness in detecting forest smoke, ultimately aiming to provide faster and more reliable fire detection solutions.

---

## **1️⃣ Image Preprocessing**:
This file contains code for data transformation and augmentation, including rotations and transformations on images and labels.

This section explains how to use the image preprocessing pipeline. Ensure you have followed the installation instructions provided in the [Getting Started](#Getting-Started) section before proceeding.

#### Running the Image Preprocessing Pipeline

The image preprocessing pipeline is designed to transform and augment images using specific operations such as rotations. This functionality is encapsulated in the `process_image` function, which is part of the `processing` module.

#### Setting Up and Running the Pipeline

Follow these steps to get the image preprocessing pipeline up and running:

### 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Installation: Clone the Repository ⚙️
First, clone the repository to your local machine to get the necessary files. Run the following command in your terminal:

```
git https://github.com/hi-paris/pyronear.git
```

##### Import Library 📦
What things you need to install the software and how to install them:

```
pip install -r requirements.txt
```

### 🛠 Usage
Run the following notebook to visualize your images with data augmentation and transformation!

```
# Import the process_image function from the processing module
from processing import process_image

# Define the paths for your image and label
image_path = "images/aiformankind_v1_000007.jpg"
label_path = "images/aiformankind_v1_000007.txt"

# Define the class names associated with your labels
class_names = ["smoke"]

# Process the image and possibly display it, depending on your function's implementation
process_image(image_path, label_path, class_names)
```
---

## **2️⃣ Super-resolution Model**: ESRGAN
Utilizes a pre-trained ESRGAN model and a from-scratch ESRGAN model for enhancing image resolution.

### From Scratch 
The "From Scratch" component involves training an ESRGAN model from the ground up using our dataset. This process includes several key scripts and modules

The following scripts are essential for the from-scratch ESRGAN model:

- config.py: Contains the configuration settings for the model, including hyperparameters, paths, and other essential parameters for training and evaluation.

- dataset.py: Manages the loading and preprocessing of the dataset. This script handles the creation of low-resolution and high-resolution image pairs required for training the ESRGAN model.

- loss.py: Defines the loss functions used in training, including the adversarial loss, perceptual loss, and pixel-wise loss, which guide the generator and discriminator during the training process.

- model.py: Implements the architecture of the ESRGAN model, including the generator and discriminator networks.

- train.py: The main training script that orchestrates the training process. It initializes the model, loads the dataset, and iteratively updates the model weights based on the defined loss functions.

- utils.py: Contains utility functions that support various tasks such as image transformations, checkpoint saving and loading, and performance metric calculations.

### Pre-trained (Tensorflow)

The "Pre-trained" component utilizes an already trained ESRGAN model implemented in Tensorflow. This approach includes:

- Model Integration: Integrating a pre-trained Tensorflow ESRGAN model into our project.
- Inference: Using this model to perform super-resolution on new, unseen low-resolution images. The pre-trained model has already learned to enhance images from extensive training on large datasets, enabling quick and effective super-resolution.
- Comparison: Comparing the performance and output quality of the pre-trained model with the custom-trained model. This helps in understanding the benefits and limitations of each approach.

![image](https://github.com/hi-paris/pyronear/assets/108234726/ebf81ce6-a9bb-4b1e-98f6-d865b676bc05)


## **3️⃣ Hyperparameter Tuning with random search**

In this section, the focus is on optimizing the parameters for Albumentations data augmentation techniques to achieve a higher Mean Average Precision (MAP) score for the model. Data augmentation plays a crucial role in enhancing the diversity of training data and improving the robustness of machine learning models. However, finding the best combination of augmentation parameters can be challenging and time-consuming.

To address this, a random search approach is employed for hyperparameter tuning. This method involves the following steps:

- Define the Search Space: Specify the range of possible values for each augmentation parameter. This includes various transformations like rotation, scaling, shifting, and flipping, among others.

- Random Sampling: Randomly sample a set of parameters from the defined search space. This helps explore a wide variety of combinations without being biased towards any specific region of the space.

- Evaluate Performance: For each set of sampled parameters, train the model and evaluate its performance using the Mean Average Precision (MAP) score. This score measures the precision of the model across different recall levels, providing a comprehensive evaluation of its accuracy.

- Select the Best Parameters: Identify the parameter set that yields the highest MAP score. These optimal parameters will be used for data augmentation in the final training process.

## **4️⃣ Data augmentation with albumentations**


https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#yolo

## **5️⃣ Training yolov10**

https://docs.ultralytics.com/models/yolov10/#usage-examples

## Licence 📜

MIT License

---
## References

---
