# Plant Disease Detection System

## Overview
This project presents a Deep Learning-based Plant Disease Detection System that leverages Convolutional Neural Networks (CNNs) to identify and classify plant diseases from leaf images with approximately 96% accuracy. The model is trained to recognize 38 different types of plant diseases across 14 different plant species, providing a scalable solution for timely disease detection, thus contributing to food security by reducing crop damage and the need for pesticides.

## Features
- **Deep Convolutional Neural Networks (CNNs):**
  - Developed a CNN model that accurately classifies plant diseases based on leaf images.
  - Trained on a large dataset to distinguish between 38 different diseases affecting 14 plant species.
  
- **High Accuracy:**
  - Achieved an accuracy of approximately 96% in disease detection, highlighting the model's effectiveness in real-world agricultural applications.
  
- **Image Processing Techniques:**
  - Implemented image processing techniques to accurately isolate plant leaves from complex backgrounds, enhancing the model's ability to correctly identify diseases in varying environmental conditions.
  
- **Food Security Contribution:**
  - By facilitating early and accurate disease detection, this system aids in reducing crop damage, minimizing the need for chemical treatments, and ultimately contributing to higher crop yields and better food security.

## Web App Preview
Here’s a preview of the web application where users can upload leaf images to predict diseases:

![Web App Preview](path_to_your_image_here.png)

*Replace the above path with the actual path to your image file in the repository.*

## Dataset
The model is trained on a diverse dataset containing images of healthy and diseased plant leaves. The dataset consists of approximately 68,000 training images and 17,000 testing images, covering 38 different diseases affecting 14 plant species.

**Dataset Link:** [Plant Village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Model Architecture
The system employs a deep CNN architecture with multiple layers designed to capture intricate patterns in the leaf images:
- **Convolutional Layers:** Extract spatial features from the input images.
- **Pooling Layers:** Reduce the dimensionality of feature maps, retaining essential features.
- **Fully Connected Layers:** Combine features to classify images into respective disease categories.
- **Softmax Output:** Produces probability distributions over 38 classes, identifying the most likely disease category.

## Training Process
- **Data Augmentation:** Applied techniques such as rotation, flipping, and scaling to increase dataset variability and improve model robustness.
- **Optimization:** Used Adam optimizer with a learning rate of `0.001`, tuned for optimal performance.
- **Loss Function:** Cross-entropy loss was minimized over multiple epochs to ensure accurate classification.
- **Validation:** Monitored validation accuracy to prevent overfitting and ensure generalizability.

## Results
- The final model achieved an accuracy of approximately 96% on the test dataset.
- Visualizations of training metrics (accuracy, loss) are provided in the notebook, demonstrating the model's learning curve.

## Usage
1. **Inference:** To run inference on new images:
    ```python
    from predict import predict_disease
    result = predict_disease('path_to_leaf_image.jpg')
    print(result)
    ```
2. **Visualization:** Use provided scripts to visualize predictions and the corresponding probabilities.

## Future Work
- **Model Improvements:** Explore advanced architectures such as EfficientNet or Transformer-based models to further enhance accuracy.
- **Real-time Deployment:** Implement a real-time disease detection system using edge devices such as Raspberry Pi.
- **Mobile App:** Develop a mobile application for farmers to easily use the model for disease detection in the field.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Special thanks to the contributors of the plant disease dataset and the open-source community for the tools and frameworks used in this project.
- Inspired by the Advanced Functional Materials Research Group (AFMRG) at IIT Indore for their focus on improving agricultural practices through innovative research.
