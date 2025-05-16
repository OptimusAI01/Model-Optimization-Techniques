# 🚀 Efficient Deep Learning: Optimizing Image Classifiers with TensorFlow

A practical exploration of model pruning, quantization, and knowledge distillation for image classification on the CIFAR-10 dataset using ResNet50 as a base. This project demonstrates techniques to significantly reduce model size and inference latency while aiming to preserve accuracy.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-critical)
![TFMO](https://img.shields.io/badge/TF%20Model%20Optimization-blueviolet)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-green)

## 🌟 Project Overview

Deep learning models, especially for computer vision tasks, can be computationally expensive and large, making them challenging to deploy on resource-constrained environments like mobile devices or edge hardware. This project tackles this challenge by:

1.  **Training a baseline image classifier** (ResNet50 via transfer learning) on CIFAR-10.
2.  **Implementing and evaluating three powerful model optimization techniques:**
    *   **Magnitude Pruning:** Sparsifying the model by removing less important weights.
    *   **Post-Training Quantization:** Reducing model precision (e.g., to INT8) for smaller size and faster inference, leveraging TensorFlow Lite.
    *   **Knowledge Distillation:** Training a smaller, faster "student" model (MobileNetV2) to mimic the behavior of the larger "teacher" model (our fine-tuned ResNet50).
3.  **Benchmarking** the original and optimized models based on accuracy, model size, and inference time.

## ✨ Key Features & Techniques Demonstrated

*   **Transfer Learning:** Utilizing a pre-trained ResNet50 on ImageNet and fine-tuning it for CIFAR-10.
*   **Efficient Input Pipelines:** Using `tf.data.Dataset` for optimized data loading and preprocessing.
*   **Magnitude Pruning:** Employing `tensorflow_model_optimization` (TFMO) toolkit for structured and unstructured pruning.
*   **Post-Training Quantization:** Converting Keras models to TensorFlow Lite (`.tflite`) format with INT8 quantization.
*   **Knowledge Distillation:** Implementing a teacher-student learning setup using KL Divergence loss.
*   **Comprehensive Evaluation:** Measuring and comparing:
    *   Accuracy on the test set.
    *   Model size on disk (MB).
    *   Inference speed (ms/image).
*   **Visualization:** Plotting results to visually compare the trade-offs of different optimization methods.

## 📊 Key Results & Highlights

This project systematically compares the performance of different optimization strategies. Here's a summary of the findings:

| Model Type             |  Size (MB) | Inference Time (ms/batch) | Size Reduction | Speed-up Factor |
| :--------------------- | :-------: | :-----------------------: | :------------: | :-------------: |
| **Teacher (ResNet50)** |  ~245 MB  |        ~40,967 ms         |       -        |        1x       |
| **Pruned Model**       |   ~245 MB* |        ~40,965 ms         |     ~1%*       |     ~1x*      |
| **Quantized (TFLite)** |    ~23 MB   |         ~25,000 ms         |     ~90%       |  -  |
| **Student (MobileNetV2)**|  ~26 MB   |        ~20,483 ms         |     ~89%       |     ~2x       |

*   **Note on Pruned Model Size/Speed:**
    *   The `.h5` file for the pruned model saved via Keras remains large because Keras saves dense layers without inherently exploiting sparsity. The *actual number of non-zero parameters* is reduced by pruning.
    *   Benefit in size/speed is more apparent when converting to sparse-aware formats (like TFLite with specific configurations) or on hardware with native sparse computation support.

**Key Takeaways:**

*   **Quantization** offered the most significant size reduction (~90%) with a relatively small drop in accuracy and notable inference speed-up when using the TFLite interpreter.
*   **Knowledge Distillation** successfully trained a much smaller MobileNetV2 model (~89% smaller) that was ~2x faster while retaining a good portion of the teacher's accuracy.
*   **Pruning** showed potential for complexity reduction, but its direct benefits in standard Keras `.h5` models require further steps (like TFLite conversion that leverages sparsity) to be fully realized in terms of size and speed.

## 🛠️ Technologies Used

*   **Python 3.x**
*   **TensorFlow 2.x**
*   **Keras** (as `tf.keras`)
*   **TensorFlow Model Optimization (TFMO)**
*   **TensorFlow Lite (TFLite)**
*   **NumPy**
*   **Matplotlib**
*   **Jupyter Notebooks** (for development and presentation)

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OptimusAI01/Model-Optimization-Techniques.git
    cd CIFAR-10-Model-Techniques
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    
    Key dependencies include: `tensorflow`, `tensorflow-model-optimization`, `matplotlib`, `numpy`.

## 🚀 How to Run

1.  Ensure all dependencies are installed (see Setup & Installation).
2.  The primary code and experiments are within the Jupyter Notebook: `[CIFAR_model_techniques.ipynb]`.
3.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook [CIFAR_model_techniques.ipynb]
    ```
4.  Open the notebook and run the cells sequentially.
    *   **Note:** Training the models can take a significant amount of time, especially the initial ResNet50 fine-tuning and the knowledge distillation process. The comments in the notebook indicate approximate training times per epoch on Kaggle GPU.


## 🔮 Future Work & Potential Enhancements

*   Explore **Quantization-Aware Training (QAT)** for potentially better accuracy with quantized models.
*   Experiment with **different student architectures** for knowledge distillation.
*   Evaluate models on **actual edge devices** (e.g., Raspberry Pi, Coral Edge TPU, Android/iOS device).
*   Investigate **structured pruning** techniques more deeply.

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), Have Fun!
