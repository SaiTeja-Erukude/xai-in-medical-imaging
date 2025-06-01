# XAI in Medical Imaging

### Overview
This repository contains code and results for detecting brain tumors in MRI scans and pneumonia in chest X-ray images using deep learning, enhanced with explainable AI (XAI) techniques.

We use two state-of-the-art convolutional neural networks: ResNet50 and DenseNet121 and apply Grad-CAM visualizations to generate heatmaps, showing which regions of the image influenced the model’s predictions.

## Getting Started

### Installation
1. Create a Virtual Environment:
    ```bash
    python -m venv venv
    ```

2. Activate the Virtual Environment:
    - On Windows:
    ```
    venv\Scripts\activate
    ```
    - On macOS/Linux:
    ```
    source venv/bin/activate
    ```

3. Clone the repository:
   ```bash
   git clone https://github.com/SaiTeja-Erukude/xai-in-medical-imaging.git
   cd xai-in-medical-imaging
   ```

4. Install the Required Packages:
    ```
    pip install -r requirements.txt
    ```