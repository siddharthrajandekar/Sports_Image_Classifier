# 🏅 Sports Image Classification with CNNs

This project classifies the sport in a given image using various Convolutional Neural Networks (CNNs): **LeNet**, **AlexNet**, **VGG16**, and **ResNet18**.



## 📊 Models Used

- **LeNet** – Lightweight architecture with simple design
- **AlexNet** – Deeper network with large receptive fields
- **VGG16** – Very deep network with small filters
- **ResNet18** – Deep network using residual connections

## 📁 Dataset

The dataset originally used has been removed. However, you can replace it with any sports image classification dataset.

Make sure to place the dataset at the expected path or update the notebook accordingly.

## 📈 Results

Model performance (accuracy and loss) is recorded over epochs and visualized for comparison across different CNNs. All the results are documented in the file reports/final_report.pdf

## 🛠 Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

You can run each notebook in order, starting from 0_data_preprocessing.ipynb.

🚀 Running the Project
Start with 0_data_preprocessing.ipynb to load and prepare the dataset.

Choose a model from the models folder.

Train using the 2_train_models.ipynb notebook.

Visualize results using 3_plot_results.ipynb.

Run predictions with 4_predict_and_visualize.ipynb.

📌 Note
This project was developed and tested in Google Colab. If running locally, you may need to update data paths or install GPU drivers for torch. 

This structure has been created for the sake of modular clarity. To run the code cohesively see main.ipynb.

For the results and accuracies of all the models see results/