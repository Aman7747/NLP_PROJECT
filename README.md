# NLP_PROJECT
---

# Swahili News Classification using AfriBERTA

This repository contains a Python script for classifying Swahili news articles into predefined categories using the AfriBERTa model. The project leverages the Hugging Face Transformers library and is designed to run on the Kaggle platform.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to classify Swahili news articles into five categories: Biashara, Burudani, Kimataifa, Kitaifa, and Michezo. The project uses the AfriBERTa model, a pre-trained language model fine-tuned for sequence classification tasks. The dataset consists of news articles with labeled categories, and the model is trained to predict these categories based on the article content.

## Dataset
The dataset used in this project is available in the `/kaggle/input/swaillidataset/` directory and includes the following files:
- `Train.csv`: Training data with labeled news articles.
- `Test.csv`: Test data without labels.
- `SampleSubmission.csv`: Sample submission file for Kaggle competition.
- `VariableDefinitions.csv`: Definitions of variables used in the dataset.

## Dependencies
To run this project, The following dependencies are required
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- PyTorch
- NLTK
- Matplotlib
- Seaborn
- WordCloud

Installation of these dependencies using pip:
```bash
pip install pandas numpy scikit-learn transformers datasets torch nltk matplotlib seaborn wordcloud
```

## Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Aman7747/NLP_PROJECT.git
   cd NLP_PROJECT
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   - Download the dataset from the [Kaggle Swahili Dataset](https://www.kaggle.com/datasets/yourdatasetlink) and place it in the `input` directory.

4. **Set Up Kaggle Environment:**
   - If you are running this on Kaggle, ensure you have the necessary permissions and environment set up.

## Usage
1. **Run the Notebook:**
   - Open the Jupyter Notebook (`nlp_project1.ipynb`) in your preferred environment.
   - Execute the cells to preprocess the data, train the model, and make predictions.

2. **Training the Model:**
   - The model is trained using the `Trainer` class from the Hugging Face Transformers library.
   - The training process involves tokenizing the data, creating datasets, and defining training arguments.

3. **Making Predictions:**
   - After training, the model can be used to predict categories for new news articles.
   - Example usage is provided in the notebook.

## Results
The model achieves the following results on the validation set:
- **Log Loss:** 0.2734
- **Classification Report:**
  - **Biashara:** Precision: 0.868, Recall: 0.897, F1-Score: 0.882
  - **Burudani:** Precision: 0.000, Recall: 0.000, F1-Score: 0.000
  - **Kimataifa:** Precision: 0.750, Recall: 0.667, F1-Score: 0.706
  - **Kitaifa:** Precision: 0.913, Recall: 0.870, F1-Score: 0.891
  - **Michezo:** Precision: 0.952, Recall: 0.984, F1-Score: 0.968

## Contributing
Contributions are welcome!

---
Feel free to customize this README.
