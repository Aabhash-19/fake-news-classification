# Fake News Classification

A comprehensive machine learning project designed to accurately classify news articles as real or fake using Natural Language Processing (NLP) and various classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-green)
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-lightblue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Project Overview

In today's digital age, the rapid spread of misinformation poses a significant challenge. This project addresses the problem of fake news by implementing a machine learning solution that can automatically classify news articles as genuine or fraudulent. The system processes raw text data through advanced NLP techniques and employs multiple classification algorithms to achieve high accuracy in detection.

## 📊 Dataset

The model is trained on a widely-used benchmark dataset consisting of two categories:

- **True.csv**: Articles sourced from Reuters, labeled as **REAL** news (Label `0`)
- **Fake.csv**: Articles collected from various unreliable sources, labeled as **FAKE** news (Label `1`)

**Dataset Statistics:**
- Total Samples: 44,898 articles
- Real News: 21,417 articles
- Fake News: 23,481 articles

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline

The text data undergoes comprehensive cleaning and preparation:

1. **Text Cleaning**:
   - Removal of special characters, punctuation, and numbers
   - Elimination of brackets, URLs, and HTML tags
   - Stripping of whitespace and non-ASCII characters
   - Handling of case consistency

2. **Data Preparation**:
   - Concatenation of real and fake datasets
   - Label assignment (0 for real, 1 for fake)
   - Shuffling of the combined dataset

### Feature Engineering

- **TF-IDF Vectorization** with maximum 20,000 features
- Inclusion of both unigrams and bigrams
- English stop words removal

### Model Training & Evaluation

The project implements and compares five classification algorithms:

1. Logistic Regression
2. Naive Bayes (MultinomialNB)
3. Linear Support Vector Classifier (LinearSVC)
4. Random Forest Classifier
5. XGBoost Classifier

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Reports
- Confusion Matrices

## 📈 Performance Results

The models achieved exceptional performance, with tree-based ensembles demonstrating the best results:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.9971** | **0.9983** | **0.9962** | **0.9972** |
| **Random Forest** | 0.9969 | 0.9979 | 0.9962 | 0.9970 |
| **Linear SVC** | 0.9952 | 0.9957 | 0.9951 | 0.9954 |
| **Logistic Regression** | 0.9882 | 0.9895 | 0.9879 | 0.9887 |
| **Naive Bayes** | 0.9482 | 0.9438 | 0.9580 | 0.9509 |

**Key Insight:** Ensemble methods (XGBoost and Random Forest) outperformed other classifiers, achieving remarkable accuracy exceeding **99.7%**, demonstrating their effectiveness for this classification task.

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Aabhash-19/fake-news-classification.git
cd fake-news-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Data

1. Download the True.csv and Fake.csv datasets
2. Place them in a dataset/ folder within the project directory
3. Update the file paths in the Jupyter notebook if necessary:

```python
true_path = "/your/path/to/dataset/True.csv"
fake_path = "/your/path/to/dataset/Fake.csv"
```

### 4. Execute the Code

Run the Jupyter notebook to execute the complete pipeline:

```bash
jupyter notebook AllModels.ipynb
```
Alternatively, run the code directly in your preferred Python environment.


## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Disclaimer:** This model is a proof-of-concept trained on a specific dataset. Its performance may vary on news from different domains or styles. Always exercise critical thinking and use automated tools as aids rather than absolute truth sources.
