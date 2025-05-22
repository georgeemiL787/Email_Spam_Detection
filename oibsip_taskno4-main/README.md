# Email Spam Detection with Machine Learning

## Project Overview
This project demonstrates how machine learning can be used to detect spam emails. By analyzing the content of emails and extracting relevant features, a classification model is trained to distinguish between legitimate (ham) and spam messages. The goal is to provide a practical tool for improving email security and reducing unwanted messages in users' inboxes.

## Features
- Data preprocessing and cleaning of a real-world email dataset
- Feature extraction from email text
- Training and evaluation of machine learning models (e.g., Multinomial Naive Bayes)
- Performance metrics: accuracy, precision, recall, F1-score, ROC-AUC
- Example function for predicting whether a new email is spam or not

## How It Works
1. **Data Loading:** The dataset (`spam.csv`) is loaded and inspected.
2. **Preprocessing:** The data is cleaned, missing values are handled, and text is prepared for analysis.
3. **Feature Engineering:** Text features are extracted using techniques such as bag-of-words (CountVectorizer).
4. **Model Training:** The data is split into training and test sets. A machine learning model (Multinomial Naive Bayes) is trained to classify emails.
5. **Evaluation:** The model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score.
6. **Prediction:** A sample function demonstrates how to use the trained model to predict if a new email is spam.

## How to Run the Project
1. **Requirements:**
   - Python 3.x
   - Jupyter Notebook or Google Colab
   - Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud
2. **Steps:**
   - Open `Email_Spam_Detection_with_Machine_Learning.ipynb` in Jupyter Notebook or upload it to Google Colab.
   - Ensure `spam.csv` is in the same directory as the notebook.
   - Run the notebook cells sequentially to execute data loading, preprocessing, model training, and evaluation.
   - Use the provided function to test the model on your own email samples.

## Machine Learning Approach
The project uses a supervised learning approach, specifically text classification. The Multinomial Naive Bayes algorithm is well-suited for this task due to its effectiveness with word count features. The model is trained on labeled data (spam/ham) and learns to identify patterns and keywords commonly associated with spam emails.

## Example Usage
After running the notebook, you can use the following function to classify new emails:

```python
def detect_spam(email_text):
    prediction = clf.predict([email_text])
    return "Spam" if prediction == 1 else "Ham"
```
 