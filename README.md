# News Ranking System

## Project Overview
The **News Ranking System** is an AI-driven application that ranks news statements based on their sentiment, from most positive to most negative. The system utilizes a pre-trained model from Hugging Face(DistilBert), which has been fine-tuned for optimal performance. The project includes data extraction, transformation, model fine-tuning, and deployment as a Flask API.

## Project Structure
```
NewsRankingSystem/
│── app.py                     # Main API entry point
│── templates/
│   ├── index.html              # Web interface for ranking news
│── assets/                     # Stores images and assets
│── notebooks/                  # Jupyter notebooks for experimentation
│── data/                       # Data folder for storing raw and cleaned data
│   ├── raw.csv                 # Raw data file
│   ├── cleaned.csv             # Cleaned data folder
│── logs/                       # Log folder for storing log files
│── src/
│   ├── data_operations/
│   │   ├── data_extraction.py     # Extracts data from Kaggle
│   │   ├── data_transformation.py # Cleans and processes data
│── model/
│   ├── data_preparation.py        # Prepares data for fine-tuning
│   ├── train.py                   # Fine-tunes the pre-trained model
│── utils/
│   ├── exceptions.py              # Custom exceptions
│   ├── logger.py                  # Custom logging utilities

```

---

## Project Workflow

1. **Data Extraction & Cleaning**
   - The dataset is extracted from Kaggle and preprocessed to remove unused column.
   
2. **Model Fine-Tuning**
   - A pre-trained sentiment analysis model from Hugging Face is fine-tuned using cleaned data.
   - The trained model is uploaded to the Hugging Face Model Hub.
   
3. **Deployment**
   - The model is deployed using Flask as an API.
   - The Hugging Face Inference API is leveraged for real-time sentiment analysis.
   
4. ### News Ranking Function
The news statements are ranked using the following equation:

**R = sorted(S, key=(s₁ ≠ "POSITIVE", -s₂))**

Where:
- **S** is the list of sentiment results.
- **s₁** represents the sentiment classification.
- **s₂** is the confidence score.


![Ranking Process](https://github.com/AdanALalawni/NewsRankingSystem-/blob/main/assets/Project_Outlines.png)

---

## Conclusion
The News Ranking System efficiently analyzes and ranks news statements based on sentiment. With a pipeline for data preparation, model training, and deployment, the system provides an easy-to-use API for sentiment-based news ranking.

![Results](https://github.com/AdanALalawni/NewsRankingSystem-/blob/main/assets/Webscreen.png)

