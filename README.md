# Sentiment-Analysis

**Description** - In this Project you will go through a Natural Language Processing Python Project creating a Sentiment Analysis classifier with NLTK's VADER and Huggingface Roberta Transformers. The project is to classify the seniment of amazon customer reviews. 

we have represented difference between model outputs from the two packages and compare the results. 

Seniment analysis is a tool to use in laguage modeling.

Link to dataset - https://drive.google.com/file/d/1SETDmEdOR0WdTFlhKe-zgcoLbcgbs_Z6/view?usp=sharing

software requirement - Jupyter Notebook

**About Dataset**
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

**Data includes:**
Reviews from Oct 1999 - Oct 2012
568,454 reviews
256,059 users
74,258 products
260 users with > 50 reviews

**A Brief Intro to Sentiment Analysis**
Sentiment analysis is an approach to identifying the emotional tone behind textual data. This helps organizations or businesses gather insights from unstructured text that comes from online sources such as surveys, social media channels, and comments. Various algorithms (models) are available for sentiment analysis tasks, and each has its pros and cons

**Implementation overview**
we have completed some sentiment analysis in python using two different techniques:

1. VADER (Valence Aware Dictionary and sEntiment Reasoner) - Bag of words approach
2. Roberta Pretrained Model from 🤗Huggingface Pipeline


<img width="922" alt="Screenshot 2023-11-29 at 7 57 13 AM" src="https://github.com/nikkipatel19/Sentiment-Analysis/assets/67902583/24a48715-904f-4e42-8590-4c79899a4dcc">





# We are showing score for positive review, negative review, neutral review and also shown compound score

# After that we have shown score of review for each column on entire dataset

**VADER vs Roberta**

VADER (Valence Aware Dictionary and sEntiment Reasoner) and RoBERTa (Robustly optimized BERT approach) are two different tools used in natural language processing (NLP) for sentiment analysis, but they have different approaches and purposes.

**VADER:**

Purpose: VADER is a rule-based sentiment analysis tool designed for social media text. It is particularly effective for short and informal texts, such as tweets or online reviews.
Features: VADER analyzes text by assigning a polarity score to each word and then combines these scores to determine the overall sentiment of the text. It is pre-trained on a lexicon of words with associated sentiment scores.
Pros and Cons:
**Pros:** Simple to use, fast, and effective for certain types of text data.
**Cons**: May not perform as well on more complex or formal texts, as it relies on a predefined lexicon.

**RoBERTa:**

Purpose: RoBERTa is a transformer-based language model, and it is a variant of BERT (Bidirectional Encoder Representations from Transformers). While it can be used for various NLP tasks, including sentiment analysis, its primary strength lies in capturing contextual information in language.
Features: RoBERTa is pre-trained on a large corpus of diverse text and is capable of understanding context and relationships between words. It requires fine-tuning for specific tasks, including sentiment analysis.
Pros and Cons:
**Pros:** Powerful and flexible, capable of handling a wide range of NLP tasks. Excels in capturing contextual nuances in language.
**Cons:** Requires more computational resources, and fine-tuning for specific tasks may be necessary.

After the implementation of 2 different techniques, we have compared a result of both.
and at the end, we have used Hugging Face Transformers library to create a sentiment analysis pipeline. 

from transformers import pipeline: This imports the pipeline module from the Hugging Face Transformers library.

pipeline("sentiment-analysis"): This creates a sentiment analysis pipeline using a pre-trained model for sentiment analysis. Hugging Face provides pre-trained models for various NLP tasks, and we have used a sentiment analysis model.

After running this code, we can use the sent_pipeline to analyze the sentiment of text. For example:

<img width="1063" alt="Screenshot 2023-11-29 at 7 54 21 AM" src="https://github.com/nikkipatel19/Sentiment-Analysis/assets/67902583/edd62eb9-0e2d-4121-997f-48bb145e50ab">
