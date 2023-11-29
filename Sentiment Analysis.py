#!/usr/bin/env python
# coding: utf-8

# #### In this notebook we will be doing some sentiment analysis in python using two different techniques:
# 
# * VADER (Valence Aware Dictionary and sentiment Reasoner) - Bag of words approach 
# * Roberta Pretrained Model from 🤗
# * Huggingface Pipeline

# ## Read in Data and NLTK Basics

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[2]:


# Read in data
df = pd.read_csv('AI.csv')
print(df.shape)
df = df.head(500)
print(df.shape)


# In[3]:


df.head()


# ## Performing Basic EDA

# In[4]:


ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# ## Basic NLTK¶

# In[5]:


example = df['Text'][55]
print(example)


# In[6]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[7]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[8]:


import nltk
nltk.download('vader_lexicon')


# In[9]:


import nltk
nltk.download('vader_lexicon')


# In[10]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# ## VADER Seniment Scoring
# We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# 
# This uses a "bag of words" approach:
# * Stop words are removed
# * Each word is scored and combined to a total score.

# In[11]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[12]:


sia.polarity_scores('I cannot stop smiling because I just got the things I have been waiting for')


# In[13]:


sia.polarity_scores('This is the worst prouct i have ever purchased.')


# In[14]:


sia.polarity_scores(example)


# In[15]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[16]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[17]:


# Now we have sentiment score and metadata
vaders.head()


# In[18]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[19]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


#  ## Roberta Pretrained Model
# * Use a model trained of a large corpus of data.
# * Transformer model accounts for the words but also the context related to other words

# In[20]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[21]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[22]:


# VADER results on example
print(example)
sia.polarity_scores(example)


# In[23]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[24]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[25]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[26]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# ## Compare Scores between models

# In[27]:


results_df.columns


# ## Combine and compare

# In[28]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# ## Review Examples:
# * Positive 1-Star and Negative 5-Star Reviews
# * Lets look at some examples where the model scoring and review score differ the most.

# In[39]:


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[40]:


results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# ## negative sentiment 5-Star view

# In[41]:


results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]


# ## The Transformers Pipeline
# * Quick & easy way to run sentiment predictions

# In[42]:


from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[43]:


sent_pipeline('I love java it is an amazing language')


# In[44]:


sent_pipeline('o wow congrats you got a job')


# In[45]:


sent_pipeline('I won every BATTLE but i am LOSING this WAR"')


# In[ ]:




