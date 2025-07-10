#!/usr/bin/env python
# coding: utf-8

# In[1]:




# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_excel('/content/labeled_dataset.xlsx')
df.head()


# In[2]:


print(df.columns)


# In[3]:


# Check for missing values
print(df.isnull().sum())

# Drop rows with missing text or label
df.dropna(subset=['article', 'Label_bias'], inplace=True)

# Basic text cleaning
import re
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove punctuation
    text = text.lower().strip()
    return text

df['clean_text'] = df['article'].apply(clean_text)

# Tokenization & stopword removal
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)


# In[4]:


# Label distribution
sns.countplot(x='Label_bias', data=df)
plt.title('Bias Label Distribution')
plt.show()

# Word cloud for biased text
from wordcloud import WordCloud

biased_text = ' '.join(df[df['Label_bias'] == 'Biased']['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(biased_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Biased Articles')
plt.show()


# In[5]:


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['Label_bias']


# In[6]:


from sklearn.preprocessing import PolynomialFeatures

# Step 1: Generate TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X = vectorizer.fit_transform(df['clean_text']).toarray()

# Step 2: Apply polynomial transformation
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_poly.shape)




# In[7]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# In[8]:


# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[9]:


import pickle

# Save model and vectorizer
with open('bias_detector_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# In[10]:


df.drop(columns=['news_link', 'outlet'], inplace=True)


# In[11]:


df.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['Label_bias'])


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[19]:


import pickle
pickle.dump(model, open("bias_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))



# Save trained model
import joblib
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
