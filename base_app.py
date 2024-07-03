import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Function to load the models
def load_models():
    models = {}
    model_names = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Regression', 'Bagging', 'AdaBoost']
    for model_name in model_names:
        try:
            with open(f'{model_name}_model.pkl', 'rb') as model_file:
                models[model_name] = pickle.load(model_file)
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
    return models

# Load the vectorizer
def load_vectorizer():
    try:
        with open('vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        return vectorizer
    except Exception as e:
        st.error(f"Error loading TfidfVectorizer: {e}")
        return None

# Class names mapping
class_names = {
    0: 'business',
    1: 'education',
    2: 'entertainment',
    3: 'sports',
    4: 'technology'
}

# Function to generate word cloud
def generate_word_cloud(text_column, title):
    all_text = ' '.join(text_column)
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {title}')
    plt.axis('off')
    st.pyplot(plt)

# Function to plot average length by category
def plot_avg_length_by_category(train_data_EDA):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.lineplot(data=train_data_EDA, x='category', y='headlines_length', estimator='mean')
    plt.title('Average Headlines Length by Category')
    plt.xlabel('Category')
    plt.ylabel('Average Headlines Length')

    plt.subplot(1, 3, 2)
    sns.lineplot(data=train_data_EDA, x='category', y='description_length', estimator='mean')
    plt.title('Average Description Length by Category')
    plt.xlabel('Category')
    plt.ylabel('Average Description Length')

    plt.subplot(1, 3, 3)
    sns.lineplot(data=train_data_EDA, x='category', y='content_length', estimator='mean')
    plt.title('Average Content Length by Category')
    plt.xlabel('Category')
    plt.ylabel('Average Content Length')

    plt.tight_layout()
    st.pyplot(plt)

# Load the models and vectorizer
models = load_models()
vectorizer = load_vectorizer()

# Sidebar with logo and navigation
st.image("https://cdn.wan-ifra.org/wp-content/uploads/2021/06/24134841/DataSci2-scaled.jpg", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Classification"])

# Home Page
if page == "Home":
    st.title("Welcome to the Text Classification App")
    st.write("Use the sidebar to navigate through different sections of the app.")

# EDA Page
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Load the training data 
    train_data = pd.read_csv('train.csv')
    
    # Calculate the length of each text field
    train_data_EDA = train_data.copy()
    train_data_EDA['headlines_length'] = train_data_EDA['headlines'].apply(len)
    train_data_EDA['description_length'] = train_data_EDA['description'].apply(len)
    train_data_EDA['content_length'] = train_data_EDA['content'].apply(len)
    
    # Generate word clouds
    generate_word_cloud(train_data_EDA['headlines'], 'Headlines')
    generate_word_cloud(train_data_EDA['description'], 'Descriptions')
    generate_word_cloud(train_data_EDA['content'], 'Content')
    
    # Plot average length by category
    plot_avg_length_by_category(train_data_EDA)
    
    # Plot the distribution of the category column for training data
    st.write("Distribution of News Categories for Training Data:")
    plt.figure(figsize=(10, 6))
    sns.countplot(y='category', data=train_data, order=train_data['category'].value_counts().index)
    plt.title('Distribution of News Categories for Training Data')
    plt.xlabel('Count')
    plt.ylabel('Category')
    st.pyplot(plt)

    # Load the test data 
    test_data = pd.read_csv('test.csv')
    
    # Plot the distribution of the category column for test data
    st.write("Distribution of News Categories for Testing Data:")
    plt.figure(figsize=(10, 6))
    sns.countplot(y='category', data=test_data, order=test_data['category'].value_counts().index)
    plt.title('Distribution of News Categories for Test Data')
    plt.xlabel('Count')
    plt.ylabel('Category')
    st.pyplot(plt)

# Classification Page
elif page == "Classification":
    st.title("Text Classification")
    
    # Input text
    input_text = st.text_area("Enter text for classification:")
    
    # Model selection
    model_choice = st.selectbox("Select a model for classification:", list(models.keys()))
    
    if st.button("Classify"):
        if input_text.strip():
            # Get the selected model
            model = models.get(model_choice)
            
            if model and vectorizer:
                try:
                    # Transform the input text using the vectorizer
                    input_tfidf = vectorizer.transform([input_text])
                    
                    # Predict the class using the model pipeline
                    prediction = model.predict(input_tfidf)
                    
                    # Map the numerical prediction to the class name
                    class_name = class_names.get(prediction[0], "Unknown class")
                    
                    # Display the result
                    st.write(f"Predicted class: {class_name}")
                except Exception as e:
                    st.error(f"Error during classification: {e}")
            else:
                st.error("Selected model or vectorizer could not be loaded.")
        else:
            st.write("Please enter some text for classification.")
