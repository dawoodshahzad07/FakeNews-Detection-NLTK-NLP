import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image

nltk.download('stopwords')
nltk.download('vader_lexicon')

vector_form = pickle.load(open('D:\\university_files\\Semester_4\\Jupyter_Tasks\\pai_ccp\\vector.pkl', 'rb'))
load_model = pickle.load(open('D:\\university_files\\Semester_4\\Jupyter_Tasks\\pai_ccp\\model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[PorterStemmer().stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)

    # Sentiment Analysis
    sentiment = SentimentIntensityAnalyzer()
    compound_score = sentiment.polarity_scores(news)['compound']

    return prediction, compound_score

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Fake News Checker",
        page_icon=":newspaper:",
    )

    st.title("Fake News Classification App")
    st.write("""
        # Detect fake news with AI!
    """)

    with st.expander("About this app"):
        st.write("""
            This app uses a machine learning model to classify news content as reliable or unreliable.
            Enter any news article or text content to check if it is fake news.
            \nCredits:
            \nDawood Shahzad
            \nAli Hassan
            \nHamza Khalid
        """)

    # Input textarea  
    news = st.text_area("Enter the news content", height=200)

    # Prediction button
    if st.button("Predict"):  
        with st.spinner("Classifying..."):
            # Get prediction
            fake, compound_score = fake_news(news) 

            # Display results
            if fake:
                st.error(f"This news looks unreliable!! Sentiment: {get_sentiment_label(compound_score)}")
            else:
                st.success(f"This news looks reliable :thumbs_up: Sentiment: {get_sentiment_label(compound_score)}")

    # Additional features
    st.markdown("---")
    st.subheader("Additional Features")
    st.write("""
        - Learning..
        - https://github.com/dawoodshahzad07
    """)

    # Footer with GIF
    st.markdown("---")
    st.subheader("Powered by Streamlit :rocket:")
    st.image("https://i.pinimg.com/originals/62/26/43/6226435516042edfe1a4514a44e2023a.gif")

# Function to get sentiment label
def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    main()
