import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

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
            fake = fake_news(news) 

            # Display results
            if fake:
                st.error("This news looks unreliable!! :warning:")
            else:
                st.success("This news looks reliable :thumbs_up:")

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

if __name__ == '__main__':
    main()
