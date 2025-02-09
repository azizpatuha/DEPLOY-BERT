import streamlit as st
import re
# import random
# import numpy as np
import pandas as pd
# import torch
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

def admin_upload_yt():

    #Fungsi Text processing

    #text processing
    def text_preprocessing(text):
        # Ubah menjadi lower case
        text = text.lower()
        # Hapus URL
        text = re.sub(r'http\S+', '', text)
        # Hapus tanda baca
        text = re.sub(r'[^\w\s]', '', text)
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        # Hapus emoji
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Menghapus karakter non-ASCII (termasuk emoji)
        return text

    #stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    #slang word
    kbba_dictionary = pd.read_csv('https://raw.githubusercontent.com/azizpatuha/kamus/refs/heads/main/kbba.txt',
                                delimiter='\t', names=['slang', 'formal'], header=None, encoding='utf-8')

    slang_dict = dict(zip(kbba_dictionary['slang'], kbba_dictionary['formal']))

    def convert_slangword(text):
        words = text.split()

        normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
        normalized_text = ' '.join(normalized_words)
        return normalized_text

    #Stopword
    # Download stopwords Bahasa Indonesia from NLTK
    nltk.download('stopwords')
    indonesian_stopwords = stopwords.words('indonesian')

    # **Hapus "tidak" dari daftar stopword**
    indonesian_stopwords.remove('tidak')

    # Cache stopwords list outside the function to avoid reloading every time
    cached_stopwords = indonesian_stopwords

    def remove_stopwords(text):
        words = text.split()
        # Hapus stopwords
        filtered_words = [word for word in words if word not in cached_stopwords]
        normalized_text = ' '.join(filtered_words)
        return normalized_text

    def preprocess_data(data):
        #menghapus kolom yang tidak digunakan
        columns_to_drop = [
                        'reviewId', 'userName', 'userImage', 'score', 
                        'thumbsUpCount', 'reviewCreatedVersion', 'at', 
                        'replyContent', 'repliedAt', 'appVersion'
                    ]
        data.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

        #menampilkan jumlah sentimen
        # st.write("Sentiment Counts:")
        # sentiment_counts = data['sentimen'].value_counts()
        # st.write(sentiment_counts)
        # st.write("TOTAL:", sentiment_counts.sum())

        # mengubah data dari string menjadi integer
        # lb = LabelEncoder()
        # data['sentimen'] = lb.fit_transform(data['sentimen'])
        # data.head()   

        #text processing
        with st.expander("1. Text Preprocessing"):  # Gunakan expander untuk tampilan yang rapi
            data["text_preprocessing"] = data["content"].apply(text_preprocessing)
            st.write("Data setelah text_preprocessing:")
            st.dataframe(data[['content', 'text_preprocessing']].head())
        

        #Stemming
        with st.expander("2. Stemming"):
            data["stemmer"] = data["text_preprocessing"].apply(stemmer.stem)
            st.write("Data setelah stemming:")
            st.dataframe(data[['text_preprocessing', 'stemmer']].head())

        #Slang
        with st.expander("3. Slang"):
            data["slang"] = data["stemmer"].apply(convert_slangword)
            st.write("Data setelah slang:")
            st.dataframe(data[['stemmer', 'slang']].head())

        #Stopword
        with st.expander("4. Stopword"):
            data["stopword_text"] = data["slang"].apply(remove_stopwords)
            st.write("Data setelah stopword:")
            st.dataframe(data[['slang', 'stopword_text']].head())

        # merubah nama kolom
        data = data.rename (columns={'stopword_text': 'comment'})
        data.head()

        # # 0 = positif, 1 = negatif
        # data.sentimen.unique()

        # #melihat data total data
        # data.info()

        #menghapus/menambahkan komentar agar semua data isinya sama rata
        data = data.dropna (subset=['content'])
        # data = data.dropna (subset=['sentimen'])
        data = data.dropna (subset=['text_preprocessing'])
        data = data.dropna (subset=['stemmer'])
        data = data.dropna (subset=['slang'])
        data = data.dropna (subset=['comment'])

        data.info()

        #menghapus kolom yang tidak digunakan setelah dilakukan text processing
        data.drop('content', axis=1, inplace=True)
        data.drop('text_preprocessing', axis=1, inplace=True)
        data.drop('stemmer', axis=1, inplace=True)
        data.drop('slang', axis=1, inplace=True)

        data.head()

        return data

    # Load Model and Tokenizer
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
    model = TFBertForSequenceClassification.from_pretrained("azizpatuha/bert_yt")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #prediksi
    def predict_sentiment(texts):
        predictions = []
        for text in texts:
            # Tokenisasi input teks
            inputs = tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
            # Melakukan prediksi
            outputs = model(**inputs)
            logits = outputs.logits
            # Mengambil argmax dari logits untuk mendapatkan prediksi
            pred = tf.argmax(logits, axis=1).numpy()[0] # Mengonversi tensor ke numpy array
            predictions.append(pred)
        return predictions

    # Generate WordCloud
    def generate_wordcloud(data, sentiment, column):
        words = " ".join(data[data["predicted"] == sentiment][column])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"WordCloud for {sentiment} Sentiment")
        st.pyplot(fig)

    # Streamlit UI
    st.title("Sentiment Analysis Report")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        #membaca file
        data = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.dataframe(data)

        if "content" not in data.columns:
            st.error("The uploaded file must contain a 'content' column.")
        else:
            # Preprocessing
            st.subheader("Preprocessing Data")
            data = preprocess_data(data)

            # Predict Sentiments
            st.subheader("Predicting Sentiments")
            data["predicted"] = predict_sentiment(data["comment"])

        # Assume labels are included in dataset for evaluation
            if "sentimen" in data.columns:
            # Map string labels ke integer
                label_mapping = {"negative": 0, "positive": 1}
                y_true = data["sentimen"].map(label_mapping)

                y_pred = data["predicted"]
                accuracy = accuracy_score(y_true, y_pred)
                class_report = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
                confusion = confusion_matrix(y_true, y_pred)

                st.subheader("Accuracy Score")
                st.write(f"Accuracy: *{accuracy:.2f}*")

                st.subheader("Classification Report")
                st.table(pd.DataFrame(class_report).transpose())

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))

                # Heatmap tanpa angka annotasi
                sns.heatmap(confusion, annot=False, fmt="d", cmap="Blues", xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")

                for i in range(2):
                    for j in range(2):
                        label = ""
                        if i == 1 and j == 1:
                            label = f"{confusion[i, j]}"
                        elif i == 0 and j == 0:
                            label = f"{confusion[i, j]}"
                        elif i == 0 and j == 1:
                            label = f"{confusion[i, j]}"
                        elif i == 1 and j == 0:
                            label = f"{confusion[i, j]}"
                        
                        # Tambahkan deskripsi di dalam kotak
                        ax.text(j + 0.5, i + 0.5, label, ha='center', va='center', fontsize=8, color="black")

                st.pyplot(fig)

            # Display Pie Chart
            st.subheader("Sentiment Distribution")
            sentiment_counts = data["predicted"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            sentiment_counts.plot.pie(
                autopct="%1.1f%%",
                labels=["negative", "positive"],
                ax=ax,
                ylabel=""  # Remove default ylabel
            )
            st.pyplot(fig)

            # Display WordClouds
            st.subheader("WordClouds")
            st.write("Positive Sentiment:")
            generate_wordcloud(data, 1, "comment")

            st.write("Negative Sentiment:")
            generate_wordcloud(data, 0, "comment")

            # Downloadable CSV
            st.subheader("Download Results")
            csv = data.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predicted_results.csv", mime="text/csv")

            # Downloadable XLSX
            st.subheader("Download Hasil")
            def convert_df(data):
                output = io.BytesIO()
                data.to_excel(output, index=False)
                output.seek(0)
                return output

            xlsx = convert_df(data)

            st.download_button(
                label="Download XLSX",
                data=xlsx,
                file_name="predicted_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

if __name__ == "__main__":
    admin_upload_yt()  
