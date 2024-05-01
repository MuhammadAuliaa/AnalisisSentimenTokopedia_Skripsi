import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from streamlit_option_menu import option_menu
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from tabulate import tabulate
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from codinganStracth import MultinomialNaiveBayes
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

selected = option_menu(None, ["Crawling", "Dataset", 'Preprocessing', 'TF-IDF' ,'Machine Learning', 'Prediction'], 
    icons=['cloud-upload', "archive", 'gear', 'activity', 'kanban', 'kanban'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

norm= {" dgn " : " dengan ", ' seller ': ' penjual ',' service ':' pelayanan ', ' tp ':' tapi ', ' recommended ':' rekomendasi ', ' kren ':' keren ', ' kereen ':' keren ', ' mantab ': ' keren ',' matching ':' sesuai ','happy':' senang ','original': 'asli ','ori':'asli ', "trusted" : "terpercaya", "angjaaaassss":"keren", " gue ": " saya ", "bgmn ":" bagaimana ", ' tdk':' tidak ', ' blum ':' belum ', 'mantaaaaaaaappp':' bagus ', 'mantaaap':'bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', ' dg ':' dengan ', 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' recommended':' rekomen ', 'recomend':' rekomen ', 'good':' bagus ', " dgn " : " dengan ", " gue ": " saya ", " dgn ":" dengan ", "bgmn ":" bagaimana ", ' tdk':' tidak ', 
' blum ':' belum ', "quality":"kualitas", 'baguss':'bagus', 'overall' : 'akhirnya', 'mantaaaaaaaappp':' bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', 
 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' real ': ' asli ', ' bnb ': ' baru ' ,
' recommended':' rekomen ', 'recomend':' rekomen ', 'good':'bagus',
'eksis ':'ada ', 'beenilai ':'bernilai ', ' dg ':' dengan ', ' ori ':' asli ', ' setting ':' atur ', " free ":" gratis ",
' yg ':' yang ', 't4 ':'tempat', ' awat ':' awet', ' mantep ':' bagus ', 'mantapp':'bagus', 
'kl ':'kalo', ' k ':' ke ', 'plg ':'pulang ', 'ajah ':'aja ', 'bgt':'banget', 'lbh ':'lebih', 'ayem':'tenang','dsana ':'disana ', 'lg':' lagi',
'pas ':'saat ', ' bnib ': ' baru ', 
' nggak ':' tidak ', 'karna ':'karena ', 'utk ':'untuk ',
' dn ':' dan ', ' mlht ':' melihat ', ' pd ':' pada ', 'mndngr ':'mendengar ', 'crita':'cerita', ' dpt ':' dapat ', ' mksh ':' terima kasih ', ' sellerrrr':' penjual', 'ori ':'asli ', ' new ':' baru ',
'sejrh':'sejarah', 'mnmbh ':'menambah ', 'sayapun':'saya', 'thn ':'tahun ', 'good':'bagus', ' awettt':' awet',
'halu ':'halusinasi ', ' nyantai ':' santai ', 'plus ':'dan ',
' ayang ':' sayang ', ' Rekomendded ':' direkomendasikan ', ' now ': ' sekarang ', 'slalu ':'selalu ', 'photo ': 'foto ', 'slah ':'salah ', 'krn':'karena', ' ga ':' tidak ', 'ok ':'oke ', ' meski':' mesti', ' para ':'parah', ' nawarin':' menawari', 'socmed':'sosial media',
' sya ':' saya ', 'siip':'bagus', ' bny ':' banyak ', ' tdk ':' tidak ', ' byk ':' banyak ', 
' pool ':' sekali ', " pgn ":" ingin ", " gue ":" saya ", " bgmn ":" bagaimana ", " ga ":" tidak ", 
" gak ":" tidak ", " dr ":" dari ", " yg ":" yang ", " lu ":" kamu ", " sya ":" saya ", 
" lancarrr ":" lancar ", " kayak ":" seperti ", " ngawur ":" sembarangan ", " k ":" ke ", 
" luasss ":" luas ", " sy ":" saya ", " thn ":" tahun ", " males ":" malas ",
" tgl ":" tanggal ", " lg ":" lagi ", " bgt ":" banget ",' gua ':' saya ', '\n':' ', ' tpi ':' tapi ', ' standar ':' biasa ', ' standart ': ' biasa ', ' sdh ':' sudah ', ' n ':' dan ', ' gk ': ' tidak ', ' mengecwakan ':' mengecewakan ', ' d ':' di '}

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

def clean(text):
  text = text.strip()
  text = text.lower()
  text = re.sub(r'[^a-zA-Z]+', ' ', text)
  return text

def labeling(rating):
    if rating == '4' or rating == '5':
        return 'Positif'
    else:
        return 'Negatif'

def tokenisasi(text):
    return text.split() 
    
def stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    text = ' '.join(text)
    stemmed_text = stemmer.stem(text)
    return stemmed_text

if selected == 'Crawling':
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Crawling Data")

    # https://www.tokopedia.com/homedoki/review
    # https://www.tokopedia.com/pengrajincom/review 
    url = st.text_input("URL")
    tombol = st.button("Crawling")  
    if tombol :
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        rating_mapping = {
        'bintang 5': 5,
        'bintang 4': 4,
        'bintang 3': 3,
        'bintang 2': 2,
        'bintang 1': 1
        }

        data = []
        for i in range(0, 2):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

            for container in containers:
                review_container = container.find('span', attrs={'data-testid': 'lblItemUlasan'})
                review = review_container.text.strip() if review_container else 'Tidak ada ulasan'

                nama_produk = container.find('p', attrs={'class': 'e1qvo2ff8'})
                produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'

                nama_pelanggan = container.find('span', attrs={'class': 'name'})
                pelanggan = nama_pelanggan.text.strip() if nama_pelanggan else 'Customer tidak ditemukan'

                rating_container = container.find('div', attrs={'data-testid': 'icnStarRating'})
                rating_label = rating_container['aria-label'] if rating_container else 'Tidak ada rating'
                rating = rating_mapping.get(rating_label, 'Tidak ada rating')

                data.append(
                    (pelanggan, produk, review, rating)
                )
        
            time.sleep(2)
            driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']").click()
            time.sleep(3)

        df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])   
        df.to_csv('data/Tokopedia.csv', index=False)
        driver.close()
        # Mengubah tipe data 'Rating' & memanggil attribut tertentu
        showDf = pd.read_csv('data/Tokopedia.csv', dtype={'Rating': 'object'})
        showDf = showDf[['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating']]

        # Menampilkan data
        st.write(showDf)

        # Pengkondisian Alert Crawling
        jumlah_data = len(showDf)
        if jumlah_data > 0:
            st.success(f"Crawling {jumlah_data} Baris Data Berhasil!")
        else:
            st.warning("Crawling Data Gagal")

if selected == 'Dataset':
    st.title("Dataset Tokopedia :")
    try:
       df = pd.read_csv("data/Tokopedia.csv", dtype={'Rating':'object'})
       df = df[['Nama Pelanggan','Produk','Ulasan', 'Rating']]
       st.dataframe(df)
    except FileNotFoundError:
       st.write("File not found, please check the file path...")


if selected == 'Preprocessing':
    st.title("Preprocessing Data")
    df = pd.read_csv("data/Tokopedia.csv", dtype={'Rating': 'object'})
    df = df[['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating']]
    st.write(df)
    
    preprocessing = st.button("Preprocessing")
    if preprocessing:
        with st.spinner('Sedang melakukan preprocessing...'):
            time.sleep(2)
            # st.success("Preprocessing Berhasil & Data Disimpan!")
            df['Ulasan'] = df['Ulasan'].fillna('')
            df['Ulasan'] = df['Ulasan'].apply(clean)
            st.write('')
            st.write(f'--------------------------------------------------------------  CLEANING  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(normalisasi)
            st.write('')
            st.write(f'--------------------------------------------------------------  NORMALIZE  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(stopword)
            st.write('')
            st.write('--------------------------------------------------------------  STOPWORD  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(tokenisasi)
            st.write('')
            st.write(f'--------------------------------------------------------------  TOKENIZE  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            # Melakukan stemming pada kolom "Ulasan"
            df['Ulasan'] = df['Ulasan'].apply(stemming)
            st.write('')
            st.write('--------------------------------------------------------------  STEMMING --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Sentimen'] = df['Rating'].apply(labeling)
            st.write('')
            st.write(f'--------------------------------------------------------------  LABELING  --------------------------------------------------------------')
            st.write(df[['Ulasan', 'Sentimen']])
            
        # Menghentikan animasi loading
        st.spinner(False)
        df = df[['Ulasan', 'Sentimen']]
        df.to_csv('data/Preprocessing.csv', index=False)

        # Pengkondisian Alert Crawling
        jumlah_data = len(df)
        if jumlah_data > 0:
            st.success(f"Preprocessing {jumlah_data} Baris & Download Data Berhasil !")
        else:
            st.warning("Preprocessing Data Gagal")    

if selected == 'TF-IDF':                 
        # Load the data from Preprocessing.csv
        df = pd.read_csv('data/Preprocessing.csv')
        df = df.dropna()

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit the vectorizer to the 'Ulasan' column
        tfidf_matrix = vectorizer.fit_transform(df['Ulasan'])

        # Convert the TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Streamlit app
        st.title("Kalkulator TF-IDF")

        # Display the TF-IDF DataFrame
        st.write(tfidf_df)

        # Display the top 10 most important words
        st.subheader("10 Kata Paling Penting Teratas")
        top_words = tfidf_df.sum().sort_values(ascending=False).head(10)

        # Add attributes to the top words DataFrame
        top_words_df = pd.DataFrame(top_words, columns=["tf_idf"])
        top_words_df["kata"] = top_words_df.index
        top_words_df.reset_index(drop=True, inplace=True)

        # Calculate the count of each word in the 'Ulasan' column
        word_counts = df['Ulasan'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0)
        top_words_df["jumlah_kemunculan"] = word_counts[top_words_df["kata"]].values

        # Reorder columns
        top_words_df = top_words_df[["kata", "jumlah_kemunculan", "tf_idf"]]  # Add 'tf_idf' column

        # Display the top words DataFrame with attributes
        st.write(top_words_df)
 

if selected == 'Machine Learning':
    df = pd.read_csv('data/Preprocessing.csv')
    df = df.dropna()
    # df = pd.concat([df[df['Sentimen'] == 'Negatif'].head(257), df[df['Sentimen'] == 'Positif'].head(400)])

    # Menghitung jumlah baris data dengan sentimen positif dan negatif sebelum SMOTE
    num_positive_before = df[df['Sentimen'] == 'Positif'].shape[0]
    num_negative_before = df[df['Sentimen'] == 'Negatif'].shape[0]

    # Menyiapkan data untuk visualisasi sebelum SMOTE
    sentimen_before = [f'Positif: {num_positive_before}', f'Negatif: {num_negative_before}']
    jumlah_data_before = [num_positive_before, num_negative_before]

    # Membuat bar chart dengan warna tiap label yang berbeda untuk sebelum SMOTE
    colors_before = ['#3691d6', '#d63636']
    plt.bar(sentimen_before, jumlah_data_before, color=colors_before)
    plt.xlabel('Sentimen (Sebelum SMOTE)')
    plt.ylabel('Jumlah Data')
    plt.title('Perbandingan Label Sentimen Sebelum SMOTE')

    # Menampilkan chart sebelum SMOTE menggunakan Streamlit
    st.pyplot(plt)

    smote_option = st.selectbox("Pilih opsi untuk menggunakan SMOTE & IMBLEARN atau tidak:", ("Tanpa SMOTE & IMBLEARN", "Dengan SMOTE & IMBLEARN"))
    tombol = st.button("Training Data")
    if tombol:
        with st.spinner("Sedang melatih model..."):
            # Bagi data menjadi atribut dan label
            X = df['Ulasan']
            y = df['Sentimen']

            # Encoding label kelas menjadi nilai biner
            y = np.where(y == 'Positif', 1, 0)

            # Inisialisasi CountVectorizer
            cv = CountVectorizer()

            # Transformasi teks menjadi representasi numerik menggunakan CountVectorizer
            X_transformed = cv.fit_transform(X)

            if smote_option == "Dengan SMOTE & IMBLEARN":
                # Lakukan SMOTE untuk mengatasi ketidakseimbangan kelas
                over = RandomOverSampler(sampling_strategy=0.5, random_state=5)
                under = RandomUnderSampler(sampling_strategy=0.85, random_state=5)
                X_resampled, y_resampled = over.fit_resample(X_transformed, y)
                X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
            else:
                X_resampled, y_resampled = X_transformed, y

            # Menghitung jumlah baris data dengan sentimen positif dan negatif setelah SMOTE
            num_positive_after = np.count_nonzero(y_resampled == 1)
            num_negative_after = np.count_nonzero(y_resampled == 0)

            # Menyiapkan data untuk visualisasi setelah SMOTE
            sentimen_after = [f'Positif: {num_positive_after}', f'Negatif: {num_negative_after}']
            jumlah_data_after = [num_positive_after, num_negative_after]

            # Membuat bar chart dengan warna tiap label yang berbeda untuk setelah SMOTE
            colors_after = ['#3691d6', '#d63636']
            # plt.bar(sentimen_after, jumlah_data_after, color=colors_after)
            plt.bar(sentimen_after, jumlah_data_after, color=colors_after)
            plt.xlabel('Sentimen (Setelah SMOTE)')
            plt.ylabel('Jumlah Data')
            plt.title('Perbandingan Label Sentimen Setelah SMOTE')

            # Menampilkan chart setelah SMOTE menggunakan Streamlit
            st.pyplot(plt)

            # Menghitung TF-IDF dari data teks
            tfidf = TfidfTransformer()
            X_tfidf = tfidf.fit_transform(X_resampled)

            # Bagi data menjadi data training dan data testing
            X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

            # Inisialisasi model Naive Bayes
            nb_model = MultinomialNB()

            # Latih model menggunakan data training
            nb_model.fit(X_train.toarray(), y_train)

            # Prediksi menggunakan data testing
            y_pred = nb_model.predict(X_test.toarray())

            # Hitung akurasi data training
            y_train_pred = nb_model.predict(X_train.toarray())
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # Hitung akurasi data testing
            test_accuracy = accuracy_score(y_test, y_pred)

            # Simpan model ke file
            joblib.dump(nb_model, 'model.pkl')

            st.success("Pelatihan Model Naive Bayes Selesai!")

            # Menghitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Definisikan label kelas
            labels = ['Negatif', 'Positif']

            # Visualisasi confusion matrix menggunakan heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            # Tampilkan classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            # Ubah classification report menjadi tabel
            table = tabulate(report_df, headers='keys', tablefmt='psql')

            # Tampilkan tabel classification report
            st.write("Classification Report :")
            # Tampilkan hasil akurasi
            st.write(f"Akurasi Data Training: {train_accuracy}")
            st.write(f"Akurasi Data Testing: {test_accuracy}")
            st.code(table)

if selected == 'Prediction':
    st.title("Prediksi Sentimen")
    input_text = st.text_area("Masukkan Teks Ulasan")
    df = pd.read_csv('data/Preprocessing.csv')
    df = df.dropna()

    if st.button("Prediksi"):
        if input_text:
            # Load model dari file
            nb_model = joblib.load('model.pkl')

            # Inisialisasi CountVectorizer dan latih dengan data latihan
            cv = CountVectorizer()
            X_transformed = cv.fit_transform(df['Ulasan'])

            # Transformasi teks input menjadi representasi numerik menggunakan CountVectorizer yang sudah dilatih
            input_transformed = cv.transform([input_text])

            # Prediksi sentimen menggunakan model yang sudah dilatih
            prediction = nb_model.predict(input_transformed.toarray())
            sentimen_prediksi = 'Positif' if prediction[0] == 1 else 'Negatif'

            # Menghitung probabilitas hasil prediksi sentimen
            probabilities = nb_model.predict_proba(input_transformed.toarray())
            proba_positif = probabilities[0, 1]
            proba_negatif = probabilities[0, 0]

            # Tampilkan hasil prediksi
            st.success("Hasil Prediksi:")
            # st.write("Sentimen:", sentimen_prediksi)
            st.write("Positif:", round(proba_positif * 100, 2), "%")
            st.write("Negatif:", round(proba_negatif * 100, 2), "%")

            # Animasi presentasi nilai sentimen
            fig, ax = plt.subplots()
            ax.axis('off')

            # Menentukan gambar yang akan ditampilkan berdasarkan sentimen prediksi
            if sentimen_prediksi == 'Positif':
                img_path = 'img/smile.png'
            else:
                img_path = 'img/angry.png'

            img = plt.imread(img_path)
            imagebox = OffsetImage(img, zoom=2)  # Mengubah zoom menjadi 0.2 untuk ukuran yang lebih kecil
            ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False)
            ax.add_artist(ab)

            def update(frame):
                ab.xybox = (frame / 100, 1)
                return ab,

            ani = animation.FuncAnimation(fig, update, frames=50, interval=50, blit=True)

            # Menampilkan animasi
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning("Masukkan teks ulasan sebelum melakukan prediksi!")
