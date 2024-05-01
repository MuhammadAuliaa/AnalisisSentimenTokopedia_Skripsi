if tombol:
        with st.spinner("Sedang melatih model..."):
            # Bagi data menjadi atribut dan label
            X = df['Ulasan']
            y = df['Sentimen']

            # Inisialisasi CountVectorizer
            cv = CountVectorizer()

            # Transformasi teks menjadi representasi numerik menggunakan CountVectorizer
            X_transformed = cv.fit_transform(X)

            # Bagi data menjadi atribut dan label
            X = df['Ulasan']
            y = df['Sentimen']

            # Encoding label kelas menjadi nilai biner
            y = np.where(y == 'Positif', 1, 0)

            # Inisialisasi CountVectorizer
            cv = CountVectorizer()

            # Transformasi teks menjadi representasi numerik menggunakan CountVectorizer
            X_transformed = cv.fit_transform(X)

            # Bagi data menjadi data training dan data testing
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, stratify=y, random_state=42)

            # Inisialisasi model Naive Bayes
            nb_model = MultinomialNB()

            # Latih model menggunakan data training
            nb_model.fit(X_train, y_train)

            # Prediksi menggunakan data testing
            y_pred = nb_model.predict(X_test)

            # Hitung akurasi model
            accuracy = accuracy_score(y_test, y_pred)

            # Simpan model ke file
            joblib.dump(nb_model, 'model.pkl')

            # Tampilkan akurasi
            st.success("Pelatihan Model Naive Bayes Selesai!")








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

            # Lakukan SMOTE untuk mengatasi ketidakseimbangan kelas
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

            # Bagi data menjadi data training dan data testing
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

            # Inisialisasi model Naive Bayes
            nb_model = MultinomialNB()

            # Latih model menggunakan data training
            nb_model.fit(X_train, y_train)

            # Prediksi menggunakan data testing
            y_pred = nb_model.predict(X_test)

            # Hitung akurasi model
            accuracy = accuracy_score(y_test, y_pred)

            # Simpan model ke file
            joblib.dump(nb_model, 'model.pkl')

            # Tampilkan akurasi
            st.success("Pelatihan Model Naive Bayes Selesai!")

            # Visualisasi distribusi kelas sebelum dan sesudah SMOTE
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.bar(np.unique(y), np.bincount(y), color=['#d63636', '#3691d6'])
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(['Negatif', 'Positif'])
            ax1.set_title('Sebelum SMOTE')
            ax1.set_xlabel('Kelas')
            ax1.set_ylabel('Jumlah')

            ax2.bar(np.unique(y_resampled), np.bincount(y_resampled), color=['#d63636', '#3691d6'])
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['Negatif', 'Positif'])
            ax2.set_title('Setelah SMOTE')
            ax2.set_xlabel('Kelas')
            ax2.set_ylabel('Jumlah')

            st.pyplot(fig)

            

            

                        