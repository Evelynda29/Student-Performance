import joblib

with open('model_graduation.pkl', 'rb') as file:
    model = joblib.load(file)


# Judul aplikasi
st.title("Prediksi Kategori Waktu Lulus Mahasiswa")

# Deskripsi singkat
st.write("Masukkan data siswa untuk memprediksi apakah mereka akan lulus tepat waktu atau terlambat.")

# Load model
try:
    with open('model_graduation.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model 'model_graduation.pkl' tidak ditemukan. Harap pastikan file model ada di direktori yang sama.")
    st.stop()

# Input dari pengguna
new_ACT = st.number_input("Masukkan nilai ACT composite score:", min_value=0.0, max_value=36.0, step=0.1)
new_SAT = st.number_input("Masukkan nilai SAT total score:", min_value=400.0, max_value=1600.0, step=10.0)
new_GPA = st.number_input("Masukkan nilai rata-rata SMA:", min_value=0.0, max_value=4.0, step=0.01)
new_income = st.number_input("Masukkan pendapatan orang tua (USD):", min_value=0.0, step=100.0)
new_education = st.number_input("Masukkan tingkat pendidikan orang tua (angka):", min_value=0.0, step=1.0)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Buat DataFrame dari input
        new_data_df = pd.DataFrame(
            [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
            columns=[
                'ACT composite score', 
                'SAT total score', 
                'high school gpa', 
                'parental income', 
                'parent_edu_numerical'
            ]
        )

        # Lakukan prediksi
        predicted_code = model.predict(new_data_df)[0]

        # Mapping hasil prediksi
        label_mapping = {0: 'On Time', 1: 'Late'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"Prediksi: Mahasiswa akan lulus **{predicted_label}**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
