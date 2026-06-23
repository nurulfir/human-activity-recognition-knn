# APLIKASI: Deteksi Aktivitas Manusia dari Sensor Smartphone
# ALGORITMA : K-Nearest Neighbor (K-NN)
# ==========================================================
# Cara kerja K-NN secara sederhana:
# Bayangkan kita masuk ke ruangan kelas baru. Untuk menebak
# "anak ini termasuk kelompok mana?", kita lihat K teman
# terdekat di sekitarnya, lalu ikut suara terbanyak dari
# mereka. K-NN melakukan hal yang sama, tapi "dekat" diukur
# dari kemiripan angka-angka sensor.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Deteksi Aktivitas - K-NN", page_icon="🏃", layout="wide")


# 1. MEMUAT DATA
@st.cache_data
def load_data():
    folder = os.path.dirname(os.path.abspath(__file__))
    X_train = pd.read_csv(os.path.join(folder, "train/X_train.txt"), sep=r"\s+", header=None)
    X_test = pd.read_csv(os.path.join(folder, "test/X_test.txt"), sep=r"\s+", header=None)
    y_train = pd.read_csv(os.path.join(folder, "train/y_train.txt"), sep=r"\s+", header=None)
    y_test = pd.read_csv(os.path.join(folder, "test/y_test.txt"), sep=r"\s+", header=None)
    activities = pd.read_csv(
        os.path.join(folder, "activity_labels.txt"), sep=r"\s+", header=None,
        names=["id", "label"]
    )
    return X_train, X_test, y_train, y_test, activities


@st.cache_data
def preprocess(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_arr = y_train.values.ravel()
    y_test_arr = y_test.values.ravel()
    return X_train_scaled, X_test_scaled, y_train_arr, y_test_arr


X_train, X_test, y_train_df, y_test_df, activities = load_data()
X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X_train, X_test, y_train_df, y_test_df)
activity_map = dict(zip(activities["id"], activities["label"]))
nama_aktivitas = [activity_map[i].replace("_", " ").title() for i in range(1, 7)]


# SIDEBAR / MENU NAVIGASI
st.sidebar.title("🏃 Deteksi Aktivitas Manusia")
st.sidebar.caption("Algoritma K-Nearest Neighbor (K-NN)")
st.sidebar.markdown("[Sumber Dataset (UCI)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)")
st.sidebar.divider()

halaman = st.sidebar.radio("Pilih Tahapan", [
    "🏠 Beranda",
    "1️⃣ Mengenal Data",
    "2️⃣ Menyiapkan Data",
    "3️⃣ Melatih Model K-NN",
    "4️⃣ Menguji Hasil Model",
    "5️⃣ Coba Prediksi Sendiri",
])


# HALAMAN: BERANDA
if halaman == "🏠 Beranda":
    st.title("Deteksi Aktivitas Manusia dari Sensor Smartphone")
    st.subheader("Menggunakan Algoritma K-Nearest Neighbor (K-NN)")

    st.markdown("""
    Aplikasi ini menjawab satu pertanyaan sederhana:

    > **Dari getaran sensor di smartphone (akselerometer & giroskop),
    > bisakah komputer menebak apa yang sedang dilakukan orang itu?**

    Misalnya: sedang **berjalan**, **naik tangga**, **turun tangga**,
    **duduk**, **berdiri**, atau **berbaring**.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Bagaimana cara kerjanya?
        1. **30 orang** memakai smartphone di pinggang sambil melakukan 6 aktivitas.
        2. Sensor merekam gerakan, lalu diubah menjadi **561 angka** (fitur) per rekaman.
        3. Algoritma **K-NN** belajar dari data lama, lalu menebak aktivitas
           data baru dengan cara **"melihat tetangga terdekat"**.

        ### Kenapa pakai K-NN?
        K-NN itu seperti menebak selera makan seseorang dengan melihat
        beberapa orang yang paling mirip dengannya — tidak perlu rumus rumit,
        cukup bandingkan kemiripan.
        """)
    with col2:
        st.info(
            "**Ringkasan Data**\n\n"
            f"- Data latih: **{X_train.shape[0]:,}** baris\n"
            f"- Data uji: **{X_test.shape[0]:,}** baris\n"
            f"- Jumlah fitur: **{X_train.shape[1]}**\n"
            "- Jumlah kelas: **6 aktivitas**"
        )

    st.divider()
    st.markdown("**Urutan menjelajah aplikasi ini (lihat menu di kiri):**")
    st.markdown("""
    1. **Mengenal Data** — lihat seperti apa data mentahnya
    2. **Menyiapkan Data** — menyamakan skala angka sebelum diproses
    3. **Melatih Model K-NN** — komputer "belajar" dari data
    4. **Menguji Hasil Model** — seberapa akurat tebakan komputer
    5. **Coba Prediksi Sendiri** — masukkan angka sendiri, lihat hasil tebakannya
    """)


# HALAMAN 1: MENGENAL DATA
elif halaman == "1️⃣ Mengenal Data":
    st.title("1️⃣ Mengenal Data")
    st.markdown("Sebelum diolah, kita lihat dulu bentuk datanya seperti apa.")

    with st.expander("📌 Penjelasan Singkat", expanded=True):
        st.markdown("""
        - Setiap baris = **satu rekaman gerakan** dari satu orang.
        - Setiap kolom = **satu angka hasil olahan sensor** (561 kolom total).
        - Tidak ada data yang kosong/hilang — datanya sudah bersih dari sumbernya.
        - Data sudah dipisah: sebagian untuk **belajar** (training),
          sebagian untuk **diuji** (testing) — supaya penilaiannya adil.
        """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Latih (Training)", f"{X_train.shape[0]:,} baris")
    col2.metric("Data Uji (Testing)", f"{X_test.shape[0]:,} baris")
    col3.metric("Jumlah Fitur", X_train.shape[1])

    st.subheader("Contoh Data Mentah (5 baris pertama)")
    st.caption("Angka-angka ini hasil olahan getaran sensor — sulit dibaca manusia, tapi mudah dibaca komputer.")
    st.dataframe(X_train.head(5).iloc[:, :8], use_container_width=True)
    st.caption("*(hanya 8 dari 561 kolom yang ditampilkan)*")

    st.subheader("6 Jenis Aktivitas yang Akan Ditebak")
    tampil_aktivitas = activities.copy()
    tampil_aktivitas.columns = ["Kode", "Nama Aktivitas"]
    st.dataframe(tampil_aktivitas, use_container_width=True, hide_index=True)

    st.subheader("Jumlah Data per Aktivitas")
    jumlah_train = pd.Series(y_train).value_counts().sort_index()
    tabel_jumlah = pd.DataFrame({
        "Aktivitas": [activity_map[i] for i in jumlah_train.index],
        "Jumlah Data Latih": jumlah_train.values
    })
    st.dataframe(tabel_jumlah, use_container_width=True, hide_index=True)
    st.caption("Jumlah tiap aktivitas cukup seimbang, jadi model tidak akan bias ke satu aktivitas saja.")


# HALAMAN 2: MENYIAPKAN DATA
elif halaman == "2️⃣ Menyiapkan Data":
    st.title("2️⃣ Menyiapkan Data (Preprocessing)")

    with st.expander("📌 Penjelasan Singkat", expanded=True):
        st.markdown("""
        **Masalahnya:** sebagian angka sensor bersekala kecil (misal 0.01),
        sebagian lagi bersekala besar (misal 100). Kalau dibiarkan begitu,
        K-NN akan **salah fokus** ke angka yang skalanya besar saja.

        **Solusinya — Standardisasi (StandardScaler):**
        Semua angka diubah ke skala yang sama (rata-rata = 0).
        Ibaratnya, sebelum membandingkan tinggi badan dan berat badan,
        kita ubah dulu keduanya jadi "skor" yang setara — biar adil dibandingkan.
        """)

    st.subheader("Langkah 1 — Cek Data Kosong")
    col1, col2 = st.columns(2)
    col1.success(f"✅ Data latih: {X_train.isnull().sum().sum()} nilai kosong")
    col2.success(f"✅ Data uji: {X_test.isnull().sum().sum()} nilai kosong")
    st.caption("Tidak ada nilai kosong sama sekali — data sudah bersih, tidak perlu diperbaiki.")

    st.subheader("Langkah 2 — Standardisasi Skala")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sebelum** distandardisasi:")
        st.dataframe(X_train.iloc[:5, :4].describe().loc[["mean", "std", "min", "max"]], use_container_width=True)
    with col2:
        st.write("**Sesudah** distandardisasi:")
        st.dataframe(
            pd.DataFrame(X_train_scaled[:, :4]).describe().loc[["mean", "std", "min", "max"]],
            use_container_width=True
        )
    st.caption("Setelah distandardisasi, rata-rata setiap kolom mendekati 0 — semua fitur jadi 'setara'.")


# HALAMAN 3: MELATIH MODEL K-NN
elif halaman == "3️⃣ Melatih Model K-NN":
    st.title("3️⃣ Melatih Model K-NN")

    with st.expander("📌 Penjelasan Singkat", expanded=True):
        st.markdown("""
        **Cara kerja K-NN:**
        1. Ambil satu data baru yang belum diketahui aktivitasnya.
        2. Cari **K data latih yang paling mirip** (paling "dekat") dengannya.
        3. Lihat aktivitas apa yang **paling banyak** muncul di antara K tetangga itu.
        4. Itulah tebakan aktivitasnya.

        **Tentang nilai K:**
        - K terlalu kecil (misal K=1) → model terlalu "hafal", gampang salah pada data baru.
        - K terlalu besar → model jadi "rata-rata" semua, kurang tajam membedakan.
        - Biasanya ada nilai K yang paling pas — bisa dicari otomatis di bawah.
        """)

    st.subheader("Langkah A (Opsional) — Cari Nilai K Terbaik Otomatis")
    if st.button("🔍 Cari K Terbaik", use_container_width=True):
        with st.spinner("Mencoba K = 1 sampai 15..."):
            daftar_k = range(1, 16)
            daftar_akurasi = []
            progres = st.progress(0)
            for i, k_coba in enumerate(daftar_k):
                model_coba = KNeighborsClassifier(n_neighbors=k_coba, n_jobs=-1)
                model_coba.fit(X_train_scaled, y_train)
                pred_coba = model_coba.predict(X_test_scaled)
                daftar_akurasi.append(accuracy_score(y_test, pred_coba))
                progres.progress((i + 1) / len(daftar_k))

            k_terbaik = list(daftar_k)[int(np.argmax(daftar_akurasi))]
            st.session_state["k_terbaik"] = k_terbaik
            st.session_state["grafik_k"] = (list(daftar_k), daftar_akurasi, k_terbaik)

    if "grafik_k" in st.session_state:
        daftar_k, daftar_akurasi, k_terbaik = st.session_state["grafik_k"]
        st.success(f"✅ Nilai K paling akurat: **K = {k_terbaik}** (akurasi {max(daftar_akurasi):.2%})")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(daftar_k, daftar_akurasi, "o-", color="#2563eb")
        ax.axvline(k_terbaik, color="#dc2626", ls="--", label=f"K terbaik = {k_terbaik}")
        ax.set_xlabel("Nilai K")
        ax.set_ylabel("Akurasi")
        ax.set_title("Akurasi untuk Setiap Nilai K")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Langkah B — Latih Model dengan Nilai K Pilihanmu")

    k_default = st.session_state.get("k_terbaik", 5)
    k = st.slider("Pilih jumlah tetangga (K)", min_value=1, max_value=20, value=k_default)

    if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
        with st.spinner(f"Melatih model K-NN dengan K = {k} ..."):
            model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            st.session_state["model"] = model
            st.session_state["y_pred"] = y_pred
            st.session_state["k_dipakai"] = k

            akurasi = accuracy_score(y_test, y_pred)
            st.success(f"🎉 Model selesai dilatih! Akurasi pada data uji: **{akurasi:.2%}**")

    if "model" in st.session_state:
        st.info(f"Model aktif saat ini menggunakan **K = {st.session_state['k_dipakai']}**. "
                "Lanjut ke menu **4️⃣ Menguji Hasil Model** untuk melihat detail performanya.")


# HALAMAN 4: MENGUJI HASIL MODEL
elif halaman == "4️⃣ Menguji Hasil Model":
    st.title("4️⃣ Menguji Hasil Model")

    with st.expander("📌 Penjelasan Singkat", expanded=True):
        st.markdown("""
        - **Akurasi** → dari semua tebakan, berapa persen yang benar.
        - **Precision** → dari semua yang ditebak "aktivitas A", berapa persen yang
          memang benar-benar aktivitas A.
        - **Recall** → dari semua data yang sebenarnya aktivitas A, berapa persen
          yang berhasil ditebak benar.
        - **F1-Score** → gabungan Precision & Recall jadi satu angka.
        - **Confusion Matrix** → tabel yang menunjukkan aktivitas mana yang
          paling sering tertukar dengan aktivitas lain.
        """)

    if "model" not in st.session_state:
        st.warning("⚠️ Model belum dilatih. Silakan ke menu **3️⃣ Melatih Model K-NN** terlebih dahulu, "
                    "atau klik tombol di bawah untuk memakai pengaturan default.")
        if st.button("Latih Model Default (K=5)"):
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            st.session_state["model"] = model
            st.session_state["y_pred"] = model.predict(X_test_scaled)
            st.session_state["k_dipakai"] = 5
            st.rerun()
    else:
        y_pred = st.session_state["y_pred"]
        st.info(f"Menampilkan hasil model dengan **K = {st.session_state['k_dipakai']}**")

        st.subheader("Ringkasan Performa")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2%}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
        col4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

        st.subheader("Confusion Matrix (Tabel Tebakan vs Kenyataan)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nama_aktivitas, yticklabels=nama_aktivitas,
            linewidths=0.5, ax=ax
        )
        ax.set_xlabel("Tebakan Model")
        ax.set_ylabel("Aktivitas Sebenarnya")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close(fig)

        benar = int(np.trace(cm))
        total = len(y_test)
        st.markdown(f"""
        **Cara membaca tabel di atas:**
        Diagonal (kotak warna paling gelap dari kiri-atas ke kanan-bawah) = tebakan yang **benar**.
        Di luar diagonal = tebakan yang **salah** (tertukar dengan aktivitas lain).

        ➡️ Total **{benar:,} dari {total:,}** data ({benar/total:.1%}) berhasil ditebak dengan benar.

        Kesalahan paling umum biasanya terjadi antara aktivitas yang gerakannya mirip,
        misalnya **Duduk vs Berdiri**, atau **Berjalan vs Naik Tangga**.
        """)

        with st.expander("Lihat detail per aktivitas (tabel lengkap)"):
            report = classification_report(
                y_test, y_pred, target_names=nama_aktivitas, output_dict=True, digits=3
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


# HALAMAN 5: COBA PREDIKSI SENDIRI
elif halaman == "5️⃣ Coba Prediksi Sendiri":
    st.title("5️⃣ Coba Prediksi Sendiri")
    st.markdown("""
    Pilih salah satu data uji di bawah, lalu lihat tebakan model dibandingkan
    dengan jawaban yang sebenarnya. Ini menunjukkan model **bekerja secara nyata**,
    bukan hanya teori.
    """)

    if "model" not in st.session_state:
        st.warning("⚠️ Model belum dilatih. Silakan ke menu **3️⃣ Melatih Model K-NN** terlebih dahulu, "
                    "atau klik tombol di bawah untuk memakai pengaturan default.")
        if st.button("Latih Model Default (K=5)"):
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            st.session_state["model"] = model
            st.session_state["y_pred"] = model.predict(X_test_scaled)
            st.session_state["k_dipakai"] = 5
            st.rerun()
    else:
        model = st.session_state["model"]

        nomor_data = st.number_input(
            f"Pilih nomor data uji (0 sampai {len(X_test_scaled) - 1})",
            min_value=0, max_value=len(X_test_scaled) - 1, value=0, step=1
        )

        if st.button("🔮 Tebak Aktivitas", type="primary", use_container_width=True):
            data_pilihan = X_test_scaled[nomor_data].reshape(1, -1)
            label_asli = int(y_test[nomor_data])
            label_tebakan = int(model.predict(data_pilihan)[0])
            proba = model.predict_proba(data_pilihan)[0]

            aktivitas_asli = activity_map[label_asli]
            aktivitas_tebakan = activity_map[label_tebakan]

            col1, col2 = st.columns(2)
            col1.metric("Aktivitas Sebenarnya", aktivitas_asli)
            col2.metric("Tebakan Model", aktivitas_tebakan)

            if label_asli == label_tebakan:
                st.success("✅ Model menebak dengan BENAR!")
            else:
                st.error("❌ Model menebak SALAH — wajar, tidak ada model yang 100% sempurna.")

            st.subheader("Tingkat Keyakinan Model untuk Tiap Aktivitas")
            warna = ["#22c55e" if i + 1 == label_tebakan else "#e2e8f0" for i in range(6)]
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.bar(nama_aktivitas, proba, color=warna, edgecolor="#94a3b8")
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            for bar, p in zip(bars, proba):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{p:.0%}", ha="center", fontsize=10)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Grafik di atas menunjukkan seberapa yakin model terhadap setiap kemungkinan "
                "aktivitas — batang tertinggi adalah aktivitas yang ditebak model."
            )