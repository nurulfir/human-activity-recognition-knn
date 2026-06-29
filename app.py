# Aplikasi Deteksi Aktivitas Manusia
# Menggunakan Algoritma K-Nearest Neighbor (K-NN)

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Deteksi Aktivitas - K-NN", page_icon="🏃", layout="wide")


# HELPER: load & parse CSV yang diupload
@st.cache_data
def load_from_csv(file_bytes: bytes):
    """
    Baca CSV, pisahkan fitur dari kolom target ('Activity').
    Kolom 'Subject' dibuang (bukan fitur prediksi).
    Label string di-encode ke integer.
    """
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))

    # Deteksi kolom target
    target_col = "Activity"
    drop_cols = ["Subject", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y_raw = df[target_col].values
    y = le.fit_transform(y_raw) + 1
    class_names = list(le.classes_)

    # Simpan kolom Subject untuk split berbasis orang (lebih realistis)
    subjects = df["Subject"].values if "Subject" in df.columns else None

    return X, y, class_names, df, feature_cols, subjects


@st.cache_data
def split_and_scale(X_bytes_hash):
    """
    Split berdasarkan Subject (orang) jika tersedia — sama seperti split
    resmi dataset UCI HAR. Ini lebih realistis: model diuji pada orang
    yang belum pernah dilihat saat training.

    Jika kolom Subject tidak ada, fallback ke random split 80/20.
    """
    X        = st.session_state["X_raw"]
    y        = st.session_state["y_raw"]
    subjects = st.session_state.get("subjects_raw")

    if subjects is not None:
        unique_subj = set(np.unique(subjects).tolist())

        # Split resmi UCI HAR — dipilih acak oleh peneliti aslinya,
        # bukan sekadar 70% urutan pertama. Inilah yang menghasilkan K=10 optimal.
        UCI_TRAIN = {1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30}
        UCI_TEST  = {2,4,9,10,12,13,18,20,24}

        # Jika subject CSV cocok dengan UCI → pakai split resmi
        # Jika dataset lain → fallback ke 70% pertama secara urutan
        if unique_subj <= (UCI_TRAIN | UCI_TEST):
            train_subj = UCI_TRAIN
            test_subj  = UCI_TEST
        else:
            all_subj   = sorted(unique_subj)
            n_train    = max(1, int(len(all_subj) * 0.7))
            train_subj = set(all_subj[:n_train])
            test_subj  = set(all_subj[n_train:])

        train_mask = np.isin(subjects, list(train_subj))
        test_mask  = np.isin(subjects, list(test_subj))
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, X_train, X_test


# Sidebar
st.sidebar.title("🏃 Deteksi Aktivitas Manusia")
st.sidebar.caption("Algoritma K-Nearest Neighbor (K-NN)")
st.sidebar.markdown(
    "[Sumber Dataset (UCI)](https://archive.ics.uci.edu/dataset/240/"
    "human+activity+recognition+using+smartphones)"
)
st.sidebar.divider()

# ── Upload CSV ──
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV dataset HAR",
    type=["csv"],
    help="Pastikan CSV memiliki kolom 'Activity' sebagai label kelas."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    new_hash   = hash(file_bytes)
    # Hanya proses ulang jika file benar-benar baru (hash berbeda)
    if new_hash != st.session_state.get("file_hash"):
        try:
            X_raw, y_raw, class_names, df_raw, feature_cols, subjects = load_from_csv(file_bytes)
            st.session_state["X_raw"]        = X_raw
            st.session_state["y_raw"]        = y_raw
            st.session_state["class_names"]  = class_names
            st.session_state["df_raw"]       = df_raw
            st.session_state["feature_cols"] = feature_cols
            st.session_state["subjects_raw"] = subjects
            st.session_state["data_loaded"]  = True
            st.session_state["file_hash"]    = new_hash
            # Reset model hanya saat file baru diupload
            for k in ["model", "y_pred", "k_dipakai", "akurasi_terakhir",
                      "k_terbaik", "grafik_k", "input_k",
                      "X_train_sc", "X_test_sc", "y_train", "y_test"]:
                st.session_state.pop(k, None)
        except Exception as e:
            st.sidebar.error(f"❌ Gagal membaca CSV: {e}")
            st.session_state["data_loaded"] = False

    if st.session_state.get("data_loaded"):
        X_raw = st.session_state["X_raw"]
        st.sidebar.success(
            f"✅ Data berhasil dimuat!\n\n"
            f"- **{X_raw.shape[0]:,}** baris\n"
            f"- **{X_raw.shape[1]}** fitur\n"
            f"- **{len(st.session_state['class_names'])}** kelas"
        )
else:
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

# ── Navigasi ──
st.sidebar.divider()
halaman = st.sidebar.radio("Pilih Tahapan", [
    "🏠 Beranda",
    "1️⃣ Mengenal Data",
    "2️⃣ Menyiapkan Data",
    "3️⃣ Melatih Model K-NN",
    "4️⃣ Menguji Hasil Model",
    "5️⃣ Coba Prediksi Sendiri",
])

# ── Lazy split (hanya dilakukan sekali setelah CSV diupload) ──
def ensure_split():
    """Pastikan data sudah di-split & di-scale, simpan ke session_state."""
    if "X_train_sc" not in st.session_state and st.session_state.get("data_loaded"):
        fh = st.session_state["file_hash"]
        res = split_and_scale(fh)
        (st.session_state["X_train_sc"],
         st.session_state["X_test_sc"],
         st.session_state["y_train"],
         st.session_state["y_test"],
         st.session_state["X_train_raw"],
         st.session_state["X_test_raw"]) = res

def need_data_warning():
    st.warning(
        "⚠️ Belum ada data. Silakan **upload file CSV** terlebih dahulu "
        "melalui panel kiri (sidebar)."
    )


# ═════════════════════════════════════════════
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
        if st.session_state.get("data_loaded"):
            X_raw = st.session_state["X_raw"]
            ensure_split()
            n_train = len(st.session_state["y_train"])
            n_test  = len(st.session_state["y_test"])
            n_class = len(st.session_state["class_names"])
            st.info(
                "**Ringkasan Data**\n\n"
                f"- Data latih: **{n_train:,}** baris\n"
                f"- Data uji: **{n_test:,}** baris\n"
                f"- Jumlah fitur: **{X_raw.shape[1]}**\n"
                f"- Jumlah kelas: **{n_class} aktivitas**"
            )
        else:
            st.info("📂 Upload dataset CSV di sidebar kiri untuk memulai.")

    st.divider()
    st.markdown("**Urutan menjelajah aplikasi ini (lihat menu di kiri):**")
    st.markdown("""
    1. **Mengenal Data** — lihat seperti apa data mentahnya
    2. **Menyiapkan Data** — menyamakan skala angka sebelum diproses
    3. **Melatih Model K-NN** — komputer "belajar" dari data
    4. **Menguji Hasil Model** — seberapa akurat tebakan komputer
    5. **Coba Prediksi Sendiri** — masukkan angka sendiri, lihat hasil tebakannya
    """)


# Halaman 1 - Mengenal Data
elif halaman == "1️⃣ Mengenal Data":
    st.title("1️⃣ Mengenal Data")

    if not st.session_state.get("data_loaded"):
        need_data_warning()
        st.stop()

    ensure_split()
    df_raw      = st.session_state["df_raw"]
    class_names = st.session_state["class_names"]
    y_train     = st.session_state["y_train"]
    y_test      = st.session_state["y_test"]
    X_raw       = st.session_state["X_raw"]

    st.markdown("Sebelum diolah, kita lihat dulu bentuk datanya seperti apa.")

    with st.expander("📌 Penjelasan Singkat", expanded=True):
        st.markdown("""
        - Setiap baris = **satu rekaman gerakan** dari satu orang.
        - Setiap kolom = **satu angka hasil olahan sensor** (561 kolom total).
        - Tidak ada data yang kosong/hilang — datanya sudah bersih dari sumbernya.
        - Data dibagi otomatis: **80% untuk belajar** (training),
          **20% untuk diuji** (testing) — supaya penilaiannya adil.
        """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Latih (Training)", f"{len(y_train):,} baris")
    col2.metric("Data Uji (Testing)",    f"{len(y_test):,} baris")
    col3.metric("Jumlah Fitur",          X_raw.shape[1])

    st.subheader("Contoh Data Mentah (5 baris pertama)")
    st.caption("Angka-angka ini hasil olahan getaran sensor — sulit dibaca manusia, tapi mudah dibaca komputer.")
    feature_cols = st.session_state["feature_cols"]
    st.dataframe(df_raw[feature_cols].head(5).iloc[:, :8], use_container_width=True)
    st.caption("*(hanya 8 dari {} kolom yang ditampilkan)*".format(len(feature_cols)))

    st.subheader("6 Jenis Aktivitas yang Akan Ditebak")
    tabel_aktivitas = pd.DataFrame({
        "Kode": list(range(1, len(class_names) + 1)),
        "Nama Aktivitas": class_names
    })
    st.dataframe(tabel_aktivitas, use_container_width=True, hide_index=True)

    st.subheader("Jumlah Data per Aktivitas")
    label_series = pd.Series(y_train)
    counts = label_series.value_counts().sort_index()
    tabel_jumlah = pd.DataFrame({
        "Aktivitas": [class_names[i - 1] for i in counts.index],
        "Jumlah Data Latih": counts.values
    })
    st.dataframe(tabel_jumlah, use_container_width=True, hide_index=True)
    st.caption("Jumlah tiap aktivitas cukup seimbang, jadi model tidak akan bias ke satu aktivitas saja.")


# Halaman 2 - Menyiapkan Data
elif halaman == "2️⃣ Menyiapkan Data":
    st.title("2️⃣ Menyiapkan Data (Preprocessing)")

    if not st.session_state.get("data_loaded"):
        need_data_warning()
        st.stop()

    ensure_split()
    X_train_raw = st.session_state["X_train_raw"]
    X_train_sc  = st.session_state["X_train_sc"]
    X_test_sc   = st.session_state["X_test_sc"]

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
    df_raw = st.session_state["df_raw"]
    null_total = df_raw.isnull().sum().sum()
    if null_total == 0:
        st.success(f"✅ Tidak ada nilai kosong sama sekali ({null_total} missing values) — data sudah bersih.")
    else:
        st.warning(f"⚠️ Ditemukan {null_total} nilai kosong. Pertimbangkan imputasi sebelum melatih model.")

    st.subheader("Langkah 2 — Standardisasi Skala")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sebelum** distandardisasi (4 fitur pertama):")
        df_before = pd.DataFrame(X_train_raw[:, :4])
        st.dataframe(
            df_before.describe().loc[["mean", "std", "min", "max"]],
            use_container_width=True
        )
    with col2:
        st.write("**Sesudah** distandardisasi (4 fitur pertama):")
        df_after = pd.DataFrame(X_train_sc[:, :4])
        st.dataframe(
            df_after.describe().loc[["mean", "std", "min", "max"]],
            use_container_width=True
        )
    st.caption("Setelah distandardisasi, rata-rata setiap kolom mendekati 0 — semua fitur jadi 'setara'.")


# Halaman 3 - Melatih Model K-NN
elif halaman == "3️⃣ Melatih Model K-NN":
    st.title("3️⃣ Melatih Model K-NN")

    if not st.session_state.get("data_loaded"):
        need_data_warning()
        st.stop()

    ensure_split()
    X_train_sc = st.session_state["X_train_sc"]
    X_test_sc  = st.session_state["X_test_sc"]
    y_train    = st.session_state["y_train"]
    y_test     = st.session_state["y_test"]

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
            daftar_k       = range(1, 16)
            daftar_akurasi = []
            progres        = st.progress(0)
            for i, k_coba in enumerate(daftar_k):
                model_coba = KNeighborsClassifier(n_neighbors=k_coba, n_jobs=1)
                model_coba.fit(X_train_sc, y_train)
                pred_coba = model_coba.predict(X_test_sc)
                daftar_akurasi.append(accuracy_score(y_test, pred_coba))
                progres.progress((i + 1) / len(daftar_k))

            k_terbaik = list(daftar_k)[int(np.argmax(daftar_akurasi))]
            st.session_state["k_terbaik"] = k_terbaik
            st.session_state["grafik_k"]  = (list(daftar_k), daftar_akurasi, k_terbaik)

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

    # Pakai number_input sebagai pengganti slider — nilainya tidak di-reset Streamlit saat rerun.
    # Saat k_terbaik baru ditemukan (Langkah A), tombol "Pakai K Terbaik" memperbarui nilainya.
    if "input_k" not in st.session_state:
        st.session_state["input_k"] = st.session_state.get("k_terbaik", 10)

    if "k_terbaik" in st.session_state:
        col_info, col_btn = st.columns([3, 1])
        col_info.info(f"K terbaik hasil pencarian: **K = {st.session_state['k_terbaik']}**")
        if col_btn.button("⬅️ Pakai K Terbaik"):
            st.session_state["input_k"] = st.session_state["k_terbaik"]
            st.rerun()

    k = st.slider("Pilih jumlah tetangga (K)", min_value=1, max_value=20,
                  value=st.session_state["input_k"], key="slider_k_widget")
    st.session_state["input_k"] = k   # simpan setiap perubahan manual

    if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
        with st.spinner(f"Melatih model K-NN dengan K = {k} ..."):
            model  = KNeighborsClassifier(n_neighbors=k, n_jobs=1)
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            akurasi = accuracy_score(y_test, y_pred)

        # Di luar spinner supaya pesan tidak hilang saat rerun
        st.session_state["model"]     = model
        st.session_state["y_pred"]    = y_pred
        st.session_state["k_dipakai"] = k
        st.session_state["akurasi_terakhir"] = akurasi

    if "model" in st.session_state:
        st.success(f"🎉 Model selesai dilatih! Akurasi pada data uji: **{st.session_state['akurasi_terakhir']:.2%}**")
        st.info(
            f"Model aktif saat ini menggunakan **K = {st.session_state['k_dipakai']}**. "
            "Lanjut ke menu **4️⃣ Menguji Hasil Model** untuk melihat detail performanya."
        )


# Halaman 4 - Menguji Hasil Model
elif halaman == "4️⃣ Menguji Hasil Model":
    st.title("4️⃣ Menguji Hasil Model")

    if not st.session_state.get("data_loaded"):
        need_data_warning()
        st.stop()

    ensure_split()
    X_train_sc  = st.session_state["X_train_sc"]
    X_test_sc   = st.session_state["X_test_sc"]
    y_train     = st.session_state["y_train"]
    y_test      = st.session_state["y_test"]
    class_names = st.session_state["class_names"]
    nama_aktivitas = [n.replace("_", " ").title() for n in class_names]

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
            model  = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            model.fit(X_train_sc, y_train)
            st.session_state["model"]     = model
            st.session_state["y_pred"]    = model.predict(X_test_sc)
            st.session_state["k_dipakai"] = 5
            st.rerun()
    else:
        y_pred = st.session_state["y_pred"]
        st.info(f"Menampilkan hasil model dengan **K = {st.session_state['k_dipakai']}**")

        st.subheader("Ringkasan Performa")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi",   f"{accuracy_score(y_test, y_pred):.2%}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
        col3.metric("Recall",    f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
        col4.metric("F1-Score",  f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

        st.subheader("Confusion Matrix (Tabel Tebakan vs Kenyataan)")
        # Urutkan label sesuai class_names
        unique_labels = sorted(np.unique(y_test))
        tick_labels   = [class_names[i - 1].replace("_", " ").title() for i in unique_labels]
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=tick_labels, yticklabels=tick_labels,
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
                y_test, y_pred,
                labels=unique_labels,
                target_names=tick_labels,
                output_dict=True,
                digits=3
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


# Halaman 5 - Coba Prediksi Sendiri 
elif halaman == "5️⃣ Coba Prediksi Sendiri":
    st.title("5️⃣ Coba Prediksi Sendiri")

    if not st.session_state.get("data_loaded"):
        need_data_warning()
        st.stop()

    ensure_split()
    X_train_sc  = st.session_state["X_train_sc"]
    X_test_sc   = st.session_state["X_test_sc"]
    y_train     = st.session_state["y_train"]
    y_test      = st.session_state["y_test"]
    class_names = st.session_state["class_names"]
    nama_aktivitas = [n.replace("_", " ").title() for n in class_names]

    st.markdown("""
    Pilih salah satu data uji di bawah, lalu lihat tebakan model dibandingkan
    dengan jawaban yang sebenarnya. Ini menunjukkan model **bekerja secara nyata**,
    bukan hanya teori.
    """)

    if "model" not in st.session_state:
        st.warning("⚠️ Model belum dilatih. Silakan ke menu **3️⃣ Melatih Model K-NN** terlebih dahulu, "
                   "atau klik tombol di bawah untuk memakai pengaturan default.")
        if st.button("Latih Model Default (K=5)"):
            model  = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            model.fit(X_train_sc, y_train)
            st.session_state["model"]     = model
            st.session_state["y_pred"]    = model.predict(X_test_sc)
            st.session_state["k_dipakai"] = 5
            st.rerun()
    else:
        model = st.session_state["model"]

        nomor_data = st.number_input(
            f"Pilih nomor data uji (0 sampai {len(X_test_sc) - 1})",
            min_value=0, max_value=len(X_test_sc) - 1, value=0, step=1
        )

        if st.button("🔮 Tebak Aktivitas", type="primary", use_container_width=True):
            data_pilihan   = X_test_sc[nomor_data].reshape(1, -1)
            label_asli     = int(y_test[nomor_data])
            label_tebakan  = int(model.predict(data_pilihan)[0])
            proba          = model.predict_proba(data_pilihan)[0]

            # Ambil nama aktivitas dari class_names (1-indexed)
            aktivitas_asli    = class_names[label_asli - 1]
            aktivitas_tebakan = class_names[label_tebakan - 1]

            col1, col2 = st.columns(2)
            col1.metric("Aktivitas Sebenarnya", aktivitas_asli)
            col2.metric("Tebakan Model",        aktivitas_tebakan)

            if label_asli == label_tebakan:
                st.success("✅ Model menebak dengan BENAR!")
            else:
                st.error("❌ Model menebak SALAH — wajar, tidak ada model yang 100% sempurna.")

            st.subheader("Tingkat Keyakinan Model untuk Tiap Aktivitas")
            # Sesuaikan jumlah warna dengan jumlah kelas di model
            model_classes = list(model.classes_)
            warna = [
                "#22c55e" if cls == label_tebakan else "#e2e8f0"
                for cls in model_classes
            ]
            tick_labels = [class_names[c - 1].replace("_", " ").title() for c in model_classes]

            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.bar(tick_labels, proba, color=warna, edgecolor="#94a3b8")
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            for bar, p in zip(bars, proba):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{p:.0%}", ha="center", fontsize=10
                )
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Grafik di atas menunjukkan seberapa yakin model terhadap setiap kemungkinan "
                "aktivitas — batang tertinggi adalah aktivitas yang ditebak model."
            )