import streamlit as st
from google.cloud import firestore
import streamlit_authenticator as stauth
# import base64
# import firebase_admin
# from firebase_admin import credentials
import json

#admin
from admin.admin_home import admin_home
from admin.admin_register import admin_register
from admin.admin_upload_gplay import admin_upload_gplay
from admin.admin_upload_yt import admin_upload_yt

#user
from user.user_home import user_home
from user.user_update import user_update
from user.user_upload_gplay import user_upload_gplay
from user.user_upload_yt import user_upload_yt


# Authenticate to Firestore with the JSON account key.
firebase_key = json.loads(st.secrets["textkey"])
db = firestore.Client.from_service_account_info(firebase_key)

# doc_ref = db.collection("login").document("account")

# Fungsi untuk memasukkan user baru ke database Firestore
def insert_user(username, name, password, auth):
    """
    Menambahkan user baru ke dalam koleksi Firestore.
    
    Parameters:
        username (str): Nama pengguna unik.
        name (str): Nama lengkap pengguna.
        password (str): Kata sandi pengguna.
        auth (str): Role atau level pengguna (misalnya admin atau user).
    
    Returns:
        Streamlit response: Pesan sukses jika berhasil atau peringatan jika username telah digunakan.
    """
    try:
        db.put({"Key": username, "Name": name, "Password": password, "Role": auth})
        return st.success("Berhasil Registrasi")
    except:
        return st.warning("username telah dipakai")



# Fungsi untuk mengambil semua data pengguna dari koleksi Firestore
def fetch_all_users():
    """
    Mengambil semua data pengguna dari koleksi 'login' di Firestore.
    
    Returns:
        list: Daftar dictionary yang berisi semua data pengguna.
    """
    users_ref = db.collection('login')
    res = users_ref.stream()
    return [doc.to_dict() for doc in res]

# Fungsi untuk mendapatkan data pengguna berdasarkan username
def get_user(username):
    """
    Mendapatkan data spesifik pengguna berdasarkan username.
    
    Parameters:
        username (str): Nama pengguna yang ingin dicari.
    
    Returns:
        list: Data pengguna yang cocok dengan username.
    """
    response = fetch_all_users()
    return [item for item in response if item["key"] == username]

# Konfigurasi awal Streamlit
st.set_page_config(
    page_title="Sentimen Analisis App",
    page_icon="📦",
    initial_sidebar_state="expanded",
)

# Fungsi utama aplikasi Streamlit
def app():
    """
    Fungsi utama yang menjalankan aplikasi Streamlit.
    - Mengatur autentikasi pengguna.
    - Menampilkan navigasi berdasarkan role pengguna (admin atau user).
    - Menyediakan logout dan akses ke berbagai halaman berdasarkan role.
    """
    # Mengambil semua pengguna dari database
    users = fetch_all_users()

    # Memproses data pengguna
    usernames = [user["key"] for user in users] 
    names = [user["name"] for user in users]
    passwords = [user["password"] for user in users]
    levels = [user["role"] for user in users]

    # Menyiapkan kredensial untuk autentikasi
    credentials = {"usernames": {}}

    for un, name, pw, lvl in zip(usernames, names, passwords, levels):
        user_dict = {"name": name, "password": pw, "levels": lvl}
        credentials["usernames"].update({un: user_dict})

    # Mengatur autentikasi menggunakan streamlit_authenticator
    authenticator = stauth.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=30)
    name_user, authenticator_status, username = authenticator.login("Login", "main")

    # Pesan error jika autentikasi gagal
    if authenticator_status == False:
        st.error("Username/Passwordnya salah")

    # Pesan peringatan jika input kosong
    if authenticator_status == None:
        st.warning("Tolong masukan username dan password anda")

    # Jika autentikasi berhasil
    if authenticator_status == True:
        # Mengambil data pengguna berdasarkan username
        response = get_user(username)
        user_data = response[0]
        nama = user_data["name"]
        level = user_data["role"]

        # Menampilkan nama pengguna di sidebar
        st.sidebar.title(f"Hallo, {nama}")

        if level == "admin":
            menu = ["Home", "Account Management", "Sentimen Upload Google Play Store", "Sentimen Upload Youtube"]
            selected = st.sidebar.selectbox("Navigasi", menu)
            authenticator.logout("logout", "sidebar")   
            if selected == "Home":
                admin_home()
            elif selected == "Account Management":
                admin_register(username)
            elif selected == "Sentimen Upload Google Play Store":
                admin_upload_gplay()
            elif selected == "Sentimen Upload Youtube":
                admin_upload_yt()
            
        elif level == "user":
            menu = ["Sentiment Predictor", "Account Management", "Sentimen Upload Google Play Store", "Sentimen Upload Youtube"]
            selected = st.sidebar.selectbox("Navigasi", menu)
            authenticator.logout("logout", "sidebar")
            if selected == "Sentiment Predictor":
                user_home()
            elif selected == "Account Management":
                user_update(username)
            elif selected == "Sentimen Upload Google Play Store":
                user_upload_gplay()    
            elif selected == "Sentimen Upload Youtube":
                user_upload_yt()

# Menjalankan fungsi utama jika file dijalankan
if __name__ == "__main__":
    app()

# -- Menghilangkan Streamlit Style bawaan --
hide_st_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)