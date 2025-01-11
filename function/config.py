import streamlit as st
from google.cloud import firestore
import json

firebase_key = json.loads(st.secrets["textkey"])
db = firestore.Client.from_service_account_info(firebase_key)

users_ref = db.collection('login')

def insert_user(username, name, password, auth):
    try:
        # Menambahkan dokumen baru ke koleksi 'login'
        users_ref.add({"key": username, "name": name, "password": password, "role": auth})
        return st.success(f"Berhasil Registrasi {username}")
    except Exception as e:
        return st.warning(f"Terjadi kesalahan: {str(e)}") 

def fetch_all_users():
    # Mengambil semua dokumen dari koleksi 'login'
    res = users_ref.stream()
    return [doc.to_dict() for doc in res]  # Mengembalikan list dari dictionary

def get_user(username):
    # Mengambil dokumen berdasarkan username
    user_ref = users_ref.where("key", "==", username).limit(1).stream()
    user_data = [doc.to_dict() for doc in user_ref]
    return user_data[0] if user_data else None  # Mengembalikan user pertama atau None jika tidak ada

def update_user(username, updates):
    # Mengupdate dokumen berdasarkan username
    user_ref = users_ref.where("key", "==", username).limit(1).stream()
    user_data = [doc for doc in user_ref]
    if user_data:
        user_data[0].reference.update(updates)  # Mengupdate dokumen pertama yang ditemukan
        return st.success(f"Berhasil mengupdate {username}")
    else:
        return st.warning("User  tidak ditemukan")

def delete_user(username):
    # Menghapus dokumen berdasarkan username
    user_ref = users_ref.where("key", "==", username).limit(1).stream()
    user_data = [doc for doc in user_ref]
    if user_data:
        user_data[0].reference.delete()  # Menghapus dokumen pertama yang ditemukan
        return st.success(f"Berhasil menghapus {username}")
    else:
        return st.warning("User  tidak ditemukan")

# def insert_user(username,name,password,auth):
#     try:
#         db.insert({"key": username, "name": name, "password": password, "level": auth})
#         return st.success(f"Berhasil Registrasi {username}")
#     except:
#         return st.warning("username telah dipakai") 

# def fetch_all_users():
#     res = db.fetch()
#     return res.items

# def get_user(username):
#     return db.get(username)

# def update_user(username, updates):
#     return db.update(updates, username)

# def delete_user(username):
#     return db.delete(username)