import streamlit as st
from streamlit_option_menu import option_menu
import function.config as users_ref
import streamlit_authenticator as stauth
import time

def user_update(username):
    st.header("⚙️ Account Management",
              help="Halaman untuk mengupdate data user")
    st.markdown("---")
    response = users_ref.get_user(username) 
    user_data = response 
    name = user_data["name"]
    
    with st.form("Update User"):
        username_user = st.text_input("Username :" , username , disabled=True)
        name_user = st.text_input("Masukan Nama :", name)
        password_user = st.text_input("Masukan Password :", type="password")
        hashed_passwords = stauth.Hasher([password_user]).generate()
        hashed_password = hashed_passwords[0]

        button_update = st.form_submit_button("Update")

        if button_update:
            if name_user == "" or password_user == "":
                st.warning("Tolong isikan datanya")
            else :
                if username != username_user:
                    st.warning('Anda tidak bisa mengubah punya orang lain!')
                else:
                    users_ref.update_user(username,updates={"name": name_user, "password" : hashed_password})
                    st.success("Pembaharuan Berhasil")
                    time.sleep(1)
                    st.experimental_rerun()

if __name__ == "__main__":
    user_update()