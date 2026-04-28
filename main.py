import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from presentation.analyze import show as show_analyze

load_dotenv()

st.set_page_config(
    page_title="SLD-IE",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    logo = Image.open("logo.png")
    st.image(logo, width=500)

st.title("Structured Literature Decoding and Insight Engine")

tab1, tab2, tab3 = st.tabs(["Análise", "Como usar", "Sobre"])

with tab1:
    show_analyze()
