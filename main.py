import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from presentation.analyze import show as show_analyze
from presentation.rapid import show as show_rapid
from presentation.results import show as show_results

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ Extração Rápida", "🔬 Análise", "📊 Resultados", "Como usar", "Sobre"])

with tab1:
    show_rapid()

with tab2:
    show_analyze()

with tab3:
    show_results()
