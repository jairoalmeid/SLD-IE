"""
Central de Estilos e Design System para o SLD (Scientific Literature Decoder).
Sobriedade acadêmica, paleta minimalista e variáveis visuais centralizadas.
"""

import streamlit as st

# Paleta de Cores Institucionais e Científicas (Slate / Navy Theme)
COLOR_PRIMARY = "#0f172a"      # Navy / Slate 900
COLOR_SECONDARY = "#334155"    # Slate 700
COLOR_TEXT = "#1e293b"         # Slate 800
COLOR_MUTED = "#64748b"        # Slate 500
COLOR_BG_CARD = "#ffffff"      # Branco Puro
COLOR_BORDER = "#e2e8f0"       # Slate 200

# Cores de Status Científico
COLOR_SUCCESS = "#16a34a"      # Verde Esmeralda Discreto
COLOR_WARNING = "#d97706"      # Âmbar Discreto
COLOR_ERROR = "#dc2626"        # Vermelho Discreto
COLOR_INFO = "#2563eb"         # Azul Safira Discreto

CSS_GLOBAL_STYLES = """
<style>
    /* Reset & Typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #1e293b;
        background-color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Minimalist Academic Cards */
    .sld-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: border-color 0.15s ease;
    }
    .sld-card:hover {
        border-color: #cbd5e1;
    }
    
    .sld-card-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    
    .sld-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .sld-card-sub {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Badges & Pills */
    .sld-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 4px;
    }
    .sld-badge-success { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .sld-badge-warning { background-color: #fffbeb; color: #92400e; border: 1px solid #fef3c7; }
    .sld-badge-error   { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .sld-badge-info    { background-color: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .sld-badge-neutral { background-color: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }

    /* Button Polish */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    .stButton>button[kind="primary"] {
        background-color: #0f172a;
        color: #ffffff;
        border: 1px solid #0f172a;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    /* Stepper Header */
    .sld-stepper {
        display: flex;
        justify-content: space-between;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 20px;
        font-size: 0.85rem;
    }
    .sld-stepper-item {
        color: #64748b;
        font-weight: 500;
    }
    .sld-stepper-item.active {
        color: #2563eb;
        font-weight: 700;
    }
    .sld-stepper-item.completed {
        color: #16a34a;
        font-weight: 600;
    }
</style>
"""


def inject_custom_styles():
    """Injeta os estilos CSS globais na sessão Streamlit."""
    st.markdown(CSS_GLOBAL_STYLES, unsafe_allow_html=True)
