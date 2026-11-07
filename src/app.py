"""Painel do projeto: resultados (#19) e simulador de commit (#20).

Uso: streamlit run src/app.py
"""
import sys
from pathlib import Path

# o streamlit roda o arquivo direto, entao a raiz do repo precisa estar no path pro "import src"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Commit Limpo", page_icon="📊", layout="wide")

pages = [st.Page(ROOT / "src" / "resultados.py", title="Resultados", icon="📊", default=True),
         st.Page(ROOT / "src" / "simulador.py", title="Simulador de commit", icon="🔎")]
st.navigation(pages).run()
