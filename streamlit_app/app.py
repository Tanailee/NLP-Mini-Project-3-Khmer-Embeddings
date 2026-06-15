from pathlib import Path
import html
import json
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


st.set_page_config(
    page_title="Khmer Temple NLP Explorer",
    page_icon="KT",
    layout="wide",
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "outputs" / "tables"
EMBEDDING_DIR = ROOT / "outputs" / "embeddings"
FIGURE_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"

EMBEDDING_DIM = 50
WINDOW_SIZE = 4
NEGATIVE_SAMPLES = 2
MIN_FREQ = 10
N_CONTEXT = 5
HIDDEN_SIZE = 512

KHMER_PUNCTUATION = [
    "។", "៕", "៖", "ៗ", "៘", "៙", "៚", "៛",
    ",", ".", "!", "?", ":", ";",
    "(", ")", "[", "]", "{", "}",
    "\"", "'", "“", "”", "‘", "’",
    "-", "–", "—", "_", "/", "\\", "|",
]

KHMER_WORDS = [
    "ប្រាសាទអង្គរវត្ត", "អង្គរវត្ត", "ប្រាសាទ", "អង្គរ", "វត្ត", "សៀមរាប",
    "ខេត្ត", "ក្រុង", "ព្រះបាទ", "ព្រះ", "សូរ្យវរ្ម័ន", "វិស្ណុ", "សាសនា",
    "ព្រហ្មញ្ញសាសនា", "ព្រះពុទ្ធសាសនា", "ខ្មែរ", "កម្ពុជា", "ថ្ម", "ចម្លាក់",
    "អប្សរា", "ស្ថាបត្យកម្ម", "ភ្នំ", "សុមេរុ", "រាជធានី", "អាណាចក្រ",
    "សតវត្ស", "ប្រវត្តិសាស្ត្រ", "បន្ទាយស្រី", "កោះកេរ", "ទេវតា", "នគរ",
    "រាជា", "ទន្លេ", "កំពូល", "ជញ្ជាំង", "កណ្តាល", "ទិស", "លោក", "ដី",
    "មហា", "បុរាណ", "សំណង់", "សិល្បៈ", "ទេសភាព", "កសាង", "ប្រើប្រាស់",
]
KHMER_WORDS = sorted(set(KHMER_WORDS), key=len, reverse=True)
KHMER_RE = re.compile(r"[\u1780-\u17FF]+")
SPLIT_RE = re.compile(r"[\s,;:()\[\]{}\"'«»“”‘’។៕?!…\-/]+")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(201, 162, 39, 0.10), transparent 30%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 38%);
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 2.4rem;
        max-width: 1400px;
        box-sizing: border-box;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(201, 162, 39, 0.10), transparent 30%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 38%);
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.25rem;
        border-radius: 18px;
        background:
            linear-gradient(135deg, rgba(11, 31, 58, 0.98), rgba(14, 51, 91, 0.94) 56%, rgba(201, 162, 39, 0.18)),
            linear-gradient(90deg, #FFF8ED, #EEF6FF);
        border: 1px solid rgba(201, 162, 39, 0.32);
        margin-bottom: 1rem;
        box-shadow: 0 16px 38px rgba(11, 31, 58, 0.15);
    }
    .hero:before {
        content: "";
        position: absolute;
        right: -70px;
        top: -80px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        border: 42px solid rgba(201, 162, 39, 0.16);
    }
    .hero:after {
        content: "";
        position: absolute;
        right: 42px;
        bottom: 24px;
        width: 210px;
        height: 88px;
        background:
            linear-gradient(90deg, transparent 0 8%, rgba(255,255,255,0.14) 8% 13%, transparent 13% 21%, rgba(255,255,255,0.14) 21% 26%, transparent 26% 34%, rgba(255,255,255,0.14) 34% 39%, transparent 39% 47%, rgba(255,255,255,0.14) 47% 52%, transparent 52% 60%, rgba(255,255,255,0.14) 60% 65%, transparent 65%),
            linear-gradient(180deg, transparent 0 28%, rgba(255,255,255,0.10) 28% 36%, transparent 36% 64%, rgba(255,255,255,0.10) 64% 72%, transparent 72%);
        opacity: 0.8;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        font-size: 2.55rem;
        line-height: 1.05;
        margin-bottom: 0.45rem;
        color: #ffffff;
        font-weight: 800;
        letter-spacing: 0;
    }
    .hero p {
        position: relative;
        z-index: 1;
        font-size: 1rem;
        color: #dbeafe;
        max-width: 850px;
        margin: 0.2rem 0;
    }
    .hero .hero-kicker {
        color: #f4d675;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.82rem;
        margin-bottom: 0.55rem;
    }
    .hero-accent-line {
        position: relative;
        z-index: 1;
        width: 96px;
        height: 4px;
        border-radius: 999px;
        background: #C9A227;
        margin: 0.85rem 0 0 0;
    }
    .soft-card {
        padding: 1.05rem;
        border-radius: 14px;
        border: 1px solid #e6edf5;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
    }
    .note {
        padding: 1rem 1.1rem;
        border-left: 5px solid #C9A227;
        background: linear-gradient(90deg, #FFF8ED 0%, #FFFFFF 100%);
        border-radius: 12px;
        color: #526070;
        margin: 0.9rem 0 1.25rem 0;
        box-shadow: 0 8px 20px rgba(11, 31, 58, 0.06);
    }
    .note strong {
        color: #0B1F3A;
    }
    .small-muted {
        color: #526070;
        font-size: 0.93rem;
    }
    .section-title {
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
        color: #0B1F3A;
        font-size: 1.28rem;
        font-weight: 760;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6edf5;
        border-top: 4px solid #C9A227;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
    }
    div[data-testid="stMetricLabel"] p {
        color: #526070;
        font-size: 0.84rem;
        font-weight: 650;
    }
    div[data-testid="stMetricValue"] {
        color: #0B1F3A;
        font-size: 1.55rem;
        font-weight: 800;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #e6edf5;
        border-top: 4px solid #C9A227;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        min-height: 106px;
        box-shadow: 0 12px 28px rgba(11, 31, 58, 0.08);
    }
    .metric-label {
        color: #526070;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #0B1F3A;
        font-size: 1.72rem;
        font-weight: 820;
        line-height: 1.1;
    }
    .feature-card, .pipeline-card, .step-card {
        background: #FFFFFF;
        border: 1px solid #e6edf5;
        border-radius: 16px;
        padding: 1.05rem;
        min-height: 128px;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
    }
    .feature-card {
        min-height: 142px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border-top: 4px solid #C9A227;
    }
    .pipeline-card {
        border-top: 3px solid #C9A227;
        min-height: 92px;
        padding: 0.82rem;
    }
    .step-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
    }
    .icon-badge {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #EEF6FF;
        border: 1px solid #d9e8f8;
        color: #0B1F3A;
        margin-bottom: 0.65rem;
    }
    .pipeline-card .icon-badge {
        width: 28px;
        height: 28px;
        margin-bottom: 0.42rem;
    }
    .feature-title, .pipeline-title, .step-title {
        color: #0B1F3A;
        font-weight: 760;
        font-size: 0.98rem;
        margin-bottom: 0.35rem;
    }
    .feature-body, .pipeline-body, .step-body {
        color: #526070;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .step-number {
        color: #C9A227;
        font-weight: 820;
        font-size: 0.88rem;
        letter-spacing: 0.06em;
    }
    [data-testid="stSidebar"] {
        background: transparent;
        flex-shrink: 0;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0B1F3A 0%, #102b4d 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e5eef8;
    }
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-weight: 800;
        line-height: 1.12;
    }
    [data-testid="stSidebar"] p {
        color: #b7c7d9;
    }
    [data-testid="stSidebar"] [role="radiogroup"] input {
        display: none;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 12px;
        padding: 0.52rem 0.5rem;
        margin: 0.16rem 0;
        transition: background 120ms ease, border 120ms ease;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(201, 162, 39, 0.20);
        border: 1px solid rgba(201, 162, 39, 0.40);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:before {
        content: "";
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 8px;
        border: 1px solid rgba(201, 162, 39, 0.52);
        display: inline-block;
        margin-right: 0.35rem;
        background-color: rgba(255, 255, 255, 0.10);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 17px 17px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(1):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 11.5L12 5l8 6.5V20a1 1 0 0 1-1 1h-5v-6h-4v6H5a1 1 0 0 1-1-1v-8.5z' stroke='%23F4D675' stroke-width='1.8' stroke-linejoin='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(2):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M7 3h7l4 4v14H7V3z' stroke='%23F4D675' stroke-width='1.8'/%3E%3Cpath d='M14 3v5h5M9 13h6M9 17h6' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(3):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 12l4 4L19 6' stroke='%23F4D675' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M5 20h14' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(4):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M7 3h7l4 4v14H7V3z' stroke='%23F4D675' stroke-width='1.8'/%3E%3Cpath d='M14 3v5h5M9 13h6M9 17h6' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(5):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 20V4M4 20h16M8 17v-5M12 17V8M16 17v-7' stroke='%23F4D675' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(6):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='6' cy='12' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Ccircle cx='18' cy='6' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Ccircle cx='18' cy='18' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Cpath d='M8.2 10.8l7.6-3.6M8.2 13.2l7.6 3.6' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(7):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 18l4-11 5 8 5-10' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='9' cy='7' r='1.8' fill='%23F4D675'/%3E%3Ccircle cx='14' cy='15' r='1.8' fill='%23F4D675'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(8):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='7' cy='7' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Ccircle cx='17' cy='7' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Ccircle cx='12' cy='17' r='2.5' stroke='%23F4D675' stroke-width='1.8'/%3E%3Cpath d='M9.2 8.4l5.6 0M8.3 9.1l2.5 5.4M15.7 9.1l-2.5 5.4' stroke='%23F4D675' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(9):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 12h12M13 7l5 5-5 5' stroke='%23F4D675' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M5 6h4M5 18h4' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(10):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M6 18V8M12 18V5M18 18v-7M4 20h16' stroke='%23F4D675' stroke-width='1.9' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(11):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M6 4h12v16H6V4z' stroke='%23F4D675' stroke-width='1.8'/%3E%3Cpath d='M9 8h6M9 12h6M9 16h4' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(12):before {
        background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M8 17H7a4 4 0 0 1 0-8 5 5 0 0 1 9.6-1.7A3.8 3.8 0 0 1 17 17h-1' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M12 19V11M9 14l3-3 3 3' stroke='%23F4D675' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked):before {
        background-color: rgba(201, 162, 39, 0.18);
        border-color: rgba(244, 214, 117, 0.95);
    }
    .sidebar-brand {
        border: 1px solid rgba(201, 162, 39, 0.28);
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-brand-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.18;
    }
    .sidebar-brand-caption {
        color: #b7c7d9;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }
    .model-card {
        background: #FFFFFF;
        border: 1px solid #E6EAF0;
        border-left: 5px solid #C9A227;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
        margin: 0.8rem 0 1rem 0;
    }
    .model-card-title {
        color: #0B1F3A;
        font-weight: 780;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }
    .model-card-body {
        color: #526070;
        font-size: 0.94rem;
        line-height: 1.5;
    }
    .result-card {
        background: linear-gradient(90deg, #E8F7EE 0%, #FFFFFF 100%);
        border: 1px solid #D9EFE2;
        border-left: 5px solid #14804A;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
        margin: 0.8rem 0 1rem 0;
    }
    .result-card.warning {
        background: linear-gradient(90deg, #FFF8ED 0%, #FFFFFF 100%);
        border: 1px solid #F0DEC1;
        border-left: 5px solid #C9A227;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }
    .result-label {
        color: #526070;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.045em;
    }
    .result-value {
        color: #0B1F3A;
        font-size: 1rem;
        font-weight: 780;
        margin-top: 0.2rem;
        overflow-wrap: anywhere;
    }
    .similarity-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border: 1px solid #E6EAF0;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(11, 31, 58, 0.07);
        margin-top: 0.55rem;
    }
    .table-shell {
        border-radius: 14px;
        overflow: hidden;
    }
    .similarity-table th {
        background: #0B1F3A;
        color: #FFFFFF;
        text-align: left;
        padding: 0.72rem 0.85rem;
        font-size: 0.84rem;
        font-weight: 760;
    }
    .similarity-table td {
        padding: 0.7rem 0.85rem;
        border-bottom: 1px solid #E6EAF0;
        color: #243447;
        font-size: 0.94rem;
    }
    .similarity-table tr:last-child td {
        border-bottom: none;
    }
    .similarity-table tr:nth-child(even) td {
        background: #F8FBFF;
    }
    .similarity-score {
        font-family: "Inter", "Segoe UI", sans-serif;
        font-weight: 700;
        color: #0B1F3A;
    }
    .interpretation-card {
        background: #EEF6FF;
        border: 1px solid #D8E8F8;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #526070;
        margin-top: 0.8rem;
    }
    .interpretation-card strong {
        color: #0B1F3A;
    }
    @media (max-width: 900px) {
        .result-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_default_text():
    paths = [DATA_DIR / "temples.txt", ROOT / "temples.txt"]
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))
    return "", ""


def read_uploaded_txt(uploaded_file):
    if uploaded_file is None:
        return "", ""
    raw_bytes = uploaded_file.getvalue()
    for encoding in ["utf-8", "utf-8-sig"]:
        try:
            return raw_bytes.decode(encoding), uploaded_file.name
        except UnicodeDecodeError:
            pass
    return "", uploaded_file.name


def clean_text(text):
    cleaned = text
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]{0,20}\]", " ", cleaned)
    cleaned = cleaned.replace("\u200b", "")
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_number_only(token):
    token = str(token).strip()
    if token == "":
        return False

    khmer_digits = "០១២៣៤៥៦៧៨៩"
    english_digits = "0123456789"
    for char in token:
        if char not in khmer_digits and char not in english_digits:
            return False
    return True


def has_khmer_character(token):
    for char in str(token):
        if "\u1780" <= char <= "\u17FF":
            return True
    return False


def is_punctuation_token(token):
    token = str(token).strip()
    if token == "":
        return False
    if token in KHMER_PUNCTUATION:
        return True
    for char in token:
        if char not in KHMER_PUNCTUATION:
            return False
    return True


def is_useful_token(token):
    if token is None:
        return False
    token = str(token).strip()
    if token == "":
        return False
    if is_punctuation_token(token):
        return False
    if is_number_only(token):
        return False
    if not has_khmer_character(token):
        return False
    if len(token) > 40:
        return False
    return True


def split_unknown_khmer_chunk(chunk, max_len=6):
    pieces = []
    start = 0
    while start < len(chunk):
        end = start + max_len
        pieces.append(chunk[start:end])
        start = end
    return pieces


def dictionary_max_match(segment):
    tokens = []
    index = 0
    while index < len(segment):
        matched_word = None
        for word in KHMER_WORDS:
            if segment.startswith(word, index):
                matched_word = word
                break
        if matched_word is not None:
            tokens.append(matched_word)
            index = index + len(matched_word)
        else:
            match = KHMER_RE.match(segment, index)
            if match:
                unknown_chunk = match.group(0)
                pieces = split_unknown_khmer_chunk(unknown_chunk)
                for piece in pieces:
                    tokens.append(piece)
                index = match.end()
            else:
                index = index + 1
    return tokens


def fallback_tokenize(text):
    tokens = []
    rough_segments = SPLIT_RE.split(text)
    for segment in rough_segments:
        if segment == "":
            continue
        khmer_parts = KHMER_RE.findall(segment)
        for part in khmer_parts:
            part_tokens = dictionary_max_match(part)
            for token in part_tokens:
                tokens.append(token)
    return tokens


def tokenize_khmer_text(text):
    word_tokenize = None
    tokenizer_name = "dictionary fallback"
    try:
        from khmer_nltk import word_tokenize as imported_word_tokenize
        word_tokenize = imported_word_tokenize
        tokenizer_name = "khmer_nltk"
    except Exception:
        try:
            from khmernltk import word_tokenize as imported_word_tokenize
            word_tokenize = imported_word_tokenize
            tokenizer_name = "khmernltk"
        except Exception:
            word_tokenize = None

    if word_tokenize is not None:
        try:
            raw_tokens = word_tokenize(text)
            if isinstance(raw_tokens, str):
                raw_tokens = raw_tokens.split()
            tokens = []
            for token in raw_tokens:
                clean_token = str(token).strip()
                if clean_token != "":
                    tokens.append(clean_token)
            if len(tokens) > 0:
                return tokens, tokenizer_name
        except Exception:
            pass

    return fallback_tokenize(text), "dictionary fallback"


def preprocess_text(text):
    cleaned = clean_text(text)
    raw_tokens, tokenizer_name = tokenize_khmer_text(cleaned)
    useful_tokens = []
    for token in raw_tokens:
        if is_useful_token(token):
            useful_tokens.append(str(token).strip())
    return cleaned, raw_tokens, useful_tokens, tokenizer_name


def build_frequency_dataframe(tokens):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    rows = []
    for word, frequency in counts.items():
        rows.append({"word": word, "frequency": frequency, "length": len(word)})

    if len(rows) == 0:
        return pd.DataFrame(columns=["word", "frequency", "length", "rank", "frequency_group"])

    df = pd.DataFrame(rows)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["frequency_group"] = pd.cut(
        df["frequency"],
        bins=[0, 2, 5, 10, 20, 50, 100000],
        labels=["1-2", "3-5", "6-10", "11-20", "21-50", "51+"],
    )
    return df


@st.cache_data
def load_csv_file(path_text):
    path = Path(path_text)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_json_file(path_text):
    path = Path(path_text)
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_data
def load_numpy_file(path_text):
    path = Path(path_text)
    if path.exists():
        return np.load(path)
    return None


def get_current_text():
    if "selected_text" not in st.session_state:
        default_text, default_name = load_default_text()
        st.session_state["selected_text"] = default_text
        st.session_state["selected_name"] = default_name
        st.session_state["selected_source"] = "Use default temples.txt"

    return (
        st.session_state.get("selected_text", ""),
        st.session_state.get("selected_name", ""),
        st.session_state.get("selected_source", "Use default temples.txt"),
    )


def show_text_preview(label, text, limit=900):
    preview = text[:limit]
    if len(text) > limit:
        preview = preview + "\n..."
    st.caption(label)
    st.code(preview if preview else "No text available.", language=None)


def show_source_note():
    st.markdown(
        """
        <div class="note">
        <strong>Model note.</strong>
        Uploaded .txt files are used for text preprocessing and EDA analysis.
        The pretrained embedding and language model are based on the original temples.txt corpus.
        Retraining on uploaded text can be added as future work.
        </div>
        """,
        unsafe_allow_html=True,
    )


def small_svg_icon(kind):
    # Simple monochrome line icons. They are inline SVG, not emoji.
    icons = {
        "text": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 6h14M5 12h14M5 18h9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        "clean": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 19l10-10M13 5l6 6M15 3l6 6M3 21l4-1 12-12-3-3L4 17l-1 4z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "tokens": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect x="4" y="5" width="6" height="6" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="14" y="5" width="6" height="6" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="4" y="15" width="6" height="4" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="14" y="15" width="6" height="4" rx="1.5" stroke="currentColor" stroke-width="1.8"/></svg>',
        "chart": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 19V5M4 19h16M8 16V11M12 16V8M16 16v-6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        "network": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="6" cy="12" r="2.5" stroke="currentColor" stroke-width="1.8"/><circle cx="18" cy="6" r="2.5" stroke="currentColor" stroke-width="1.8"/><circle cx="18" cy="18" r="2.5" stroke="currentColor" stroke-width="1.8"/><path d="M8.2 10.8l7.6-3.6M8.2 13.2l7.6 3.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        "map": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 18l4-12 5 9 5-9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="14" cy="15" r="2" fill="currentColor"/></svg>',
        "model": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M8 6h8M8 18h8M12 6v12M6 10h12M6 14h12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
        "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 12l4 4L19 6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    }
    return icons.get(kind, icons["text"])


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title, body, icon_name):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="icon-badge">{small_svg_icon(icon_name)}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_card(number, title, body, icon_name):
    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="icon-badge">{small_svg_icon(icon_name)}</div>
            <div class="step-number">STEP {number}</div>
            <div class="pipeline-title">{title}</div>
            <div class="pipeline-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_card(number, title, body):
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-number">STEP {number}</div>
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_note(title, body):
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-title">{html.escape(title)}</div>
            <div class="model-card-body">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_embedding_result_card(selected_word, status, nearest_count):
    card_class = "result-card"
    if status != "Found":
        card_class = "result-card warning"
        status_text = "Not found"
    else:
        status_text = "Found in trained vocabulary"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="result-grid">
                <div>
                    <div class="result-label">Selected Word</div>
                    <div class="result-value">{html.escape(str(selected_word))}</div>
                </div>
                <div>
                    <div class="result-label">Vocabulary Status</div>
                    <div class="result-value">{html.escape(status_text)}</div>
                </div>
                <div>
                    <div class="result-label">Nearest Words</div>
                    <div class="result-value">{nearest_count}</div>
                </div>
                <div>
                    <div class="result-label">Model Source</div>
                    <div class="result-value">Original temples.txt corpus</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similarity_table(nearest_df):
    table_rows = []
    for row_index, row in nearest_df.iterrows():
        rank = int(row["Rank"])
        word = html.escape(str(row["Word"]))
        similarity = float(row["Cosine Similarity"])
        table_rows.append(
            "<tr>"
            + f"<td>{rank}</td>"
            + f"<td>{word}</td>"
            + f"<td><span class='similarity-score'>{similarity:.4f}</span></td>"
            + "</tr>"
        )

    table_html = (
        "<div class='table-shell'><table class='similarity-table'>"
        "<thead><tr><th>Rank</th><th>Similar Word</th><th>Cosine Similarity</th></tr></thead>"
        "<tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_similarity_interpretation():
    st.markdown(
        """
        <div class="interpretation-card">
            <strong>Similarity interpretation</strong><br>
            Above 0.50: strong contextual similarity.<br>
            Between 0.30 and 0.50: moderate contextual similarity.<br>
            Below 0.30: weak contextual similarity.
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_chart(fig, height=420, bargap=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13, color="#0B1F3A"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=40, r=40, t=70, b=60),
        colorway=["#0B1F3A", "#4E9FE5", "#C9A227", "#6C757D", "#1F9D55"],
    )
    fig.update_xaxes(gridcolor="#E6EAF0", zerolinecolor="#E6EAF0")
    fig.update_yaxes(gridcolor="#E6EAF0", zerolinecolor="#E6EAF0")
    if bargap is not None:
        fig.update_layout(bargap=bargap)
    return fig


def build_requirement_checklist():
    rows = [
        ["Khmer text corpus", "Use temples.txt", "Default corpus is data/temples.txt or temples.txt", "Completed"],
        ["Khmer tokenization", "Use Khmer tokenizer", "khmernltk / khmer-nltk with fallback handling", "Completed"],
        ["Frequency filtering", "Ignore words with frequency < 10", "MIN_FREQ = 10", "Completed"],
        ["Space/dirty token removal", "Ignore spaces and noisy tokens", "punctuation, number-only, empty, and non-Khmer noise removed", "Completed"],
        ["Skip-gram embeddings", "Build skip-gram model/classifier", "Skip-gram with negative sampling", "Completed"],
        ["Embedding dimension", "50", "EMBEDDING_DIM = 50", "Completed"],
        ["Context window", "+/-4", "WINDOW_SIZE = 4", "Completed"],
        ["Negative sampling", "k = 2", "NEGATIVE_SAMPLES = 2", "Completed"],
        ["PCA visualization", "Select 2 components and plot words in 2D", "PCA map from learned skip-gram embeddings", "Completed"],
        ["Neural language model", "Predict next word from previous n words", "Neural LM predicts next word", "Completed"],
        ["Neural LM context size", "n = 5", "N_CONTEXT = 5", "Completed"],
        ["Hidden layer size", "h = 512", "HIDDEN_SIZE = 512", "Completed"],
        ["Scratch embedding model", "Learn embeddings from scratch during training", "Scratch neural LM implemented and compared", "Completed"],
        ["Result comparison", "Compare skip-gram embeddings and scratch embeddings", "Model comparison page and results table", "Completed"],
        ["Report summary", "Max 4-page summary", "Project Report page included", "Completed"],
    ]
    return pd.DataFrame(rows, columns=["Requirement", "Expected Setting", "Project Implementation", "Status"])


def get_model_comparison_data():
    comparison_df = load_csv_file(str(TABLE_DIR / "model_comparison.csv"))
    source_label = "Saved notebook result"
    if comparison_df.empty:
        comparison_df = fallback_model_comparison()
        source_label = "Fallback notebook result"
    return comparison_df, source_label


def get_best_model_row(comparison_df):
    if "Test Perplexity" not in comparison_df.columns:
        return None
    clean_df = comparison_df.copy()
    clean_df["Test Perplexity"] = pd.to_numeric(clean_df["Test Perplexity"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Test Perplexity"])
    if clean_df.empty:
        return None
    best_index = clean_df["Test Perplexity"].idxmin()
    return clean_df.loc[best_index]


def get_perplexity_value(comparison_df, model_name, fallback_value):
    if comparison_df.empty or "Model" not in comparison_df.columns or "Test Perplexity" not in comparison_df.columns:
        return fallback_value
    for row_index, row in comparison_df.iterrows():
        if str(row["Model"]) == model_name:
            try:
                return float(row["Test Perplexity"])
            except Exception:
                return fallback_value
    return fallback_value


def render_best_model_card(comparison_df, source_label):
    best_row = get_best_model_row(comparison_df)
    if best_row is None:
        model_name = "Neural LM with Fixed Skip-gram Embeddings"
        best_perplexity = 115.10
    else:
        model_name = str(best_row["Model"])
        best_perplexity = float(best_row["Test Perplexity"])

    ngram_perplexity = get_perplexity_value(comparison_df, "N-gram baseline", 158.84)
    fixed_perplexity = get_perplexity_value(comparison_df, "Neural LM fixed", 115.10)
    scratch_perplexity = get_perplexity_value(comparison_df, "Neural LM scratch", 229.07)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-grid">
                <div>
                    <div class="result-label">Best Performing Model</div>
                    <div class="result-value">{html.escape(model_name)}</div>
                </div>
                <div>
                    <div class="result-label">Best Test Perplexity</div>
                    <div class="result-value">{best_perplexity:.2f}</div>
                </div>
                <div>
                    <div class="result-label">Result Source</div>
                    <div class="result-value">{html.escape(source_label)}</div>
                </div>
                <div>
                    <div class="result-label">Selected Reason</div>
                    <div class="result-value">Lowest perplexity</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_model_note(
        "Why it performs best",
        "This model uses pretrained skip-gram embeddings from the Khmer temple corpus. The embeddings provide useful contextual information, helping the neural language model generalize better than the scratch embedding model on a small corpus.",
    )
    columns = st.columns(3)
    columns[0].metric("N-gram Perplexity", f"{ngram_perplexity:.2f}")
    columns[1].metric("Fixed Skip-gram LM", f"{fixed_perplexity:.2f}")
    columns[2].metric("Scratch LM", f"{scratch_perplexity:.2f}")
    st.write(
        "Lower perplexity means the model is less surprised by the test text. "
        "In this project, the fixed skip-gram neural LM achieved the lowest perplexity, "
        "so it is selected as the best model."
    )


def render_summary_card(title, body):
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-title">{html.escape(title)}</div>
            <div class="model-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_metric_grid(items):
    columns = st.columns(len(items))
    for index, item in enumerate(items):
        label, value, help_text = item
        columns[index].metric(label, value, help=help_text)


def get_project_metrics():
    cleaned_tokens = load_csv_file(str(TABLE_DIR / "cleaned_tokens.csv"))
    vocabulary = load_csv_file(str(TABLE_DIR / "vocabulary.csv"))

    if "token" in cleaned_tokens.columns:
        total_tokens = len(cleaned_tokens)
    else:
        total_tokens = 0

    if "word" in vocabulary.columns:
        vocab_size = len(vocabulary)
    else:
        vocab_size = 0

    return total_tokens, vocab_size


def load_embedding_resources():
    embeddings = load_numpy_file(str(EMBEDDING_DIR / "skipgram_embeddings.npy"))
    word2idx = load_json_file(str(EMBEDDING_DIR / "word2idx.json"))
    idx2word_json = load_json_file(str(EMBEDDING_DIR / "idx2word.json"))
    vocabulary = load_csv_file(str(TABLE_DIR / "vocabulary.csv"))
    token_frequency = load_csv_file(str(TABLE_DIR / "token_frequency.csv"))

    idx2word = []
    if idx2word_json:
        sorted_items = sorted(idx2word_json.items(), key=lambda item: int(item[0]))
        for key, value in sorted_items:
            idx2word.append(value)
    elif "word" in vocabulary.columns:
        idx2word = vocabulary["word"].astype(str).tolist()

    return embeddings, word2idx, idx2word, vocabulary, token_frequency


def prepare_clustering_data(embeddings, word2idx, idx2word, token_frequency):
    if embeddings is None or not word2idx or len(idx2word) == 0:
        return pd.DataFrame(), np.array([])

    frequency_map = {}
    if not token_frequency.empty and "word" in token_frequency.columns and "frequency" in token_frequency.columns:
        for row_index, row in token_frequency.iterrows():
            frequency_map[str(row["word"])] = float(row["frequency"])

    words_for_clustering = []
    vectors_for_clustering = []

    for word in idx2word:
        if word == "<UNK>":
            continue
        if word not in word2idx:
            continue
        word_index = int(word2idx[word])
        if word_index < 0 or word_index >= len(embeddings):
            continue
        words_for_clustering.append(word)
        vectors_for_clustering.append(embeddings[word_index])

    if len(vectors_for_clustering) == 0:
        return pd.DataFrame(), np.array([])

    vectors_array = np.array(vectors_for_clustering)

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        pca_coordinates = pca.fit_transform(vectors_array)
    except Exception:
        pca_coordinates = np.zeros((len(words_for_clustering), 2))

    rows = []
    for index, word in enumerate(words_for_clustering):
        rows.append(
            {
                "word": word,
                "frequency": frequency_map.get(word, 1.0),
                "PC1": pca_coordinates[index, 0],
                "PC2": pca_coordinates[index, 1],
            }
        )

    embedding_df = pd.DataFrame(rows)
    return embedding_df, vectors_array


def compute_silhouette_scores(vectors):
    if vectors is None or len(vectors) < 3:
        return pd.DataFrame(columns=["k", "silhouette_score"])

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception:
        return pd.DataFrame(columns=["k", "silhouette_score"])

    max_k = min(10, len(vectors) - 1)
    rows = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        rows.append({"k": k, "silhouette_score": float(score)})

    return pd.DataFrame(rows)


def run_kmeans_clustering(embedding_df, vectors, k):
    clustered_df = embedding_df.copy()
    if clustered_df.empty or vectors is None or len(vectors) == 0:
        clustered_df["cluster"] = []
        return clustered_df

    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=int(k), random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        clustered_df["cluster"] = labels.astype(str)
    except Exception:
        clustered_df["cluster"] = "0"

    return clustered_df


def create_cluster_summary(clustered_df):
    if clustered_df.empty or "cluster" not in clustered_df.columns:
        return pd.DataFrame(columns=["Cluster", "Number of Words", "Top Frequent Words", "Interpretation"])

    rows = []
    cluster_names = sorted(clustered_df["cluster"].astype(str).unique(), key=lambda value: int(value))
    for cluster_name in cluster_names:
        cluster_df = clustered_df[clustered_df["cluster"].astype(str) == cluster_name]
        cluster_df = cluster_df.sort_values("frequency", ascending=False)
        top_words = cluster_df["word"].astype(str).head(10).tolist()

        if len(cluster_df) >= 20:
            interpretation = "This cluster contains a broad group of contextual words from the temple corpus."
        elif float(cluster_df["frequency"].max()) >= 20:
            interpretation = "This cluster contains frequent temple/context words."
        else:
            interpretation = "This cluster may contain smaller contextual or domain-specific word groups."

        rows.append(
            {
                "Cluster": cluster_name,
                "Number of Words": len(cluster_df),
                "Top Frequent Words": ", ".join(top_words),
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def find_khmer_font():
    try:
        from matplotlib import font_manager
    except Exception:
        return None

    preferred_fonts = [
        "Noto Sans Khmer",
        "Noto Serif Khmer",
        "Khmer OS Siemreap",
        "Khmer OS Battambang",
        "Khmer OS Content",
        "Khmer OS",
        "DaunPenh",
        "Leelawadee UI",
    ]

    installed_fonts = font_manager.fontManager.ttflist
    for preferred_font in preferred_fonts:
        for font in installed_fonts:
            if preferred_font.lower() == font.name.lower():
                return font.name

    for preferred_font in preferred_fonts:
        for font in installed_fonts:
            if preferred_font.lower() in font.name.lower():
                return font.name

    return None


def contains_khmer_text(text):
    for character in str(text):
        if "\u1780" <= character <= "\u17FF":
            return True
    return False


def plot_cluster_pca(clustered_df):
    fig = px.scatter(
        clustered_df,
        x="PC1",
        y="PC2",
        color="cluster",
        size="frequency",
        hover_data=["word", "frequency", "cluster"],
        title="PCA Visualization of Word Embeddings Colored by K-means Cluster",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "cluster": "Cluster"},
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=0.5, color="#0B1F3A")))
    return style_plotly_chart(fig, height=560)


def plot_dendrogram(clustered_df, vectors, number_words=40):
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
    except Exception as error:
        st.warning("Hierarchical clustering dependencies are not available: " + str(error))
        return None

    if clustered_df.empty or vectors is None or len(vectors) < 3:
        return None

    selected_df = clustered_df.sort_values("frequency", ascending=False).head(number_words)
    selected_indices = selected_df.index.tolist()
    selected_vectors = vectors[selected_indices]
    selected_words = selected_df["word"].astype(str).tolist()

    distance_matrix = pdist(selected_vectors, metric="cosine")
    linkage_matrix = linkage(distance_matrix, method="average")

    khmer_font_name = find_khmer_font()
    if khmer_font_name is not None:
        plt.rcParams["font.family"] = [khmer_font_name, "DejaVu Sans"]
    else:
        st.warning("No Khmer-supported font found. Khmer labels may not display correctly. Please install Noto Sans Khmer or Khmer OS fonts.")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, labels=selected_words, leaf_rotation=90, leaf_font_size=12, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram of Frequent Khmer Words", fontsize=16, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylabel("Cosine Distance", fontsize=12, fontname="DejaVu Sans")
    ax.grid(axis="y", alpha=0.25)
    if khmer_font_name is not None:
        for label in ax.get_xticklabels():
            if contains_khmer_text(label.get_text()):
                label.set_fontfamily([khmer_font_name, "DejaVu Sans"])
            else:
                label.set_fontname("DejaVu Sans")
            label.set_fontsize(11)
    fig.tight_layout()
    return fig


def plot_numbered_dendrogram(clustered_df, vectors, number_words=40):
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
    except Exception as error:
        st.warning("Hierarchical clustering dependencies are not available: " + str(error))
        return None, pd.DataFrame()

    if clustered_df.empty or vectors is None or len(vectors) < 3:
        return None, pd.DataFrame()

    selected_df = clustered_df.sort_values("frequency", ascending=False).head(number_words).copy()
    selected_indices = selected_df.index.tolist()
    selected_vectors = vectors[selected_indices]

    label_rows = []
    short_labels = []
    for label_index, row in enumerate(selected_df.itertuples(), start=1):
        short_id = "W" + str(label_index).zfill(2)
        short_labels.append(short_id)
        label_rows.append({"ID": short_id, "Khmer Word": row.word, "Frequency": row.frequency})

    distance_matrix = pdist(selected_vectors, metric="cosine")
    linkage_matrix = linkage(distance_matrix, method="average")

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, labels=short_labels, leaf_rotation=90, leaf_font_size=12, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram of Frequent Khmer Words", fontsize=16, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylabel("Cosine Distance", fontsize=12, fontname="DejaVu Sans")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    label_df = pd.DataFrame(label_rows)
    return fig, label_df


def cosine_nearest_words(query_word, embeddings, word2idx, idx2word, top_k=10):
    if query_word not in word2idx:
        return pd.DataFrame()

    query_index = int(word2idx[query_word])
    query_vector = embeddings[query_index]
    query_norm = np.linalg.norm(query_vector)

    rows = []
    for index in range(len(idx2word)):
        if index == query_index:
            continue
        other_vector = embeddings[index]
        denominator = query_norm * np.linalg.norm(other_vector)
        if denominator == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vector, other_vector) / denominator)
        rows.append({"Rank": 0, "Word": idx2word[index], "Cosine Similarity": similarity})

    rows = sorted(rows, key=lambda row: row["Cosine Similarity"], reverse=True)
    top_rows = rows[:top_k]
    for index, row in enumerate(top_rows):
        row["Rank"] = index + 1
    return pd.DataFrame(top_rows)


def fallback_model_comparison():
    return pd.DataFrame(
        [
            {"Model": "N-gram baseline", "Test Perplexity": 158.84, "Top-1 Accuracy": np.nan, "Top-5 Accuracy": np.nan, "Notes": "Sample result from notebook"},
            {"Model": "Neural LM fixed", "Test Perplexity": 115.10, "Top-1 Accuracy": np.nan, "Top-5 Accuracy": np.nan, "Notes": "Sample result from notebook"},
            {"Model": "Neural LM scratch", "Test Perplexity": 229.07, "Top-1 Accuracy": np.nan, "Top-5 Accuracy": np.nan, "Notes": "Sample result from notebook"},
        ]
    )


if torch is not None and nn is not None:
    class NeuralLanguageModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, n_context, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.hidden = nn.Linear(n_context * embedding_dim, hidden_size)
            self.relu = nn.ReLU()
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, context_ids):
            embedded = self.embedding(context_ids)
            flattened = embedded.reshape(embedded.shape[0], -1)
            hidden_output = self.relu(self.hidden(flattened))
            logits = self.output(hidden_output)
            return logits


@st.cache_resource
def load_fixed_lm_model(model_path_text, vocab_size):
    if torch is None or nn is None:
        return None, "PyTorch is not available."

    model_path = Path(model_path_text)
    if not model_path.exists():
        return None, "Neural LM model file not found."

    try:
        model = NeuralLanguageModel(vocab_size, EMBEDDING_DIM, N_CONTEXT, HIDDEN_SIZE)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, ""
    except Exception as error:
        return None, str(error)


def frequency_fallback_prediction(token_frequency, top_k=5):
    if token_frequency.empty or "word" not in token_frequency.columns:
        return pd.DataFrame()
    top_words = token_frequency.head(top_k)
    total = float(top_words["frequency"].sum())
    rows = []
    for index, row in top_words.iterrows():
        if total == 0:
            probability = 0.0
        else:
            probability = float(row["frequency"] / total)
        rows.append({"Rank": len(rows) + 1, "Predicted Word": row["word"], "Probability": probability})
    return pd.DataFrame(rows)


def neural_prediction(input_text, model, word2idx, idx2word):
    cleaned, raw_tokens, useful_tokens, tokenizer_name = preprocess_text(input_text)
    if len(useful_tokens) == 0:
        useful_tokens = input_text.split()

    context_tokens = useful_tokens[-N_CONTEXT:]
    while len(context_tokens) < N_CONTEXT:
        context_tokens.insert(0, "<UNK>")

    context_ids = []
    for token in context_tokens:
        context_ids.append(int(word2idx.get(token, word2idx.get("<UNK>", 0))))

    with torch.no_grad():
        tensor = torch.tensor([context_ids], dtype=torch.long)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_values, top_indices = torch.topk(probabilities, k=min(5, len(idx2word)))

    rows = []
    for index in range(len(top_indices)):
        word_index = int(top_indices[index].item())
        rows.append(
            {
                "Rank": index + 1,
                "Predicted Word": idx2word[word_index],
                "Probability": float(top_values[index].item()),
            }
        )
    return pd.DataFrame(rows), context_tokens, tokenizer_name


def render_eda(tokens):
    if len(tokens) == 0:
        st.warning("No useful Khmer tokens were found for EDA.")
        return

    freq_df = build_frequency_dataframe(tokens)
    average_length = float(np.mean([len(token) for token in tokens]))
    most_frequent_word = freq_df.iloc[0]["word"]
    rare_words = int((freq_df["frequency"] < MIN_FREQ).sum())

    metric_columns = st.columns(5)
    metric_columns[0].metric("Useful Tokens", f"{len(tokens):,}")
    metric_columns[1].metric("Unique Tokens", f"{len(freq_df):,}")
    metric_columns[2].metric("Avg Token Length", f"{average_length:.2f}")
    metric_columns[3].metric("Most Frequent Word", str(most_frequent_word))
    metric_columns[4].metric("Rare Words", f"{rare_words:,}")

    tab1, tab2, tab3 = st.tabs(["Frequency", "Length", "Relationships"])

    with tab1:
        top30 = freq_df.head(30)
        top30_for_plot = top30.sort_values("frequency", ascending=True)
        fig = px.bar(
            top30_for_plot,
            x="frequency",
            y="word",
            orientation="h",
            text="frequency",
            title="Top 30 Frequent Khmer Words",
            labels={"word": "Word", "frequency": "Frequency"},
            hover_data=["length"],
        )
        fig.update_traces(
            textposition="outside",
            marker_color="#0B1F3A",
            marker_line_color="#C9A227",
            marker_line_width=0.8,
        )
        fig.update_xaxes(title="Frequency")
        fig.update_yaxes(title="Word")
        fig = style_plotly_chart(fig, height=650, bargap=0.25)
        st.plotly_chart(fig, use_container_width=True)

        use_log_scale = st.checkbox("Use log scale for y-axis", value=True)
        fig = px.histogram(
            freq_df,
            x="frequency",
            nbins=30,
            title="Word Frequency Distribution (Highly Skewed)",
            labels={"frequency": "Frequency"},
        )
        fig.update_traces(marker_color="#4E9FE5", marker_line_color="#FFFFFF", marker_line_width=0.8)
        fig = style_plotly_chart(fig, height=420, bargap=0.1)
        if use_log_scale:
            fig.update_yaxes(type="log", title="Number of Words (log scale)")
        else:
            fig.update_yaxes(title="Number of Words")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Word frequencies are highly skewed. Most words occur rarely, while a few words occur very frequently.")

        fig = px.line(
            freq_df,
            x="rank",
            y="frequency",
            title="Rank vs Frequency (Zipf-like Curve)",
            labels={"rank": "Frequency Rank", "frequency": "Frequency"},
            hover_data=["word"],
        )
        fig.update_traces(line_color="#C9A227", line_width=2.5)
        fig = style_plotly_chart(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        length_df = pd.DataFrame({"token_length": [len(token) for token in tokens]})
        fig = px.histogram(
            length_df,
            x="token_length",
            nbins=30,
            title="Token Length Distribution",
            labels={"token_length": "Token Length"},
        )
        fig.update_traces(marker_color="#0B1F3A", marker_line_color="#FFFFFF", marker_line_width=0.8)
        fig = style_plotly_chart(fig, height=420, bargap=0.12)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            freq_df,
            x="frequency_group",
            y="length",
            title="Frequency Group vs Token Length",
            labels={"frequency_group": "Frequency Group", "length": "Token Length"},
        )
        fig = style_plotly_chart(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(
            freq_df,
            x="length",
            y="frequency",
            hover_data=["word"],
            title="Token Length vs Frequency",
            labels={"length": "Token Length", "frequency": "Frequency"},
        )
        fig.update_traces(marker=dict(color="#4E9FE5", size=8, line=dict(width=0.6, color="#0B1F3A")))
        fig = style_plotly_chart(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Khmer Temple<br>NLP Explorer</div>
        <div class="sidebar-brand-caption">NLP Mini Project 3</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Presentation Summary",
        "Mini Project Checklist",
        "Text File Input & Preprocessing",
        "EDA Dashboard",
        "Word Embedding Explorer",
        "PCA Map",
        "Advanced Clustering",
        "Next Word Prediction",
        "Model Comparison",
        "Project Report",
        "About & Deployment",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Master of Data Science")
st.sidebar.caption("Institute of Technology of Cambodia")


# -----------------------------
# Page 1: Home
# -----------------------------
if page == "Home":
    total_tokens, vocab_size = get_project_metrics()
    if total_tokens == 0:
        total_tokens = 9075
    if vocab_size == 0:
        vocab_size = 175
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Khmer NLP Portfolio Project</div>
            <h1>Khmer Temple NLP Explorer</h1>
            <p>Khmer Word Embeddings, PCA Visualization, and Neural Language Modeling</p>
            <p>
            A professional dashboard for demonstrating Khmer text preprocessing, semantic word vectors,
            and language-model evaluation from NLP Mini Project 3.
            </p>
            <div class="hero-accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Key Project Metrics</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Total Tokens", f"{total_tokens:,}")
    with metric_columns[1]:
        render_metric_card("Vocabulary Size", f"{vocab_size:,}")
    with metric_columns[2]:
        render_metric_card("Embedding Dimension", str(EMBEDDING_DIM))
    with metric_columns[3]:
        render_metric_card("Best Model Perplexity", "115.10")

    with st.expander("View model settings"):
        setting_columns = st.columns(4)
        setting_columns[0].metric("Context Window", "+/-4")
        setting_columns[1].metric("Negative Samples", NEGATIVE_SAMPLES)
        setting_columns[2].metric("Neural LM Context", N_CONTEXT)
        setting_columns[3].metric("Hidden Layer", HIDDEN_SIZE)

    show_source_note()

    st.markdown('<div class="section-title">Main Actions</div>', unsafe_allow_html=True)
    feature_columns = st.columns(3)
    feature_items = [
        ("Analyze Khmer Text", "Use default temples.txt or upload a Khmer .txt file for preprocessing and EDA.", "text"),
        ("Explore Word Embeddings", "Search trained vocabulary and inspect nearest words using cosine similarity.", "network"),
        ("Review Model Results", "Compare n-gram, fixed embedding neural LM, and scratch embedding neural LM.", "chart"),
    ]
    for column_index, item in enumerate(feature_items):
        title, body, icon_name = item
        with feature_columns[column_index]:
            render_feature_card(title, body, icon_name)

    with st.expander("View full project pipeline"):
        st.markdown('<div class="section-title">Compact Project Pipeline</div>', unsafe_allow_html=True)
        pipeline_items = [
            ("Khmer Text", "Original corpus or uploaded Khmer .txt file.", "text"),
            ("Cleaning", "Remove references, URLs, and repeated spaces.", "clean"),
            ("Tokenization", "Create useful Khmer word tokens.", "tokens"),
            ("EDA", "Explore frequency and token patterns.", "chart"),
            ("Skip-gram", "Learn context-based word embeddings.", "network"),
            ("PCA", "Project embeddings into 2D.", "map"),
            ("Neural LM", "Predict the next word.", "model"),
            ("Evaluation", "Compare model results.", "check"),
        ]
        for start in range(0, len(pipeline_items), 4):
            pipeline_columns = st.columns(4)
            for column_index, item in enumerate(pipeline_items[start:start + 4]):
                title, body, icon_name = item
                with pipeline_columns[column_index]:
                    step_number = str(start + column_index + 1).zfill(2)
                    render_pipeline_card(step_number, title, body, icon_name)

    render_model_note(
        "Recommended Demo Flow",
        "1. Text File Input & Preprocessing  |  2. EDA Dashboard  |  3. Word Embedding Explorer  |  4. PCA Map  |  5. Model Comparison",
    )

    st.markdown('<div class="section-title">How to Use This App</div>', unsafe_allow_html=True)
    step_columns = st.columns(3)
    with step_columns[0]:
        render_step_card("01", "Choose a Text Source", "Use the default temples.txt corpus or upload a Khmer .txt file for analysis.")
    with step_columns[1]:
        render_step_card("02", "Explore Preprocessing and EDA", "Review cleaned text, useful tokens, word frequencies, and interactive charts.")
    with step_columns[2]:
        render_step_card("03", "Use Trained NLP Outputs", "Explore pretrained embeddings, PCA maps, model comparison, and next-word prediction.")

    with st.expander("What this app demonstrates"):
        st.write(
            "The app shows how Khmer text can be cleaned, tokenized, converted into word vectors, "
            "visualized with PCA, and used for language modeling. It also explains why small, "
            "domain-specific corpora can produce high `<UNK>` rates and overfitting in neural models."
        )

# -----------------------------
# Page 2: Presentation Summary
# -----------------------------
elif page == "Presentation Summary":
    st.title("Presentation Summary")
    total_tokens, vocab_size = get_project_metrics()
    if total_tokens == 0:
        total_tokens = 9075
    if vocab_size == 0:
        vocab_size = 175

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Professor Presentation View</div>
            <h1>Mini Project 3 Summary</h1>
            <p>A concise story of the Khmer word embedding and neural language modeling project.</p>
            <div class="hero-accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    columns[0].metric("Dataset", "temples.txt")
    columns[1].metric("Total Tokens", f"{total_tokens:,}")
    columns[2].metric("Vocabulary Size", f"{vocab_size:,}")
    columns[3].metric("Best Perplexity", "115.10")

    render_summary_card(
        "1. Project Objective",
        "This project builds Khmer word embeddings and neural language models using a Khmer temple corpus. It demonstrates the full NLP workflow from preprocessing to semantic representation, PCA visualization, and next-word prediction.",
    )
    render_summary_card(
        "2. Dataset",
        "Dataset: temples.txt<br>Topic: Khmer temple Wikipedia text<br>Total tokens: "
        + f"{total_tokens:,}<br>Vocabulary size: {vocab_size:,} after MIN_FREQ = 10<br>"
        + "Uploaded .txt files are for EDA only in the Streamlit app.",
    )
    render_summary_card(
        "3. Preprocessing Result",
        "Khmer tokenization is applied. Punctuation, number-only tokens, noisy tokens, and empty tokens are removed. Frequency filtering uses MIN_FREQ = 10, and rare words are represented with &lt;UNK&gt;.",
    )
    render_summary_card(
        "4. Embedding Method",
        "The project uses skip-gram with negative sampling. Embedding dimension = 50, context window = +/-4, and negative samples = 2.",
    )
    render_summary_card(
        "5. PCA Result",
        "PCA reduces 50-dimensional embeddings into 2D for interactive visualization. This helps inspect contextual word relationships in the temple corpus.",
    )
    render_summary_card(
        "6. Language Modeling Result",
        "The app compares an n-gram baseline, fixed skip-gram neural LM, and scratch embedding neural LM. The best model is the fixed skip-gram neural LM, with test perplexity around 115.10.",
    )
    render_summary_card(
        "7. Main Conclusion",
        "The fixed skip-gram neural language model achieved the best perplexity, showing that pretrained contextual embeddings helped next-word prediction on the Khmer temple corpus.",
    )
    render_summary_card(
        "8. Limitations",
        "The corpus is small and domain-specific. Khmer tokenization can still be challenging. Some rare meaningful words become &lt;UNK&gt; because MIN_FREQ = 10. Nearest words may show contextual association rather than perfect synonyms.",
    )
    render_summary_card(
        "9. Future Work",
        "Train on a larger Khmer corpus, add a retraining option for uploaded .txt files, improve Khmer tokenizer and compound word handling, and deploy the project as an educational Khmer NLP demo.",
    )


# -----------------------------
# Page 3: Mini Project Checklist
# -----------------------------
elif page == "Mini Project Checklist":
    st.title("Mini Project 3 Requirement Checklist")
    render_model_note(
        "Requirement mapping",
        "This checklist maps the deployed system to the professor's Mini Project 3 requirements.",
    )

    checklist_df = build_requirement_checklist()
    st.dataframe(checklist_df, use_container_width=True, hide_index=True)


# -----------------------------
# Page 4: Text File Input & Preprocessing
# -----------------------------
elif page == "Text File Input & Preprocessing":
    st.title("Text File Input & Preprocessing")
    show_source_note()

    text_source = st.radio(
        "Choose text source",
        ["Use default temples.txt", "Upload new .txt file"],
        horizontal=True,
    )

    selected_text = ""
    selected_name = ""

    if text_source == "Use default temples.txt":
        selected_text, selected_name = load_default_text()
        if selected_text == "":
            st.error("Default corpus was not found. Please add `data/temples.txt` or `temples.txt`.")
    else:
        uploaded_file = st.file_uploader("Upload a Khmer .txt file", type=["txt"])
        selected_text, selected_name = read_uploaded_txt(uploaded_file)
        if uploaded_file is None:
            st.info("Upload a `.txt` file to analyze your own Khmer text.")
        elif selected_text == "":
            st.error("The uploaded file could not be read as UTF-8 or UTF-8-SIG text.")
        else:
            st.success("Uploaded file loaded: " + selected_name)

    if selected_text:
        st.session_state["selected_text"] = selected_text
        st.session_state["selected_name"] = selected_name
        st.session_state["selected_source"] = text_source

        cleaned, raw_tokens, useful_tokens, tokenizer_name = preprocess_text(selected_text)
        freq_df = build_frequency_dataframe(useful_tokens)
        average_token_length = 0.0
        if len(useful_tokens) > 0:
            average_token_length = float(np.mean([len(token) for token in useful_tokens]))

        st.subheader("Raw Text Preview")
        st.metric("Source", selected_name)
        st.metric("Character Count", f"{len(selected_text):,}")
        show_text_preview("Raw text", selected_text)

        st.subheader("Cleaned Text and Tokens")
        show_text_preview("Cleaned text", cleaned)
        st.write("Tokenizer used:", tokenizer_name)
        st.write("First 100 useful tokens:")
        st.code(str(useful_tokens[:100]), language=None)

        columns = st.columns(4)
        columns[0].metric("Raw Tokens", f"{len(raw_tokens):,}")
        columns[1].metric("Useful Tokens", f"{len(useful_tokens):,}")
        columns[2].metric("Unique Tokens", f"{len(freq_df):,}")
        columns[3].metric("Avg Token Length", f"{average_token_length:.2f}")

        st.dataframe(freq_df.head(30), use_container_width=True)


# -----------------------------
# Page 3: EDA Dashboard
# -----------------------------
elif page == "EDA Dashboard":
    st.title("EDA Dashboard")
    selected_text, selected_name, selected_source = get_current_text()
    st.caption("Current text source: " + selected_source + " | " + selected_name)

    if selected_text == "":
        st.error("No text is available. Please add `data/temples.txt` or upload a `.txt` file.")
    else:
        cleaned, raw_tokens, useful_tokens, tokenizer_name = preprocess_text(selected_text)
        st.info("EDA is based on the currently selected text. Uploaded files use preprocessing and EDA only.")
        render_eda(useful_tokens)


# -----------------------------
# Page 4: Word Embedding Explorer
# -----------------------------
elif page == "Word Embedding Explorer":
    st.title("Word Embedding Explorer")
    show_source_note()
    render_model_note(
        "How cosine similarity is used",
        "Cosine similarity compares the selected word vector with other word vectors learned by the skip-gram model. Higher similarity means the words appeared in more similar contexts in the temple corpus.",
    )
    render_model_note(
        "Small-corpus limitation",
        "Because the training corpus is small and temple-specific, nearest words may represent contextual association rather than perfect synonym meaning.",
    )

    embeddings, word2idx, idx2word, vocabulary, token_frequency = load_embedding_resources()
    if embeddings is None or not word2idx or len(idx2word) == 0:
        st.warning("Embedding files not found. Please run the notebook first to generate outputs.")
    else:
        available_words = [word for word in idx2word if word != "<UNK>"]
        default_index = 0
        selected_word = st.selectbox("Select a Khmer word from trained vocabulary", available_words, index=default_index)
        typed_word = st.text_input("Or type a Khmer word to search inside trained vocabulary")

        if typed_word.strip() != "":
            query_word = typed_word.strip()
        else:
            query_word = selected_word

        if query_word in word2idx:
            nearest_df = cosine_nearest_words(query_word, embeddings, word2idx, idx2word, top_k=10)
            nearest_df = nearest_df.rename(columns={"Word": "Similar Word"})
            nearest_df["Cosine Similarity"] = nearest_df["Cosine Similarity"].round(4)
            table_df = nearest_df.rename(columns={"Similar Word": "Word"})
            render_embedding_result_card(query_word, "Found", len(nearest_df))
            render_similarity_table(table_df)
            render_similarity_interpretation()
        else:
            render_embedding_result_card(query_word, "Not Found", 0)
            st.warning("This word is not in the trained vocabulary.")
            st.write("Try one of these frequent available words:")
            if not token_frequency.empty and "word" in token_frequency.columns:
                st.dataframe(token_frequency.head(20), use_container_width=True)
            else:
                st.write(available_words[:30])


# -----------------------------
# Page 5: PCA Map
# -----------------------------
elif page == "PCA Map":
    st.title("PCA Map")
    show_source_note()
    render_model_note(
        "PCA visualization note",
        "This map uses pretrained skip-gram embeddings from the original temples.txt corpus. PCA compresses 50-dimensional vectors into two dimensions, so it should be read as an approximate visual guide.",
    )

    embeddings, word2idx, idx2word, vocabulary, token_frequency = load_embedding_resources()
    if embeddings is None or len(idx2word) == 0:
        st.warning("Embedding files not found. Please run the notebook first to generate outputs.")
    else:
        try:
            from sklearn.decomposition import PCA

            frequency_map = {}
            if not token_frequency.empty and "word" in token_frequency.columns:
                for row_index, row in token_frequency.iterrows():
                    frequency_map[str(row["word"])] = float(row["frequency"])

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embeddings)

            rows = []
            for index, word in enumerate(idx2word):
                frequency = frequency_map.get(word, 1.0)
                rows.append(
                    {
                        "word": word,
                        "PC1": coords[index, 0],
                        "PC2": coords[index, 1],
                        "frequency": frequency,
                    }
                )
            pca_df = pd.DataFrame(rows)
            pca_df = pca_df[pca_df["word"] != "<UNK>"]

            left, right = st.columns(2)
            number_words = left.slider("Number of words to display", 30, min(200, len(pca_df)), min(100, len(pca_df)))
            min_frequency = right.slider("Minimum frequency", 1, int(max(pca_df["frequency"].max(), 1)), 1)
            highlight_word = st.text_input("Search/highlight a word on the PCA map")

            filtered_df = pca_df[pca_df["frequency"] >= min_frequency].sort_values("frequency", ascending=False).head(number_words)
            filtered_df["frequency_group"] = pd.cut(
                filtered_df["frequency"],
                bins=[0, 5, 10, 20, 50, 100000],
                labels=["1-5", "6-10", "11-20", "21-50", "51+"],
            )

            fig = px.scatter(
                filtered_df,
                x="PC1",
                y="PC2",
                hover_data=["word", "frequency"],
                size="frequency",
                color="frequency_group",
                title="PCA Map of Skip-gram Khmer Word Embeddings",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
            )

            if highlight_word.strip() != "" and highlight_word.strip() in pca_df["word"].values:
                selected_row = pca_df[pca_df["word"] == highlight_word.strip()]
                fig.add_scatter(
                    x=selected_row["PC1"],
                    y=selected_row["PC2"],
                    mode="markers+text",
                    text=selected_row["word"],
                    textposition="top center",
                    marker=dict(size=18, color="#dc2626", symbol="star"),
                    name="Highlighted word",
                )

            st.plotly_chart(fig, use_container_width=True)
            st.info("PCA compresses 50-dimensional embeddings into 2D, so the plot is an approximate visualization of word relationships.")
        except Exception as error:
            st.error("Could not create PCA map: " + str(error))


# -----------------------------
# Page 6: Advanced Clustering
# -----------------------------
elif page == "Advanced Clustering":
    st.title("Advanced Clustering of Khmer Word Embeddings")
    show_source_note()
    render_model_note(
        "Optional advanced analysis",
        "This page explores the learned skip-gram word embeddings using K-means and hierarchical clustering. These methods are used for interpretation and do not replace the required Mini Project 3 tasks.",
    )
    render_model_note(
        "How to read this page",
        "Clustering is performed on the original 50-dimensional embeddings. PCA is used only to visualize the clusters in 2D.",
    )

    embeddings, word2idx, idx2word, vocabulary, token_frequency = load_embedding_resources()
    if embeddings is None or not word2idx or len(idx2word) == 0:
        st.warning("Embedding files not found. Please run the notebook first to generate skip-gram embeddings.")
    else:
        embedding_df, vectors = prepare_clustering_data(embeddings, word2idx, idx2word, token_frequency)
        if embedding_df.empty or len(vectors) < 3:
            st.warning("Not enough embedding data is available for clustering.")
        else:
            saved_silhouette = load_csv_file(str(TABLE_DIR / "kmeans_silhouette_scores.csv"))
            if saved_silhouette.empty:
                silhouette_df = compute_silhouette_scores(vectors)
                silhouette_source = "Computed from saved embeddings"
            else:
                silhouette_df = saved_silhouette
                silhouette_source = "Saved notebook output"

            st.subheader("K-means Silhouette Score")
            if silhouette_df.empty:
                st.warning("Could not calculate silhouette scores. Please check scikit-learn installation.")
                best_k = 2
                best_score = 0.0
            else:
                silhouette_df["k"] = pd.to_numeric(silhouette_df["k"], errors="coerce")
                silhouette_df["silhouette_score"] = pd.to_numeric(silhouette_df["silhouette_score"], errors="coerce")
                silhouette_df = silhouette_df.dropna(subset=["k", "silhouette_score"])
                best_row = silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]
                best_k = int(best_row["k"])
                best_score = float(best_row["silhouette_score"])

                fig = px.line(
                    silhouette_df,
                    x="k",
                    y="silhouette_score",
                    markers=True,
                    title="Silhouette Score for K-means Clustering",
                    labels={"k": "Number of clusters K", "silhouette_score": "Silhouette Score"},
                )
                fig.update_traces(line=dict(color="#0B1F3A", width=3), marker=dict(size=9, color="#C9A227"))
                fig = style_plotly_chart(fig, height=390)
                st.plotly_chart(fig, use_container_width=True)

                columns = st.columns(3)
                with columns[0]:
                    render_metric_card("Best K", str(best_k))
                with columns[1]:
                    render_metric_card("Silhouette Score", f"{best_score:.4f}")
                with columns[2]:
                    render_metric_card("Score Source", silhouette_source)

                st.info(
                    "The best K is the value with the highest silhouette score. "
                    "A higher score means words inside the same cluster are more similar and more separated from other clusters."
                )

            st.subheader("K-means PCA Cluster Map")
            max_words = len(embedding_df)
            min_words = min(30, max_words)
            default_words = min(100, max_words)
            left, right = st.columns(2)
            number_words = left.slider("Number of words shown", min_words, max_words, default_words)
            k_options = ["Use best K"]
            for k in range(2, min(10, len(vectors) - 1) + 1):
                k_options.append(str(k))
            selected_k_option = right.selectbox("Number of clusters", k_options)
            if selected_k_option == "Use best K":
                selected_k = best_k
            else:
                selected_k = int(selected_k_option)

            clustered_df = run_kmeans_clustering(embedding_df, vectors, selected_k)
            visible_df = clustered_df.sort_values("frequency", ascending=False).head(number_words)
            cluster_fig = plot_cluster_pca(visible_df)
            st.plotly_chart(cluster_fig, use_container_width=True)
            st.caption("Words in the same color belong to the same K-means cluster. Interpret clusters as contextual groups, not perfect semantic categories.")

            st.subheader("Cluster Summary Table")
            saved_summary = load_csv_file(str(TABLE_DIR / "kmeans_cluster_summary.csv"))
            if saved_summary.empty or selected_k_option != "Use best K":
                summary_df = create_cluster_summary(clustered_df)
            else:
                summary_df = saved_summary
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.subheader("Top Words by Cluster")
            cluster_list = sorted(clustered_df["cluster"].astype(str).unique(), key=lambda value: int(value))
            selected_cluster = st.selectbox("Select Cluster", cluster_list)
            cluster_words_df = clustered_df[clustered_df["cluster"].astype(str) == selected_cluster]
            cluster_words_df = cluster_words_df.sort_values("frequency", ascending=False)
            cluster_words_df = cluster_words_df.rename(columns={"word": "Word", "frequency": "Frequency", "cluster": "Cluster"})
            st.dataframe(cluster_words_df[["Word", "Frequency", "Cluster"]].head(30), use_container_width=True, hide_index=True)

            st.subheader("Hierarchical Clustering Dendrogram")
            min_dendrogram_words = min(20, len(clustered_df))
            max_dendrogram_words = min(50, len(clustered_df))
            default_dendrogram_words = min(40, len(clustered_df))
            label_style = st.radio(
                "Dendrogram label style",
                ["Khmer word labels", "Numbered labels with mapping table"],
                horizontal=True,
            )
            dendrogram_words = st.slider(
                "Number of words for dendrogram",
                min_dendrogram_words,
                max_dendrogram_words,
                default_dendrogram_words,
            )

            if label_style == "Khmer word labels":
                saved_dendrogram = FIGURE_DIR / "hierarchical_dendrogram_khmer.png"
                if saved_dendrogram.exists() and dendrogram_words == 40:
                    st.image(str(saved_dendrogram), caption="Hierarchical clustering dendrogram with Khmer word labels", use_container_width=True)
                else:
                    dendrogram_fig = plot_dendrogram(clustered_df, vectors, dendrogram_words)
                    if dendrogram_fig is None:
                        st.warning("Could not create dendrogram from the available data.")
                    else:
                        st.pyplot(dendrogram_fig, use_container_width=True)
            else:
                saved_numbered = FIGURE_DIR / "hierarchical_dendrogram_numbered.png"
                label_table = load_csv_file(str(TABLE_DIR / "dendrogram_word_labels.csv"))
                if saved_numbered.exists() and dendrogram_words == 40:
                    st.image(str(saved_numbered), caption="Hierarchical clustering dendrogram with numbered labels", use_container_width=True)
                else:
                    numbered_fig, label_table = plot_numbered_dendrogram(clustered_df, vectors, dendrogram_words)
                    if numbered_fig is None:
                        st.warning("Could not create numbered dendrogram from the available data.")
                    else:
                        st.pyplot(numbered_fig, use_container_width=True)

                st.markdown(
                    "To improve readability, this alternative dendrogram uses short word IDs. "
                    "The ID-to-word mapping table is provided below the plot."
                )
                if not label_table.empty:
                    st.dataframe(label_table, use_container_width=True, hide_index=True)

            st.info(
                "The dendrogram groups selected frequent words by embedding similarity. "
                "Shorter branches indicate more similar embedding patterns."
            )


# -----------------------------
# Page 7: Next Word Prediction
# -----------------------------
elif page == "Next Word Prediction":
    st.title("Next Word Prediction")
    show_source_note()
    render_model_note(
        "Prediction model note",
        "Next-word prediction uses the saved neural language model from the original temples.txt project. Uploaded files are not used to retrain the model in the app.",
    )

    embeddings, word2idx, idx2word, vocabulary, token_frequency = load_embedding_resources()
    if not word2idx or len(idx2word) == 0:
        st.warning("Vocabulary files not found. Please run the notebook first to generate outputs.")
    else:
        user_context = st.text_input("Enter 5 Khmer words separated by spaces")
        predict_button = st.button("Predict Next Word")

        if predict_button:
            if user_context.strip() == "":
                st.warning("Please enter 5 Khmer words.")
            else:
                model, model_error = load_fixed_lm_model(str(MODEL_DIR / "neural_lm_fixed.pt"), len(idx2word))
                if model is not None:
                    try:
                        prediction_df, context_tokens, tokenizer_name = neural_prediction(user_context, model, word2idx, idx2word)
                        st.success("Prediction from pretrained fixed skip-gram neural LM.")
                        st.write("Context tokens used:", context_tokens)
                        st.dataframe(prediction_df, use_container_width=True)
                    except Exception as error:
                        st.warning("Neural model prediction failed. Using frequency fallback instead.")
                        st.caption(str(error))
                        st.dataframe(frequency_fallback_prediction(token_frequency), use_container_width=True)
                else:
                    st.warning("Neural LM model file not found or could not be loaded. Using frequency fallback prediction.")
                    if model_error:
                        st.caption(model_error)
                    fallback_df = frequency_fallback_prediction(token_frequency)
                    if fallback_df.empty:
                        st.error("Fallback prediction is unavailable because token frequency output is missing.")
                    else:
                        st.dataframe(fallback_df, use_container_width=True)

        st.info("Next-word prediction uses the pretrained model from the original temples.txt corpus.")


# -----------------------------
# Page 7: Model Comparison
# -----------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    comparison_df, source_label = get_model_comparison_data()
    if source_label == "Fallback notebook result":
        st.warning("Model comparison CSV not found. Showing fallback notebook result values.")

    render_best_model_card(comparison_df, source_label)
    render_model_note(
        "How to read this comparison",
        "Lower perplexity means the model is less surprised by the test text. Accuracy metrics show how often the correct next word appears as the top prediction or within the top five predictions.",
    )

    st.dataframe(comparison_df, use_container_width=True)

    if "Test Perplexity" in comparison_df.columns:
        fig = px.bar(
            comparison_df,
            x="Model",
            y="Test Perplexity",
            text="Test Perplexity",
            title="Test Perplexity by Model (Lower is Better)",
            labels={"Test Perplexity": "Test Perplexity"},
            color="Model",
            color_discrete_sequence=["#0B1F3A", "#4E9FE5", "#C9A227", "#6C757D", "#1F9D55"],
        )
        fig.update_traces(
            texttemplate="%{y:.2f}",
            textposition="outside",
            marker_line_width=1,
            marker_line_color="#FFFFFF",
        )
        fig = style_plotly_chart(fig, height=420, bargap=0.5)
        fig.update_layout(legend_title_text="Model")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower perplexity means better language modeling performance.")

    for metric_name in ["Top-1 Accuracy", "Top-5 Accuracy"]:
        if metric_name in comparison_df.columns and comparison_df[metric_name].notna().sum() > 0:
            fig = px.bar(
                comparison_df,
                x="Model",
                y=metric_name,
                text=metric_name,
                title=metric_name + " by Model (Higher is Better)",
                color="Model",
                color_discrete_sequence=["#0B1F3A", "#4E9FE5", "#C9A227", "#6C757D", "#1F9D55"],
            )
            fig.update_traces(
                texttemplate="%{y:.2f}",
                textposition="outside",
                marker_line_width=1,
                marker_line_color="#FFFFFF",
            )
            fig = style_plotly_chart(fig, height=420, bargap=0.5)
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(legend_title_text="Model")
            st.plotly_chart(fig, use_container_width=True)

    st.write(
        "In the current cleaned notebook result, the fixed skip-gram neural LM has the lowest "
        "test perplexity. This suggests the pretrained skip-gram embeddings helped next-word "
        "prediction. The scratch LM can overfit because the corpus is small."
    )


# -----------------------------
# Page 8: Project Report
# -----------------------------
elif page == "Project Report":
    st.title("Project Report")

    sections = {
        "Project Objective": "Build Khmer word embeddings and neural language models for a Khmer temple corpus.",
        "Dataset": "The main dataset is `data/temples.txt`, a domain-specific Khmer corpus about temples and Angkor-related content.",
        "Preprocessing": "The app cleans text, tokenizes Khmer words, removes punctuation and number-only tokens, and keeps meaningful Khmer tokens.",
        "Skip-gram with Negative Sampling": "The notebook trains skip-gram embeddings with dimension 50, context window +/-4, and 2 negative samples.",
        "PCA Visualization": "PCA compresses 50-dimensional embeddings into 2D for approximate visualization of word relationships.",
        "Neural Language Model": "The neural LM predicts the next word from the previous 5 words with a hidden layer of 512 units.",
        "Result Summary": "The cleaned run produced a meaningful vocabulary without punctuation tokens. The fixed skip-gram neural LM achieved the best perplexity in the saved comparison.",
        "Limitations": "The corpus is small and domain-specific. `MIN_FREQ = 10` maps many rare but meaningful words to `<UNK>`. Uploaded `.txt` files are used for EDA only unless retraining is added later.",
        "Future Work": "Future versions can retrain embeddings on uploaded corpora, save PCA coordinates, add better Khmer tokenizers, and improve the next-word prediction UI.",
    }

    for title, body in sections.items():
        with st.expander(title, expanded=True):
            st.write(body)


# -----------------------------
# Page 9: About & Deployment
# -----------------------------
elif page == "About & Deployment":
    st.title("About & Deployment")

    st.subheader("Project Folder Structure")
    st.code(
        """data/
notebook/
outputs/tables/
outputs/embeddings/
models/
streamlit_app/app.py
README.md
requirements.txt
.gitignore""",
        language="text",
    )

    st.subheader("Run Locally")
    st.code("streamlit run streamlit_app/app.py", language="bash")

    st.subheader("Deploy on Streamlit Community Cloud")
    st.write("1. Push the project to GitHub.")
    st.write("2. Go to Streamlit Community Cloud.")
    st.write("3. Connect your GitHub repository.")
    st.write("4. Select `streamlit_app/app.py` as the app file.")
    st.write("5. Deploy.")

    st.subheader("GitHub-ready Checklist")
    checklist = pd.DataFrame(
        [
            {"Item": "Clean notebook", "Status": "Ready"},
            {"Item": "Streamlit app", "Status": "Ready"},
            {"Item": "requirements.txt", "Status": "Ready"},
            {"Item": "Relative paths", "Status": "Ready"},
            {"Item": "Default corpus", "Status": "Ready if committed"},
            {"Item": "Saved outputs/models", "Status": "Recommended for full app features"},
        ]
    )
    st.dataframe(checklist, use_container_width=True)

    st.subheader("Files Needed for Deployment")
    st.write("At minimum: `streamlit_app/app.py`, `requirements.txt`, and `data/temples.txt`.")
    st.write("For full embedding, PCA, model comparison, and prediction features, also commit `outputs/` and `models/`.")
