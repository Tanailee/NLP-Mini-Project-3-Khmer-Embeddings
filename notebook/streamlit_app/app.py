
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Khmer Temple NLP Explorer', layout='wide')
st.title('Khmer Temple NLP Explorer')
st.write('Interactive demo for Khmer word embeddings and neural language modeling.')

page = st.sidebar.selectbox('Choose Page', ['Corpus Overview', 'EDA Dashboard', 'PCA Map', 'Model Comparison', 'About Project'])

if page == 'Corpus Overview':
    st.header('Corpus Overview')
    text_path = Path('data/temples.txt')
    if text_path.exists():
        text = text_path.read_text(encoding='utf-8')
        st.metric('Characters', len(text))
        st.text_area('Sample Text', text[:1000], height=250)
    else:
        st.warning('Please place temples.txt inside the data folder.')

elif page == 'EDA Dashboard':
    st.header('EDA Dashboard')
    vocab_path = Path('outputs/tables/raw_vocabulary_table.csv')
    if vocab_path.exists():
        vocab_df = pd.read_csv(vocab_path)
        top_words = vocab_df.sort_values('frequency', ascending=False).head(30)
        fig = px.bar(top_words, x='token', y='frequency', title='Top 30 Khmer Tokens')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Run the notebook first.')

elif page == 'PCA Map':
    st.header('PCA Embedding Map')
    pca_path = Path('outputs/tables/skipgram_pca.csv')
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path)
        fig = px.scatter(pca_df, x='PC1', y='PC2', size='frequency', hover_data=['word'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Run the notebook first.')

elif page == 'Model Comparison':
    st.header('Model Comparison')
    comparison_path = Path('outputs/tables/model_comparison.csv')
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df)
    else:
        st.warning('Run the notebook first.')

else:
    st.header('About Project')
    st.write('This project applies Khmer preprocessing, word embeddings, PCA, and language modeling.')
