import streamlit as st
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Dashboard de E-commerce & IA", layout="wide")

URL_BANCO = 'postgresql://postgres:senha_secreta@localhost:5432/postgres'
engine = create_engine(URL_BANCO)

@st.cache_data
def carregar_dados():
    query_metrics = '''
    SELECT 
        COUNT(DISTINCT p.id_pedido) as total_vendas,
        SUM(ip.quantidade * ip.preco_unitario) as faturamento_total
    FROM pedidos p
    JOIN itens_pedido ip ON p.id_pedido = ip.id_pedido
    WHERE p.status_pedido = 'Concluido';
    '''
    query_cat = '''
    SELECT c.nome_categoria, SUM(ip.quantidade * ip.preco_unitario) AS faturamento
    FROM itens_pedido ip
    JOIN produtos pr ON ip.id_produto = pr.id_produto
    JOIN categorias c ON pr.id_categoria = c.id_categoria
    JOIN pedidos p ON ip.id_pedido = p.id_pedido
    WHERE p.status_pedido = 'Concluido'
    GROUP BY c.nome_categoria;
    '''
    query_features = '''
    SELECT 
        c.id_cliente,
        c.nome,
        c.email,
        COUNT(DISTINCT p.id_pedido) AS total_pedidos,
        COALESCE(SUM(ip.quantidade * ip.preco_unitario), 0) AS total_gasto,
        COALESCE(AVG(ip.quantidade * ip.preco_unitario), 0) AS ticket_medio
    FROM clientes c
    LEFT JOIN pedidos p ON c.id_cliente = p.id_cliente AND p.status_pedido = 'Concluido'
    LEFT JOIN itens_pedido ip ON p.id_pedido = ip.id_pedido
    GROUP BY c.id_cliente, c.nome, c.email;
    '''
    metrics = pd.read_sql(query_metrics, con=engine)
    categories = pd.read_sql(query_cat, con=engine)
    features = pd.read_sql(query_features, con=engine)
    return metrics, categories, features

df_metrics, df_categories, df_features = carregar_dados()

st.title("📊 Painel Executivo & Predição de Churn com IA")
st.markdown("Indicadores comerciais e inteligência artificial para retenção de clientes.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    faturamento = df_metrics['faturamento_total'].iloc[0]
    st.metric(label="Faturamento Total Concluído", value=f"R$ {faturamento:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
with col2:
    st.metric(label="Total de Pedidos Convertidos", value=int(df_metrics['total_vendas'].iloc[0]))

st.divider()

col_esq, col_dir = st.columns([1.2, 1])

with col_esq:
    st.subheader("Faturamento por Categoria Comercial")
    st.bar_chart(data=df_categories, x='nome_categoria', y='faturamento')

with col_dir:
    st.subheader("Predição de Risco de Churn (IA)")
    st.markdown("Clique no botão abaixo para rodar o modelo preditivo sobre a nossa base de clientes ativa.")

    if st.button("Executar Análise Preditiva"):
        if os.path.exists('modelo_churn.pkl'):
            modelo = joblib.load('modelo_churn.pkl')
            X = df_features[['total_pedidos', 'total_gasto', 'ticket_medio']]
            df_features['previsao_churn'] = modelo.predict(X)

            clientes_em_risco = df_features[df_features['previsao_churn'] == 1][['id_cliente', 'nome', 'email']]

            if not clientes_em_risco.empty:
                st.warning(f"Atenção: A IA identificou {len(clientes_em_risco)} clientes com alto risco de abandonar a loja!")
                st.dataframe(clientes_em_risco, use_container_width=True)
            else:
                st.success("Excelente! A IA analisou a base e nenhum cliente foi classificado em risco crítico.")
        else:
            st.error("Arquivo do modelo ('modelo_churn.pkl') não foi encontrado. Execute o treino no Jupyter primeiro.")
