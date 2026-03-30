import pyodbc
import pandas as pd
from prophet import Prophet
import holidays
from datetime import datetime

# ==============================
# LOG DE EXECUÇÃO / EXPLICAÇÃO
# ==============================
log_execucao = []
log_execucao.append("INÍCIO DO PROCESSO DE PREVISÃO DE VENDAS")
log_execucao.append("Objetivo: gerar previsão diária de vendas para o ano de 2026 por categoria.")

log_execucao.append(
    "\nREGRAS DE CÁLCULO UTILIZADAS NO MODELO:\n"
    "- Venda diária calculada como a soma do SALEAMT por data e categoria.\n"
    "- Granularidade diária obrigatória: dias sem venda são tratados como valor zero.\n"
    "- Categorias com menos de 10 dias de histórico são desconsideradas.\n"
    "- Outliers tratados via corte no percentil 98.\n"
    "- Período pós-salário definido como dias 1 a 10.\n"
    "- Sazonalidade semanal e anual em modo multiplicativo.\n"
    "- Tendência linear com penalização de mudanças bruscas.\n"
    "- Impacto automático de feriados nacionais.\n"
    "- Dias 25/12 e 01/01 tratados como DIAS FECHADOS (venda = 0 no histórico e na previsão).\n"
    "- Resultados finais arredondados para valores inteiros."
)

# ==============================
# 1. CONEXÃO E QUERY
# ==============================
conn = pyodbc.connect("DSN=Teraprod;UID=BRWAA1;PWD=Henri@2024")

query = """
SELECT
    D.CALENDARDAYDT AS DATA,
    MNLOCCD,
    PrchDivDesc,
    CAST(SUM(SALEAMT) AS INT) AS VL_TOTAL
FROM BR_PRD_VIEW.RTL_FT_SALE_TICKET_DIL DA 
INNER JOIN BR_PRD_VIEW.RTL_DM_LOCATION LO ON (DA.LOCID = LO.LOCID)
INNER JOIN BR_PRD_VIEW.RTL_DM_ITEM IT ON (IT.ITEMIDSK = DA.ITEMIDSK)
LEFT JOIN BR_PRD_VIEW.RTL_DAY D ON (DA.CALDAYDT = D.CALENDARDAYDT)
LEFT JOIN BR_PRD_VIEW.RTL_CEA_WEEK W ON (D.CEAWEEKID = W.CEAWEEKID)
WHERE D.CEAWEEKID BETWEEN '202401' AND '202552'
AND DA.LOCID = 160
AND PrchDivID IN (1,3)
GROUP BY D.CALENDARDAYDT, MNLOCCD, PrchDivDesc
ORDER BY DATA
"""

df = pd.read_sql(query, conn)
df['DATA'] = pd.to_datetime(df['DATA'])

log_execucao.append("1) Dados extraídos do banco Teraprod (2024–2025, loja 160, categorias 1 e 3).")

# ==============================
# 2. CONFIGURAÇÕES GLOBAIS
# ==============================
categorias = df['PrchDivDesc'].unique()
resultado_final = pd.DataFrame()

log_execucao.append(f"2) Identificadas {len(categorias)} categorias com histórico.")

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def is_payday_period(ds):
    return 1 if 1 <= ds.day <= 10 else 0

def is_closed_day(ds):
    return 1 if (ds.month == 12 and ds.day == 25) or (ds.month == 1 and ds.day == 1) else 0

log_execucao.append(
    "3) Criadas variáveis explicativas: efeito salário (dias 1 a 10) "
    "e indicador de dias fechados (25/12 e 01/01)."
)
# ==============================
# 3. LOOP POR CATEGORIA
# ==============================
for cat in categorias:
    log_execucao.append(f"\nProcessando categoria: {cat}")

    df_cat = df[df['PrchDivDesc'] == cat].copy()
    df_group = df_cat.groupby('DATA')['VL_TOTAL'].sum().reset_index()
    df_group.columns = ['ds', 'y']

    if len(df_group) < 10:
        log_execucao.append(
            f"Categoria {cat} ignorada por histórico insuficiente ({len(df_group)} dias)."
        )
        continue

    # Preencher lacunas de dias
    df_group = df_group.set_index('ds').asfreq('D').fillna(0).reset_index()

    # Tratamento de outliers
    limite = df_group['y'].quantile(0.98)
    df_group['y'] = df_group['y'].clip(upper=limite)

    # Regressoras
    df_group['is_payday'] = df_group['ds'].apply(is_payday_period)
    df_group['is_closed'] = df_group['ds'].apply(is_closed_day)

    # Forçar venda zero em dias fechados (histórico)
    df_group.loc[df_group['is_closed'] == 1, 'y'] = 0

    log_execucao.append(
        f"Categoria {cat}: histórico diário consolidado, outliers tratados "
        "e vendas zeradas em dias fechados."
    )

    # ==============================
    # 4. MODELAGEM
    # ==============================
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    model.add_country_holidays(country_name='BR')
    model.add_regressor('is_payday')
    model.add_regressor('is_closed')

    model.fit(df_group)

    log_execucao.append(f"Modelo treinado com sucesso para a categoria {cat}.")

    # ==============================
    # 5. FORECAST 2026
    # ==============================
    ultima_data = df_group['ds'].max()
    dias_ate_fim_2026 = (datetime(2026, 12, 31) - ultima_data).days

    future = model.make_future_dataframe(periods=dias_ate_fim_2026)
    future['is_payday'] = future['ds'].apply(is_payday_period)
    future['is_closed'] = future['ds'].apply(is_closed_day)

    forecast = model.predict(future)

    # Forçar venda zero na previsão em dias fechados
    forecast.loc[
        forecast['is_closed'] == 1,
        ['yhat', 'yhat_lower', 'yhat_upper']
    ] = 0

    forecast_2026 = forecast[
        (forecast['ds'] >= '2026-01-01') &
        (forecast['ds'] <= '2026-12-31')
    ].copy()

    forecast_2026['categoria'] = cat

    resultado_final = pd.concat([
        resultado_final,
        forecast_2026[['ds', 'categoria', 'yhat', 'yhat_lower', 'yhat_upper']]
    ])

    log_execucao.append(
        f"Previsão 2026 concluída para a categoria {cat}."
    )

# ==============================
# 6. EXPORTAÇÃO
# ==============================
if not resultado_final.empty:
    resultado_final[['yhat', 'yhat_lower', 'yhat_upper']] = (
        resultado_final[['yhat', 'yhat_lower', 'yhat_upper']]
        .round(0).astype(int)
    )

    nome_arquivo = "previsao_vendas_SCN_2026.xlsx"
    resultado_final.to_excel(nome_arquivo, index=False)

    log_execucao.append(
        f"\nArquivo '{nome_arquivo}' gerado com {len(resultado_final)} linhas."
    )
else:
    log_execucao.append("\nNenhuma categoria gerou previsão válida.")

# ==============================
# 7. RELATÓRIO EXPLICATIVO
# ==============================
print("\n==============================")
print("RELATÓRIO EXPLICATIVO DA EXECUÇÃO")
print("==============================\n")

for linha in log_execucao:
    print(linha)

with open("explicacao_execucao_previsao_2026_1loja.txt", "w", encoding="utf-8") as f:
    for linha in log_execucao:
        f.write(linha + "\n")