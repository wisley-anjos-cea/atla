import os
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    excel_path = os.path.join(BASE_DIR, "dados_exemplo_dashboard.xlsx")
    
    try:
        df = pd.read_excel(excel_path)
        meses = ["Jan", "Fev", "Mar"]
        df["Total"] = df[meses].sum(axis=1)

        # 1. Totais por Categoria (Ex: Hardware, Acessórios)
        cat_totals = df.groupby("Categoria")[meses + ["Total"]].sum().to_dict(orient="index")

        # 2. Totais por Subcategoria (Ex: Notebooks, Mouses)
        sub_totals = df.groupby("Subcategoria")[meses + ["Total"]].sum().to_dict(orient="index")

        # 3. Dados dos Produtos individuais
        prod_details = df.set_index("Produto")[meses + ["Total"]].to_dict(orient="index")

        # Unimos tudo em um único dicionário para o HTML
        dados = {**cat_totals, **sub_totals, **prod_details}
        
        return render_template("index.html", dados=dados)
    
    except Exception as e:
        return f"Erro: {e}"

if __name__ == "__main__":
    app.run(debug=True)