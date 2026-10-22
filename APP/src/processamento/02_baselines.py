"""
Módulo 02 — Baselines de Priorização Mensal de Municípios (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivo:
Avaliar as regras de priorização de referência (baselines) nos 24 meses alvo do conjunto de teste
(janeiro de 2024 a dezembro de 2025), definindo o patamar mínimo (benchmark) que os modelos
de Aprendizado de Máquina deverão superar.

Baselines avaliados:
1. persistencia: Ordena por focos_t decrescente (repetir o mês de emissão anterior).
2. historico_fixo: Total de focos observados de 2019 a 2023 na base por município (lista idêntica todo mês).
3. climatologia_focos: Ordena por focos_media_hist_mes decrescente.
4. rf_inpe: Ordena por base_rf_inpe_media_t decrescente (Risco de Fogo INPE do mês de emissão).
5. oraculo: Ordena pelo próprio alvo y decrescente (teto ótimo teórico).
Desempate em todas as regras: cd_mun crescente.

Métricas calculadas por mês alvo τ:
- Captura@k(τ) = soma de y no top-k ÷ soma de y dos 144 municípios (k = 10, 15, 20, 25, 30).
- Captura*@15(τ) = Captura@15(método) ÷ Captura@15(oráculo).
- Captura ponderada@15(τ) = soma de (y * pct_floresta / 100) no top-15 ÷ soma de (y * pct_floresta / 100) dos 144 municípios.
- Spearman entre o score/ranking do método e y.
- Rotatividade: fração dos municípios do top-15 que não estavam no top-15 do mês alvo anterior.

Saídas:
- data/processed/resultados/baselines_mensal.csv
- data/processed/resultados/baselines_resumo.csv
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import (
    avaliar_metodo_mensal,
    adicionar_captura_star_15,
    gerar_tabela_resumo
)

logger = setup_logger("processamento_02_baselines")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def executar_baselines():
    logger.info("Iniciando Módulo 02: Baselines de Priorização Mensal...")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    mod_path = processed_dir / "aphim_modelagem.parquet"
    base_path = processed_dir / "aphim_base_municipio_mes.parquet"

    if not mod_path.exists():
        raise FileNotFoundError(f"Base de modelagem '{mod_path}' não encontrada. Execute 01_atributos_alvo.py primeiro.")
    if not base_path.exists():
        raise FileNotFoundError(f"Base de municípios '{base_path}' não encontrada.")

    df_mod = pd.read_parquet(mod_path)
    df_base = pd.read_parquet(base_path)

    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_base["cd_mun"] = df_base["cd_mun"].astype(str).str.zfill(7)

    # 1. Pré-computar histórico fixo (início do treino até 2023, estritamente pré-teste)
    ano_treino_inicio = int(config.get("particao", {}).get("treino", ["2019-01", "2023-11"])[0][:4])
    logger.info(f"Calculando total de focos históricos {ano_treino_inicio}-2023 por município para o baseline historico_fixo...")
    df_treino_hist = df_base[df_base["ano"].between(ano_treino_inicio, 2023)]
    focos_hist_fixo = df_treino_hist.groupby("cd_mun")["focos_count"].sum().to_dict()


    df_mod["score_hist_fixo"] = df_mod["cd_mun"].map(focos_hist_fixo)

    # 2. Definição dos baselines e suas respectivas colunas de ordenação
    baselines = {
        "persistencia": "focos_t",
        "historico_fixo": "score_hist_fixo",
        "climatologia_focos": "focos_media_hist_mes",
        "rf_inpe": "base_rf_inpe_media_t",
        "oraculo": "y"
    }

    # 3. Extrair linhas de teste (24 meses alvo: 2024-01 a 2025-12)
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    meses_alvo_teste = sorted(df_teste["mes_alvo"].unique())
    logger.info(f"Meses alvo do conjunto de teste ({len(meses_alvo_teste)} meses): {meses_alvo_teste[0]} a {meses_alvo_teste[-1]}")

    if len(meses_alvo_teste) != 24:
        raise ValueError(f"Esperado 24 meses alvo de teste, encontrado {len(meses_alvo_teste)}.")

    # Obter também a partição imediatamente anterior (mes_alvo = '2023-12', gerada por mes_emissao = '2023-11')
    # para permitir o cálculo contínuo da rotatividade no primeiro mês de teste (2024-01)
    df_prev_baseline = df_mod[df_mod["mes_alvo"] == "2023-12"].copy()

    # 4. Cálculo mensal das métricas por baseline utilizando módulo compartilhado
    logger.info("Executando simulação de rankings e cálculo das métricas de priorização via metricas_priorizacao...")
    dfs_metodos = []
    for nome_baseline, col_score in baselines.items():
        df_m = avaliar_metodo_mensal(
            df_periodo=df_teste,
            col_score=col_score,
            nome_metodo=nome_baseline,
            df_ancora=df_prev_baseline
        )
        dfs_metodos.append(df_m)

    df_mensal = pd.concat(dfs_metodos, ignore_index=True)
    df_mensal = df_mensal.rename(columns={"metodo": "baseline"})
    df_mensal = adicionar_captura_star_15(df_mensal, col_metodo="baseline", nome_oraculo="oraculo")

    colunas_mensal = [
        "baseline", "mes_alvo",
        "captura_10", "captura_15", "captura_20", "captura_25", "captura_30",
        "captura_star_15", "captura_pond_15",
        "spearman", "rotatividade"
    ]
    df_mensal = df_mensal[colunas_mensal].sort_values(["baseline", "mes_alvo"]).reset_index(drop=True)

    # 5. Tabela Resumo (Média dos 24 meses + Estação Seca)
    logger.info("Consolidando tabela resumo com médias de 24 meses e estação seca...")
    meses_seca = [m for m in meses_alvo_teste if m[5:7] in ["07", "08", "09", "10", "11"]]
    ordem_baselines = ["oraculo", "climatologia_focos", "persistencia", "historico_fixo", "rf_inpe"]
    df_resumo = gerar_tabela_resumo(
        df_mensal=df_mensal,
        col_metodo="baseline",
        meses_seca=meses_seca,
        ordem_metodos=ordem_baselines
    )

    # 7. Exportar saídas
    out_mensal = resultados_dir / "baselines_mensal.csv"
    out_resumo = resultados_dir / "baselines_resumo.csv"

    logger.info(f"Salvando resultados mensais em '{out_mensal}'...")
    df_mensal.to_csv(out_mensal, index=False, encoding="utf-8")

    logger.info(f"Salvando resumo dos baselines em '{out_resumo}'...")
    df_resumo.to_csv(out_resumo, index=False, encoding="utf-8")

    # 8. Imprimir tabela-resumo e curva de Captura@k no terminal
    imprimir_relatorio_terminal(df_resumo)

    logger.info("Módulo 02 finalizado com sucesso!")
    return df_mensal, df_resumo

def imprimir_relatorio_terminal(df_resumo: pd.DataFrame):
    """Exibe no terminal a tabela comparativa e a curva de captura dos baselines."""
    print("\n" + "=" * 95)
    print("                      TABELA-RESUMO DOS BASELINES DE PRIORIZAÇÃO (24 MESES)")
    print("=" * 95)
    header = f"{'Baseline':<20} | {'Captura@15':<10} | {'Captura*@15':<12} | {'Cap.Pond@15':<11} | {'Spearman':<9} | {'Rotativ.':<9} | {'Cap@15 Seca':<11}"
    print(header)
    print("-" * 95)
    for _, r in df_resumo.iterrows():
        b_name = r["baseline"]
        c15 = f"{r['captura_15']*100:.2f}%"
        c_star = f"{r['captura_star_15']*100:.2f}%"
        c_pond = f"{r['captura_pond_15']*100:.2f}%"
        sp = f"{r['spearman']:.4f}"
        rot = f"{r['rotatividade']*100:.2f}%"
        c_seca = f"{r['captura_15_seca']*100:.2f}%"
        print(f"{b_name:<20} | {c15:<10} | {c_star:<12} | {c_pond:<11} | {sp:<9} | {rot:<9} | {c_seca:<11}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("                   CURVA DE CAPTURA@k (k = 10, 15, 20, 25, 30 MUNICÍPIOS)")
    print("=" * 95)
    header_k = f"{'Baseline':<20} | {'Captura@10':<12} | {'Captura@15':<12} | {'Captura@20':<12} | {'Captura@25':<12} | {'Captura@30':<12}"
    print(header_k)
    print("-" * 95)
    for _, r in df_resumo.iterrows():
        b_name = r["baseline"]
        c10 = f"{r['captura_10']*100:.2f}%"
        c15 = f"{r['captura_15']*100:.2f}%"
        c20 = f"{r['captura_20']*100:.2f}%"
        c25 = f"{r['captura_25']*100:.2f}%"
        c30 = f"{r['captura_30']*100:.2f}%"
        print(f"{b_name:<20} | {c10:<12} | {c15:<12} | {c20:<12} | {c25:<12} | {c30:<12}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    executar_baselines()
