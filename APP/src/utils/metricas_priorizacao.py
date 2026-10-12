"""
Módulo Utilitário — Métricas de Priorização e Avaliação de Ranking (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Métricas implementadas:
1. Captura@k(τ): Fração dos focos totais concentrada nos k primeiros municípios prioritários.
2. Captura*@15(τ): Eficiência relativa do método em relação ao oráculo teórico (Captura@15 / Captura@15_oraculo).
3. Captura ponderada@15(τ): Captura ponderada pela cobertura florestal (y * pct_floresta / 100).
4. Spearman(τ): Coeficiente de correlação de postos de Spearman entre o score/ranking e o alvo real y.
5. Rotatividade(τ): Fração de novos municípios no top-15 em relação ao mês anterior (|Top15_t \ Top15_{t-1}| / 15).
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def calcular_metricas_mes(
    sub_df: pd.DataFrame,
    col_score: str,
    col_target: str = "y",
    col_weight: str = "pct_floresta",
    prev_top15: Optional[Set[str]] = None,
    k_valores: Tuple[int, ...] = (10, 15, 20, 25, 30)
) -> Tuple[Dict[str, float], Set[str]]:
    """
    Calcula as métricas de priorização para um único mês alvo τ entre os 144 municípios.
    Regra de desempate: col_score decrescente, cd_mun crescente.
    """
    if len(sub_df) != 144:
        raise ValueError(f"Esperado 144 municípios para o mês alvo, obtido {len(sub_df)}.")

    total_y = float(sub_df[col_target].sum())
    total_y_pond = float((sub_df[col_target] * sub_df[col_weight] / 100.0).sum())

    # Ordenação estrita do ranking
    sub_sorted = sub_df.sort_values(
        by=[col_score, "cd_mun"],
        ascending=[False, True]
    ).reset_index(drop=True)

    top15_muns = set(sub_sorted.iloc[:15]["cd_mun"])

    # 1. Rotatividade em relação ao mês anterior
    if prev_top15 is not None:
        novos = len(top15_muns - prev_top15)
        rotatividade = novos / 15.0
    else:
        rotatividade = np.nan

    # 2. Captura@k
    metrics: Dict[str, float] = {}
    for k in k_valores:
        soma_k = sub_sorted.iloc[:k][col_target].sum()
        metrics[f"captura_{k}"] = float(soma_k / total_y) if total_y > 0 else 0.0

    # 3. Captura ponderada@15
    soma_pond_15 = (sub_sorted.iloc[:15][col_target] * sub_sorted.iloc[:15][col_weight] / 100.0).sum()
    metrics["captura_pond_15"] = float(soma_pond_15 / total_y_pond) if total_y_pond > 0 else 0.0

    # 4. Correlação de postos de Spearman
    sp_stat = spearmanr(sub_sorted[col_score], sub_sorted[col_target]).statistic
    if np.isnan(sp_stat):
        sp_stat = 0.0
    metrics["spearman"] = float(sp_stat)
    metrics["rotatividade"] = float(rotatividade)

    return metrics, top15_muns

def avaliar_metodo_mensal(
    df_periodo: pd.DataFrame,
    col_score: str,
    nome_metodo: str,
    df_ancora: Optional[pd.DataFrame] = None,
    col_target: str = "y",
    col_weight: str = "pct_floresta",
    k_valores: Tuple[int, ...] = (10, 15, 20, 25, 30)
) -> pd.DataFrame:
    """
    Avalia um método/baseline ao longo de uma série de meses alvo.
    Caso df_ancora seja fornecido (ex: 2023-12), calcula o top-15 inicial para ancorar a rotatividade.
    """
    meses_alvo = sorted(df_periodo["mes_alvo"].unique())

    # Ancoragem do mês anterior (ex: 2023-12)
    prev_top15: Optional[Set[str]] = None
    if df_ancora is not None and len(df_ancora) == 144:
        ancora_sorted = df_ancora.sort_values(
            by=[col_score, "cd_mun"],
            ascending=[False, True]
        )
        prev_top15 = set(ancora_sorted.iloc[:15]["cd_mun"])

    registros = []
    for mes in meses_alvo:
        sub = df_periodo[df_periodo["mes_alvo"] == mes].copy()
        m_dict, top15_curr = calcular_metricas_mes(
            sub_df=sub,
            col_score=col_score,
            col_target=col_target,
            col_weight=col_weight,
            prev_top15=prev_top15,
            k_valores=k_valores
        )
        prev_top15 = top15_curr

        reg = {
            "metodo": nome_metodo,
            "mes_alvo": mes,
            **m_dict
        }
        registros.append(reg)

    return pd.DataFrame(registros)

def adicionar_captura_star_15(
    df_mensal: pd.DataFrame,
    col_metodo: str = "metodo",
    nome_oraculo: str = "oraculo"
) -> pd.DataFrame:
    """
    Calcula Captura*@15 = Captura@15(metodo) / Captura@15(oraculo) para cada mês alvo.
    """
    df_out = df_mensal.copy()
    sub_oraculo = df_out[df_out[col_metodo] == nome_oraculo]
    if sub_oraculo.empty:
        raise ValueError(f"Oráculo ('{nome_oraculo}') não encontrado em df_mensal para cálculo de Captura*@15.")

    oraculo_caps = sub_oraculo.set_index("mes_alvo")["captura_15"].to_dict()

    df_out["captura_star_15"] = df_out.apply(
        lambda r: float(r["captura_15"] / oraculo_caps[r["mes_alvo"]]) if oraculo_caps.get(r["mes_alvo"], 0) > 0 else 0.0,
        axis=1
    )
    return df_out

def gerar_tabela_resumo(
    df_mensal: pd.DataFrame,
    col_metodo: str = "metodo",
    meses_seca: Optional[List[str]] = None,
    ordem_metodos: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Gera a tabela-resumo agregando as médias temporais de todas as métricas,
    incluindo a média de Captura@15 na estação seca.
    """
    cols_medias = [
        "captura_10", "captura_15", "captura_20", "captura_25", "captura_30",
        "captura_star_15", "captura_pond_15",
        "spearman", "rotatividade"
    ]
    # Manter apenas colunas presentes
    cols_existentes = [c for c in cols_medias if c in df_mensal.columns]

    df_resumo = df_mensal.groupby(col_metodo)[cols_existentes].mean().reset_index()

    if meses_seca is not None and len(meses_seca) > 0:
        df_seca = df_mensal[df_mensal["mes_alvo"].isin(meses_seca)]
        resumo_seca = df_seca.groupby(col_metodo)["captura_15"].mean().rename("captura_15_seca")
        df_resumo = df_resumo.merge(resumo_seca, on=col_metodo, how="left")

    if ordem_metodos is not None:
        df_resumo["_ordem"] = df_resumo[col_metodo].map(
            lambda m: ordem_metodos.index(m) if m in ordem_metodos else 999
        )
        df_resumo = df_resumo.sort_values("_ordem").drop(columns=["_ordem"]).reset_index(drop=True)

    return df_resumo
