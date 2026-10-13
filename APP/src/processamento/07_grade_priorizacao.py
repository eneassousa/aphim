"""
Módulo 07 — Priorização Espacial em Nível 2 (Grade) (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Objetivo:
Implementar a priorização em dois níveis:
- Nível 1: ranking municipal dos 144 municípios (modelo preditivo vencedor de perigo).
- Nível 2: dentro dos 15 municípios do topo em cada mês, identificar os quadrados da grade
  que concentram o fogo do mês seguinte ("dentro de Altamira, para onde mandar a brigada?").

Modelagem e Avaliação:
- Modelo de Célula: XGBoost com objective="count:poisson", semente 42, treinado em todas as células.
- Validação temporal em 2 dobras (2022 e 2023) nos 15 municípios com mais focos observados em t.
- Baselines: persistencia_celula, climatologia_celula, historico_fixo_celula (2019-2023) e oraculo_celula.
- Avaliação nos 24 meses de teste (2024-2025): CapturaÁrea@5%, @10%, @20%, Captura Ponta a Ponta,
  Área total coberta em km² e % do Pará, médias mensais, agregadas e na estação seca.
- Testes estatísticos pareados (Wilcoxon + Bootstrap 95% + Holm-Bonferroni).
- Mapa cartográfico de exemplo para Altamira em 2024-09.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import wilcoxon
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyproj import Transformer

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_07_grade_priorizacao")

FEATURES_CELULA = [
    "focos_t",
    "focos_t1",
    "focos_t2",
    "focos_mm3",
    "focos_mesmo_mes_ano_ant",
    "focos_media_hist_mes",
    "focos_vizinhos_t",
    "focos_vizinhos_media_hist_mes",
    "mes_sin",
    "mes_cos",
    "focos_mun_t",
    "prec_mm",
    "ur_pct",
    "dpv_kpa",
    "umid_solo",
    "anom_prec_mm"
]

def carregar_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração '{config_path}' não encontrado.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    adj_pvals = np.empty(m)
    for i in range(m):
        adj_pvals[i] = (m - i) * sorted_pvals[i]
    for i in range(1, m):
        if adj_pvals[i] < adj_pvals[i - 1]:
            adj_pvals[i] = adj_pvals[i - 1]
    adj_pvals = np.clip(adj_pvals, 0.0, 1.0)
    original_pvals = np.empty(m)
    original_pvals[sorted_indices] = adj_pvals
    return original_pvals

def calcular_captura_area_mes(
    df_mes: pd.DataFrame,
    col_score: str,
    top15_muns: List[str],
    cell_area_map: Dict[int, float],
    pct_area_k: float = 10.0
) -> Tuple[float, float, float]:
    """
    Calcula a CapturaÁrea@k% para um único mês dentro dos municípios de top15.
    Retorna: (captura_area_k, focos_capturados, focos_top15).
    """
    sub_top15 = df_mes[df_mes["cd_mun"].isin(top15_muns)]
    focos_capturados = 0.0
    focos_top15 = 0.0

    for cd_m, grp in sub_top15.groupby("cd_mun"):
        focos_mun = grp["y"].sum()
        focos_top15 += focos_mun

        # Ordenar células decrescente pelo score com desempate determinístico por cell_id
        grp_sort = grp.sort_values(by=[col_score, "cell_id"], ascending=[False, True])
        cell_areas = grp_sort["area_km2"].values
        cum_areas = np.cumsum(cell_areas)
        total_mun_area = cum_areas[-1] if len(cum_areas) > 0 else 0.0
        target_area = (pct_area_k / 100.0) * total_mun_area

        # Células que somam os primeiros k% da área do município
        mask = (cum_areas - cell_areas) < target_area
        focos_capturados += grp_sort.iloc[mask]["y"].sum()

    cap = (focos_capturados / focos_top15) if focos_top15 > 0 else 0.0
    return float(cap), float(focos_capturados), float(focos_top15)

def validar_hiperparametros(
    df_treino: pd.DataFrame,
    cell_area_map: Dict[int, float]
) -> Tuple[Dict[str, any], float, List[Dict[str, any]]]:
    """
    Executa busca em grade pequena de hiperparâmetros por validação temporal nas dobras 2022 e 2023.
    Critério: CapturaÁrea@10% média nos 15 municípios com mais focos observados em t.
    Retorna: (melhor_params, melhor_score, todas_combinacoes).
    """
    logger.info("Iniciando sintonia de hiperparâmetros por validação temporal (dobras 2022 e 2023)...")

    df_tr_22 = df_treino[df_treino["mes_alvo"] < "2022-01"].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()

    df_tr_23 = df_treino[df_treino["mes_alvo"] < "2023-01"].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    grid_params = []
    for md in [4, 6]:
        for lr in [0.05, 0.1]:
            for ne in [300, 600]:
                for mcw in [1, 10]:
                    grid_params.append({
                        "max_depth": md,
                        "learning_rate": lr,
                        "n_estimators": ne,
                        "min_child_weight": mcw
                    })

    logger.info(f"Testando {len(grid_params)} combinações de hiperparâmetros...")

    # Identificar top 15 municípios por focos observados em t para cada mês de validação
    def get_top15_obs_emissao(df_subset: pd.DataFrame) -> Dict[str, List[str]]:
        top15_dict = {}
        for m_alvo, grp in df_subset.groupby("mes_alvo"):
            # Focos observados do município em t (agrupados por cd_mun)
            mun_focos_t = grp.groupby("cd_mun")["focos_mun_t"].first()
            top15_muns = mun_focos_t.sort_values(ascending=False).head(15).index.tolist()
            top15_dict[m_alvo] = top15_muns
        return top15_dict

    top15_val_22 = get_top15_obs_emissao(df_val_22)
    top15_val_23 = get_top15_obs_emissao(df_val_23)

    melhor_score = -1.0
    melhor_params = None
    todas_combinacoes = []

    for i, p in enumerate(grid_params):
        t0 = time.time()
        # 1. Dobra 2022
        m22 = xgb.XGBRegressor(
            objective="count:poisson",
            tree_method="hist",
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            min_child_weight=p["min_child_weight"],
            random_state=42,
            n_jobs=-1
        )
        m22.fit(df_tr_22[FEATURES_CELULA], df_tr_22["y"])
        df_val_22["pred_val"] = m22.predict(df_val_22[FEATURES_CELULA])

        caps_22 = []
        for m_alvo, grp in df_val_22.groupby("mes_alvo"):
            cap, _, _ = calcular_captura_area_mes(grp, "pred_val", top15_val_22[m_alvo], cell_area_map, 10.0)
            caps_22.append(cap)
        score_22 = float(np.mean(caps_22))

        # 2. Dobra 2023
        m23 = xgb.XGBRegressor(
            objective="count:poisson",
            tree_method="hist",
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            min_child_weight=p["min_child_weight"],
            random_state=42,
            n_jobs=-1
        )
        m23.fit(df_tr_23[FEATURES_CELULA], df_tr_23["y"])
        df_val_23["pred_val"] = m23.predict(df_val_23[FEATURES_CELULA])

        caps_23 = []
        for m_alvo, grp in df_val_23.groupby("mes_alvo"):
            cap, _, _ = calcular_captura_area_mes(grp, "pred_val", top15_val_23[m_alvo], cell_area_map, 10.0)
            caps_23.append(cap)
        score_23 = float(np.mean(caps_23))

        score_medio = float((score_22 + score_23) / 2.0)
        elapsed = time.time() - t0

        todas_combinacoes.append({
            "parametros": p,
            "score_2022": score_22,
            "score_2023": score_23,
            "score_medio": score_medio,
            "tempo_s": round(elapsed, 2)
        })

        logger.info(
            f"Combinação {i+1:02d}/{len(grid_params)}: {p} -> "
            f"2022: {score_22*100:.2f}% | 2023: {score_23*100:.2f}% | Média: {score_medio*100:.2f}% ({elapsed:.1f}s)"
        )

        if score_medio > melhor_score:
            melhor_score = score_medio
            melhor_params = p

    logger.info(f"Melhor hiperparâmetro selecionado: {melhor_params} com CapturaÁrea@10% média = {melhor_score*100:.2f}%.")
    return melhor_params, melhor_score, todas_combinacoes

def avaliar_metodos_teste(
    df_teste: pd.DataFrame,
    df_pred_l1: pd.DataFrame,
    cell_area_map: Dict[int, float],
    area_total_pa: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa a avaliação nos 24 meses de teste para os métodos:
    xgboost, persistencia_celula, climatologia_celula, historico_fixo_celula, oraculo_celula.
    """
    logger.info("Iniciando avaliação dos rankings espaciais nos 24 meses de teste (2024-2025)...")

    metodos = [
        ("xgboost", "pred_xgb"),
        ("persistencia_celula", "focos_t"),
        ("climatologia_celula", "focos_media_hist_mes"),
        ("historico_fixo_celula", "focos_fixo_2019_2023"),
        ("oraculo_celula", "y")
    ]

    meses_alvo = sorted(df_teste["mes_alvo"].unique())
    meses_seca = [m for m in meses_alvo if m[5:7] in ["07", "08", "09", "10", "11"]]

    registros_mensais = []

    for m_alvo in meses_alvo:
        sub_l1 = df_pred_l1[df_pred_l1["mes_alvo"] == m_alvo].sort_values(
            by=["pred_vencedor", "cd_mun"], ascending=[False, True]
        )
        top15_muns = sub_l1["cd_mun"].head(15).tolist()

        df_m = df_teste[df_teste["mes_alvo"] == m_alvo].copy()
        focos_total_pa = float(df_m["y"].sum())

        for nome_met, col_sc in metodos:
            sub_top15 = df_m[df_m["cd_mun"].isin(top15_muns)]
            focos_top15_tot = 0.0
            focos_cap_5 = 0.0
            focos_cap_10 = 0.0
            focos_cap_20 = 0.0
            area_cob_10 = 0.0

            for cd_m, grp in sub_top15.groupby("cd_mun"):
                f_mun = grp["y"].sum()
                focos_top15_tot += f_mun

                grp_s = grp.sort_values(by=[col_sc, "cell_id"], ascending=[False, True])
                c_areas = grp_s["area_km2"].values
                cum_a = np.cumsum(c_areas)
                tot_a = cum_a[-1] if len(cum_a) > 0 else 0.0

                # 5%
                m_5 = (cum_a - c_areas) < (0.05 * tot_a)
                focos_cap_5 += grp_s.iloc[m_5]["y"].sum()

                # 10%
                m_10 = (cum_a - c_areas) < (0.10 * tot_a)
                sel_10 = grp_s.iloc[m_10]
                focos_cap_10 += sel_10["y"].sum()
                area_cob_10 += sel_10["area_km2"].sum()

                # 20%
                m_20 = (cum_a - c_areas) < (0.20 * tot_a)
                focos_cap_20 += grp_s.iloc[m_20]["y"].sum()

            cap_area_5 = (focos_cap_5 / focos_top15_tot) if focos_top15_tot > 0 else 0.0
            cap_area_10 = (focos_cap_10 / focos_top15_tot) if focos_top15_tot > 0 else 0.0
            cap_area_20 = (focos_cap_20 / focos_top15_tot) if focos_top15_tot > 0 else 0.0

            cap_ponta_ponta = (focos_cap_10 / focos_total_pa) if focos_total_pa > 0 else 0.0
            pct_area_pa = (area_cob_10 / area_total_pa) * 100.0

            registros_mensais.append({
                "mes_alvo": m_alvo,
                "metodo": nome_met,
                "captura_area_5": cap_area_5,
                "captura_area_10": cap_area_10,
                "captura_area_20": cap_area_20,
                "captura_ponta_ponta_10": cap_ponta_ponta,
                "area_coberta_km2": area_cob_10,
                "pct_area_pa": pct_area_pa,
                "focos_capturados_5": focos_cap_5,
                "focos_capturados_10": focos_cap_10,
                "focos_capturados_20": focos_cap_20,
                "focos_top15": focos_top15_tot,
                "focos_pa": focos_total_pa
            })

    df_mensal = pd.DataFrame(registros_mensais)

    # 2. Resumo com Médias Mensais, Agregadas e Estação Seca
    logger.info("Consolidando tabela resumo com métricas médias, agregadas e na estação seca...")
    linhas_resumo = []

    for nome_met, _ in metodos:
        sub = df_mensal[df_mensal["metodo"] == nome_met]
        sub_seca = sub[sub["mes_alvo"].isin(meses_seca)]

        # Médias mensais
        m_cap5 = sub["captura_area_5"].mean()
        m_cap10 = sub["captura_area_10"].mean()
        m_cap20 = sub["captura_area_20"].mean()
        m_pp = sub["captura_ponta_ponta_10"].mean()
        m_area_km2 = sub["area_coberta_km2"].mean()
        m_pct_area = sub["pct_area_pa"].mean()

        # Agregadas (soma capturados / soma focos)
        agreg_cap5 = sub["focos_capturados_5"].sum() / sub["focos_top15"].sum()
        agreg_cap10 = sub["focos_capturados_10"].sum() / sub["focos_top15"].sum()
        agreg_cap20 = sub["focos_capturados_20"].sum() / sub["focos_top15"].sum()
        agreg_pp = sub["focos_capturados_10"].sum() / sub["focos_pa"].sum()

        # Estação seca
        m_cap10_seca = sub_seca["captura_area_10"].mean()
        m_pp_seca = sub_seca["captura_ponta_ponta_10"].mean()

        linhas_resumo.append({
            "metodo": nome_met,
            "cap_area_5_media": m_cap5,
            "cap_area_10_media": m_cap10,
            "cap_area_20_media": m_cap20,
            "cap_ponta_ponta_media": m_pp,
            "cap_area_5_agregada": agreg_cap5,
            "cap_area_10_agregada": agreg_cap10,
            "cap_area_20_agregada": agreg_cap20,
            "cap_ponta_ponta_agregada": agreg_pp,
            "cap_area_10_seca": m_cap10_seca,
            "cap_ponta_ponta_seca": m_pp_seca,
            "area_coberta_km2_media": m_area_km2,
            "pct_area_pa_media": m_pct_area
        })

    df_resumo = pd.DataFrame(linhas_resumo)

    # 3. Testes Estatísticos Pareados (Wilcoxon + Bootstrap 95% + Holm)
    logger.info("Executando testes pareados de Wilcoxon e Bootstrap 95% com correção de Holm...")
    xgb_caps = df_mensal[df_mensal["metodo"] == "xgboost"].set_index("mes_alvo")["captura_area_10"].loc[meses_alvo].values
    comparadores = {
        "persistencia_celula": df_mensal[df_mensal["metodo"] == "persistencia_celula"].set_index("mes_alvo")["captura_area_10"].loc[meses_alvo].values,
        "climatologia_celula": df_mensal[df_mensal["metodo"] == "climatologia_celula"].set_index("mes_alvo")["captura_area_10"].loc[meses_alvo].values,
        "historico_fixo_celula": df_mensal[df_mensal["metodo"] == "historico_fixo_celula"].set_index("mes_alvo")["captura_area_10"].loc[meses_alvo].values
    }

    rng = np.random.RandomState(42)
    linhas_testes = []
    p_brutos = []

    for comp_nome, comp_vals in comparadores.items():
        diffs = xgb_caps - comp_vals
        dif_media = float(np.mean(diffs))

        # Bootstrap 95%
        idx_boot = rng.randint(0, len(diffs), size=(10000, len(diffs)))
        boot_means = np.mean(diffs[idx_boot], axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        # Wilcoxon
        w_res = wilcoxon(xgb_caps, comp_vals, alternative="two-sided")
        pval = float(w_res.pvalue)
        p_brutos.append(pval)

        linhas_testes.append({
            "modelo": "xgboost",
            "comparado_com": comp_nome,
            "diferenca_media": dif_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": pval
        })

    p_adj = holm_bonferroni(np.array(p_brutos))
    for i, p_h in enumerate(p_adj):
        linhas_testes[i]["p_holm"] = float(p_h)

    df_testes = pd.DataFrame(linhas_testes)

    return df_mensal, df_resumo, df_testes

def gerar_mapa_exemplo_altamira(
    gdf_cells: gpd.GeoDataFrame,
    df_teste: pd.DataFrame,
    focos_parquet_path: Path,
    mun_gpkg_path: Path,
    out_png_path: Path
):
    """
    Gera o mapa cartográfico de exemplo para Altamira em 2024-09:
    Células coloridas pela previsão do XGBoost e pontos de focos observados sobrepostos.
    """
    logger.info("Gerando mapa cartográfico de exemplo para Altamira em setembro de 2024...")
    cd_altamira = "1500602"
    mes_exemplo = "2024-09"

    # 1. Carregar contorno municipal de Altamira em EPSG:5880
    gdf_mun = gpd.read_file(mun_gpkg_path).to_crs(epsg=5880)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)
    geom_altamira = gdf_mun[gdf_mun["cd_mun"] == cd_altamira].copy()

    # 2. Células de Altamira com predições do XGBoost
    df_alt_m = df_teste[(df_teste["mes_alvo"] == mes_exemplo) & (df_teste["cd_mun"] == cd_altamira)].copy()
    gdf_alt_cells = gdf_cells[gdf_cells["cell_id"].isin(df_alt_m["cell_id"])].copy()
    gdf_alt_cells = gdf_alt_cells.merge(df_alt_m[["cell_id", "pred_xgb", "y", "area_km2"]], on="cell_id")

    # Identificar top 10% da área para destacar
    gdf_alt_cells = gdf_alt_cells.sort_values(by=["pred_xgb", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    cum_a = gdf_alt_cells["area_km2"].cumsum().values
    tot_a = cum_a[-1]
    mask_top10 = (cum_a - gdf_alt_cells["area_km2"].values) < (0.10 * tot_a)
    gdf_alt_cells["top_10pct"] = mask_top10

    # 3. Carregar focos reais de Altamira em 2024-09
    df_focos = pd.read_parquet(
        focos_parquet_path,
        columns=["latitude", "longitude", "acq_date", "cd_mun"]
    )
    df_focos["cd_mun"] = df_focos["cd_mun"].astype(str).str.zfill(7)
    df_focos_alt = df_focos[
        (df_focos["cd_mun"] == cd_altamira) &
        (df_focos["acq_date"].astype(str).str.startswith(mes_exemplo))
    ].copy()

    # Projetar focos para EPSG:5880
    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos_alt["longitude"].values, df_focos_alt["latitude"].values)
    gdf_focos_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:5880"
    )

    focos_observados = len(gdf_focos_pts)
    focos_cap_top10 = gdf_alt_cells[gdf_alt_cells["top_10pct"]]["y"].sum()
    pct_captura = (focos_cap_top10 / focos_observados * 100.0) if focos_observados > 0 else 0.0

    # 4. Plotagem com matplotlib
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

    # Limite do município
    geom_altamira.plot(ax=ax, color="#f8f9fa", edgecolor="#343a40", linewidth=1.5, zorder=1)

    # Células coloridas por previsão do XGBoost
    vmax = float(np.percentile(gdf_alt_cells["pred_xgb"], 99))
    vmax = max(vmax, 10.0)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    choro = gdf_alt_cells.plot(
        column="pred_xgb",
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        edgecolor="#ced4da",
        linewidth=0.3,
        legend=True,
        legend_kwds={
            "label": "Focos Previstos pelo XGBoost (count:poisson)",
            "orientation": "horizontal",
            "shrink": 0.65,
            "pad": 0.03
        },
        zorder=2
    )

    # Contorno das células do Top 10% da área
    gdf_alt_cells[gdf_alt_cells["top_10pct"]].plot(
        ax=ax,
        facecolor="none",
        edgecolor="#004085",
        linewidth=1.2,
        zorder=3
    )

    # Focos reais de satélite como pontos
    if len(gdf_focos_pts) > 0:
        gdf_focos_pts.plot(
            ax=ax,
            color="#212529",
            markersize=3.0,
            alpha=0.6,
            zorder=4
        )

    # Anotação informativa
    texto_info = (
        f"Altamira — Setembro/2024\n"
        f"• Total de focos observados: {focos_observados:,}\n"
        f"• Top 10% da área selecionada: {gdf_alt_cells['top_10pct'].sum()} células (borda azul)\n"
        f"• Focos capturados no top 10%: {focos_cap_top10:,.0f} ({pct_captura:.1f}%)\n"
        f"• Pontos pretos: focos reais VIIRS"
    )
    ax.text(
        0.03, 0.03, texto_info,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#adb5bd", alpha=0.92),
        zorder=5
    )

    ax.set_title(
        "APHIM — Priorização Espacial em Nível 2 (Grade 10 km)\nMunicípio de Altamira (PA) — Setembro de 2024",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_axis_off()
    plt.tight_layout()

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mapa de exemplo salvo com sucesso em '{out_png_path}'.")

def main():
    t_start = time.time()
    logger.info("=== INICIANDO SCRIPT 07: PRIORIZAÇÃO ESPACIAL EM NÍVEL 2 (GRADE) ===")

    parser = argparse.ArgumentParser(description="Modelagem e Priorização em Grade Espacial.")
    parser.add_argument("--tamanho_m", type=float, default=None, help="Tamanho da célula em metros.")
    args = parser.parse_args()

    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    interim_dir = Path(config["caminhos"]["interim"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    tamanho_m = args.tamanho_m
    if tamanho_m is None:
        tamanho_m = float(config.get("grade", {}).get("tamanho_m", 10000))
    tamanho_str = f"{int(tamanho_m)}"

    logger.info(f"Tamanho de célula configurado: {tamanho_m:,.0f} metros ({tamanho_m/1000:.1f} km).")

    # 1. Carregar base de modelagem e GeoPackage de células
    parquet_path = processed_dir / f"grade_{tamanho_str}m_modelagem.parquet"
    gpkg_path = processed_dir / f"grade_{tamanho_str}m_celulas.gpkg"

    if not parquet_path.exists() or not gpkg_path.exists():
        raise FileNotFoundError(
            f"Arquivos da etapa 06 não encontrados ('{parquet_path}' ou '{gpkg_path}'). "
            f"Execute primeiro o módulo 06_grade_base.py."
        )

    logger.info(f"Carregando base de modelagem da grade de '{parquet_path}'...")
    df_mod = pd.read_parquet(parquet_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    logger.info(f"Carregando geometrias das células de '{gpkg_path}'...")
    gdf_cells = gpd.read_file(gpkg_path)
    gdf_cells["cd_mun"] = gdf_cells["cd_mun"].astype(str).str.zfill(7)
    cell_area_map = dict(zip(gdf_cells["cell_id"], gdf_cells["area_km2_dentro_pa"]))
    area_total_pa = float(gdf_cells["area_km2_dentro_pa"].sum())

    # Mapear área para o dataframe de modelagem
    df_mod["area_km2"] = df_mod["cell_id"].map(cell_area_map)

    # 2. Calcular baseline 'historico_fixo_celula' (total 2019-2023)
    logger.info("Calculando total de focos por célula de 2019 a 2023 (historico_fixo_celula)...")
    df_hist_alvos = df_mod[df_mod["mes_alvo"] <= "2023-12"]
    s_alvos = df_hist_alvos.groupby("cell_id")["y"].sum()
    df_jan19 = df_mod[df_mod["mes_emissao"] == "2019-01"]
    s_jan19 = df_jan19.groupby("cell_id")["focos_t"].sum()
    s_hist_fixo = (s_alvos + s_jan19).to_dict()

    df_mod["focos_fixo_2019_2023"] = df_mod["cell_id"].map(s_hist_fixo).fillna(0.0)

    # 3. Carregar predições municipais de Nível 1 (modelo vencedor de 03b se existir, senão perigo original)
    preds_v2_path = resultados_dir / "features_v2_predicoes_teste.parquet"
    preds_p1_path = resultados_dir / "perigo_predicoes_teste.parquet"

    if preds_v2_path.exists():
        logger.info(f"Carregando predições municipais do modelo vencedor de 03b de '{preds_v2_path}'...")
        df_pred_l1 = pd.read_parquet(preds_v2_path)
        col_pred_l1 = "pred_vencedor"
    elif preds_p1_path.exists():
        logger.info(f"Carregando predições municipais de '{preds_p1_path}'...")
        df_pred_l1 = pd.read_parquet(preds_p1_path)
        col_pred_l1 = "pred_rf_A"
        df_pred_l1["pred_vencedor"] = df_pred_l1[col_pred_l1]
    else:
        raise FileNotFoundError("Nenhum arquivo de predições de teste encontrado.")

    df_pred_l1["cd_mun"] = df_pred_l1["cd_mun"].astype(str).str.zfill(7)

    # 4. Partição Treino e Validação de Hiperparâmetros
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    del df_mod
    import gc
    gc.collect()

    t_val_start = time.time()
    melhores_params, melhor_score, todas_comb = validar_hiperparametros(df_treino, cell_area_map)
    t_val_elapsed = time.time() - t_val_start

    # Salvar hiperparâmetros detalhados em JSON
    out_hiper_json = resultados_dir / f"grade_{tamanho_str}m_hiperparametros.json"
    with open(out_hiper_json, "w", encoding="utf-8") as f:
        json.dump({
            "tamanho_m": int(tamanho_m),
            "melhores_parametros": melhores_params,
            "melhor_score_validacao": melhor_score,
            "todas_combinacoes": todas_comb
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Hiperparâmetros salvos com sucesso em '{out_hiper_json}'.")

    # 5. Treinar modelo XGBoost final em todo o conjunto de treino (2020 a 2023)
    logger.info("Treinando modelo XGBoost final em todas as linhas de treino...")
    t_train_start = time.time()
    modelo_final = xgb.XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        max_depth=melhores_params["max_depth"],
        learning_rate=melhores_params["learning_rate"],
        n_estimators=melhores_params["n_estimators"],
        min_child_weight=melhores_params["min_child_weight"],
        random_state=42,
        n_jobs=-1
    )
    modelo_final.fit(df_treino[FEATURES_CELULA], df_treino["y"])
    t_train_elapsed = time.time() - t_train_start
    logger.info(f"Modelo final treinado com sucesso em {t_train_elapsed:.1f} segundos.")

    # 6. Gerar predições de teste (2024-2025) e avaliar
    logger.info("Gerando predições de teste para as células...")
    t_eval_start = time.time()
    df_teste["pred_xgb"] = modelo_final.predict(df_teste[FEATURES_CELULA])

    # 7. Avaliação nos 24 meses de teste
    df_mensal, df_resumo, df_testes = avaliar_metodos_teste(
        df_teste=df_teste,
        df_pred_l1=df_pred_l1,
        cell_area_map=cell_area_map,
        area_total_pa=area_total_pa
    )
    t_eval_elapsed = time.time() - t_eval_start

    # 8. Salvar saídas CSV
    out_mensal_csv = resultados_dir / f"grade_{tamanho_str}m_mensal.csv"
    out_resumo_csv = resultados_dir / f"grade_{tamanho_str}m_resumo.csv"
    out_testes_csv = resultados_dir / f"grade_{tamanho_str}m_testes.csv"

    df_mensal.to_csv(out_mensal_csv, index=False, encoding="utf-8")
    df_resumo.to_csv(out_resumo_csv, index=False, encoding="utf-8")
    df_testes.to_csv(out_testes_csv, index=False, encoding="utf-8")

    # Salvar tempos em JSON
    out_tempos_json = resultados_dir / f"grade_{tamanho_str}m_tempos.json"
    tempos_dict = {
        "tamanho_m": int(tamanho_m),
        "tempo_validacao_s": round(t_val_elapsed, 2),
        "tempo_treino_final_s": round(t_train_elapsed, 2),
        "tempo_avaliacao_s": round(t_eval_elapsed, 2),
        "tempo_total_s": round(time.time() - t_start, 2)
    }
    with open(out_tempos_json, "w", encoding="utf-8") as f:
        json.dump(tempos_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Tempos de execução salvos com sucesso em '{out_tempos_json}'.")
    logger.info(f"Tabelas salvas com sucesso em '{resultados_dir}'.")

    # 9. Gerar mapa de exemplo para Altamira em 2024-09
    mun_gpkg_path = processed_dir / "municipios_pa.gpkg"
    focos_parquet_path = interim_dir / "firms_focos_pa.parquet"
    out_png_path = resultados_dir / f"grade_{tamanho_str}m_exemplo_altamira_2024-09.png"

    gerar_mapa_exemplo_altamira(
        gdf_cells=gdf_cells,
        df_teste=df_teste,
        focos_parquet_path=focos_parquet_path,
        mun_gpkg_path=mun_gpkg_path,
        out_png_path=out_png_path
    )

    t_total = time.time() - t_start

    # Tamanhos dos arquivos
    gpkg_mb = gpkg_path.stat().st_size / (1024 * 1024)
    parquet_mb = parquet_path.stat().st_size / (1024 * 1024)
    png_mb = out_png_path.stat().st_size / (1024 * 1024)

    # 10. Apresentação dos Resultados
    print("\n" + "=" * 100)
    print("      APHIM — PRIORIZAÇÃO EM DOIS NÍVEIS (GRADE 10 KM) — RESULTADOS NO TESTE (2024-2025)")
    print("=" * 100)
    print("\nTABELA 1 — RESUMO COMPARATIVO DOS MÉTODOS DE CÉLULA (MÉDIAS MENSAIS, AGREGADAS E SECA):")
    cols_show = [
        "metodo", "cap_area_5_media", "cap_area_10_media", "cap_area_20_media",
        "cap_ponta_ponta_media", "cap_area_10_agregada", "cap_ponta_ponta_agregada",
        "cap_area_10_seca", "cap_ponta_ponta_seca", "pct_area_pa_media"
    ]
    df_show = df_resumo[cols_show].copy()
    for c in cols_show[1:]:
        df_show[c] = df_show[c].apply(lambda x: f"{x*100:.2f}%" if c != "pct_area_pa_media" else f"{x:.2f}%")
    df_show.columns = [
        "Método", "CapÁrea@5%", "CapÁrea@10%", "CapÁrea@20%",
        "PontaPonta@10%", "CapÁrea@10% Agr", "PontaPonta Agr",
        "CapÁrea@10% Seca", "PontaPonta Seca", "% Área PA"
    ]
    print(df_show.to_string(index=False))

    print("\nTABELA 2 — TESTES ESTATÍSTICOS PAREADOS NA CAPTURAÁREA@10% (WILCOXON + BOOTSTRAP 95% + HOLM):")
    df_t_show = df_testes.copy()
    df_t_show["diferenca_media"] = df_t_show["diferenca_media"].apply(lambda x: f"{x*100:+.2f} p.p.")
    df_t_show["ic_95"] = df_t_show.apply(lambda r: f"[{r['ic_95_inf']*100:+.2f}%, {r['ic_95_sup']*100:+.2f}%]", axis=1)
    df_t_show["p_valor"] = df_t_show["p_valor"].apply(lambda p: f"{p:.4e}" if p < 0.001 else f"{p:.4f}")
    df_t_show["p_holm"] = df_t_show["p_holm"].apply(lambda p: f"{p:.4e}" if p < 0.001 else f"{p:.4f}")
    cols_t = ["modelo", "comparado_com", "diferenca_media", "ic_95", "p_valor", "p_holm"]
    df_t_show = df_t_show[cols_t]
    df_t_show.columns = ["Modelo", "Comparador", "Dif. Média", "IC 95% Bootstrap", "p-valor Wilcoxon", "p-valor Holm"]
    print(df_t_show.to_string(index=False))

    print("\nTABELA 3 — TEMPOS E TAMANHOS DE ARQUIVOS (BASE PARA ESTIMATIVA DE 5 KM E 2,5 KM):")
    print(f"• Tempo total de execução do script 07:      {t_total:.1f} segundos")
    print(f"• Tempo de ajuste do modelo final (XGBoost): {t_train_elapsed:.1f} segundos")
    print(f"• Tamanho do arquivo GPKG da grade:          {gpkg_mb:.2f} MB")
    print(f"• Tamanho do arquivo Parquet de modelagem:   {parquet_mb:.2f} MB")
    print(f"• Tamanho do mapa PNG de exemplo:            {png_mb:.2f} MB")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()
