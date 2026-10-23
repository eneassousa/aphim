"""
Módulo 17 — Avaliação dos Modelos Congelados em 2026 (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Espacial no Pará.

Objetivos:
1. Avaliar os modelos congelados D0 (município) e G2 (grade) em 2026 nos regimes:
   - T25 (retreinado com alvos até 2025-12, versão principal)
   - T23 (treinado com alvos até 2023-12, versão secundária)
2. Nível 1 (Município):
   - D0 vs Climatologia, Persistência, Histórico Fixo
   - Métrica: Captura@15
3. Nível 2 (Grade de 5 km):
   - Células restritas aos 15 municípios do topo de cada mês segundo o D0 de 2026
   - G2 vs G1, G0, Histórico Fixo, Climatologia, Persistência, Oráculo
   - Métricas: CapturaÁrea@5%, @10%, @20% e Captura Ponta a Ponta
4. Estatísticas:
   - Diferença média, Bootstrap 95% (10.000 reamostragens, semente 42)
   - Wilcoxon pareado e correção de Holm-Bonferroni
   - Contagem de meses vencidos
   - Desagregação por estação seca (jul-out) e chuva (jan-jun), e meses NRT
5. Salvar previsões (só T25) e mapas de Altamira e São Félix do Xingu no mês de maior fogo.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from scipy.stats import wilcoxon
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import calcular_metricas_mes

logger = setup_logger("processamento_17_teste_2026_modelos")

# Atributos D0 (20 features)
FEATURES_D0 = [
    "focos_media_hist_mes",
    "focos_t",
    "focos_t1",
    "mes_sin",
    "focos_vizinhos_t",
    "mes_cos",
    "prec_6m",
    "anom_prec_6m",
    "razao_fogo_t",
    "anom_prec_mm",
    "anom_t2m_media_c",
    "focos_t2",
    "variacao_anual_t",
    "umid_solo_t1",
    "razao_fogo_vizinhos_t",
    "anom_umid_solo",
    "anom_umid_solo_3m",
    "anom_prec_3m",
    "meses_secos_seguidos",
    "prec_3m"
]

# Atributos Grade
FEATURES_G0 = [
    "focos_t", "focos_t1", "focos_t2", "focos_mm3", "focos_mesmo_mes_ano_ant",
    "focos_media_hist_mes", "focos_vizinhos_t", "focos_vizinhos_media_hist_mes",
    "mes_sin", "mes_cos", "focos_mun_t", "prec_mm", "ur_pct", "dpv_kpa",
    "umid_solo", "anom_prec_mm"
]

FEATURES_G1 = FEATURES_G0 + [
    "deter_desmat_cel_3m", "deter_desmat_cel_12m",
    "deter_degrad_cel_3m", "deter_degrad_cel_12m",
    "deter_desmat_viz_12m"
]

FEATURES_RECORR_BASE = [
    "meses_desde_ultimo_foco", "anos_com_fogo_3a",
    "frac_meses_fogo_24m", "focos_janela_retorno",
    "frac_meses_fogo_24m_viz"
]

# Hiperparâmetros congelados
PARAMS_D0 = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1
}

PARAMS_G0 = {
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 600,
    "min_child_weight": 1
}

PARAMS_G1 = {
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "min_child_weight": 1
}

PARAMS_G2 = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_weight": 10
}

def get_next_month(mes_str: str) -> str:
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    if mes == 12:
        return f"{ano + 1:04d}-01"
    return f"{ano:04d}-{mes + 1:02d}"

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0: return np.array([], dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_p = np.empty(n, dtype=float)
    running_max = 0.0
    for k in range(n):
        val = sorted_p[k] * (n - k)
        running_max = max(running_max, val)
        adj_p[k] = min(1.0, running_max)
    out = np.empty(n, dtype=float)
    out[order] = adj_p
    return out

def bootstrap_ci_diff(diferencas: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diferencas)
    if n == 0: return (0.0, 0.0)
    indices = rng.integers(0, n, size=(n_boot, n))
    medias = np.mean(diferencas[indices], axis=1)
    ic_inf = float(np.percentile(medias, 2.5))
    ic_sup = float(np.percentile(medias, 97.5))
    return ic_inf, ic_sup

def treinar_modelo_xgb(params: Dict[str, Any], X_tr: pd.DataFrame, y_tr: np.ndarray) -> xgb.XGBRegressor:
    try:
        model = xgb.XGBRegressor(
            **params,
            objective="count:poisson",
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        model.fit(X_tr, y_tr)
        return model
    except Exception:
        model = xgb.XGBRegressor(
            **params,
            objective="count:poisson",
            tree_method="hist",
            device="cpu",
            random_state=42
        )
        model.fit(X_tr, y_tr)
        return model

def calcular_captura_area_mes(
    df_mes: pd.DataFrame,
    col_score: str,
    top15_muns: List[str],
    pct_area_k: float = 10.0
) -> Tuple[float, float, float]:
    """Calcula a CapturaÁrea@k% para um único mês dentro dos 15 municípios conforme o protocolo original.
    
    Retorna:
        cap: proporção de focos capturados nos k% de cada município / total de focos nos 15 municípios
        focos_capturados: total absoluto de focos capturados nos 15 municípios
        area_coberta: área total selecionada (km²)
    """
    sub_top15 = df_mes[df_mes["cd_mun"].isin(top15_muns)]
    focos_capturados = 0.0
    focos_top15 = 0.0
    area_coberta = 0.0

    for _, grp in sub_top15.groupby("cd_mun"):
        f_mun = grp["y"].sum()
        focos_top15 += f_mun

        grp_sort = grp.sort_values(by=[col_score, "cell_id"], ascending=[False, True])
        cell_areas = grp_sort["area_km2"].values
        cum_areas = np.cumsum(cell_areas)
        total_mun_area = cum_areas[-1] if len(cum_areas) > 0 else 0.0
        target_area = (pct_area_k / 100.0) * total_mun_area

        mask = (cum_areas - cell_areas) < target_area
        sel = grp_sort.iloc[mask]
        focos_capturados += sel["y"].sum()
        area_coberta += sel["area_km2"].sum()

    cap = (focos_capturados / focos_top15) if focos_top15 > 0 else 0.0
    return float(cap), float(focos_capturados), float(area_coberta)

def gerar_mapa_municipio(
    cd_mun: str,
    nm_mun: str,
    mes_exemplo: str,
    gdf_mun: gpd.GeoDataFrame,
    gdf_cells: gpd.GeoDataFrame,
    df_pred_grade: pd.DataFrame,
    focos_parquet_path: Path,
    out_png_path: Path
):
    logger.info(f"Gerando mapa cartográfico para {nm_mun} ({cd_mun}) em {mes_exemplo}...")
    geom_mun = gdf_mun[gdf_mun["cd_mun"] == cd_mun].to_crs(epsg=5880)

    df_sub = df_pred_grade[(df_pred_grade["mes_alvo"] == mes_exemplo) & (df_pred_grade["cd_mun"] == cd_mun)].copy()
    if df_sub.empty:
        logger.warning(f"Sem dados de grade para {nm_mun} em {mes_exemplo}.")
        return

    gdf_m_cells = gdf_cells[gdf_cells["cell_id"].isin(df_sub["cell_id"])].to_crs(epsg=5880).copy()
    gdf_m_cells = gdf_m_cells.merge(df_sub[["cell_id", "pred_g2", "y", "area_km2"]], on="cell_id")

    gdf_m_cells = gdf_m_cells.sort_values(by=["pred_g2", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    cum_a = gdf_m_cells["area_km2"].cumsum().values
    tot_a = cum_a[-1]
    mask_top10 = (cum_a - gdf_m_cells["area_km2"].values) < (0.10 * tot_a)
    gdf_m_cells["top_10pct"] = mask_top10

    # Focos reais
    df_focos = pd.read_parquet(focos_parquet_path, columns=["latitude", "longitude", "acq_date", "cd_mun"])
    df_focos["cd_mun"] = df_focos["cd_mun"].astype(str).str.zfill(7)
    df_focos_sub = df_focos[(df_focos["cd_mun"] == cd_mun) & (df_focos["acq_date"].astype(str).str.startswith(mes_exemplo))].copy()

    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos_sub["longitude"].values, df_focos_sub["latitude"].values) if not df_focos_sub.empty else ([], [])
    gdf_focos_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs="EPSG:5880")

    focos_obs = len(gdf_focos_pts)
    focos_cap10 = gdf_m_cells[gdf_m_cells["top_10pct"]]["y"].sum()
    pct_cap = (focos_cap10 / focos_obs * 100.0) if focos_obs > 0 else 0.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

    geom_mun.plot(ax=ax, color="#f8f9fa", edgecolor="#343a40", linewidth=1.5, zorder=1)

    vmax = float(np.percentile(gdf_m_cells["pred_g2"], 99)) if len(gdf_m_cells) > 0 else 10.0
    vmax = max(vmax, 5.0)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    gdf_m_cells.plot(
        column="pred_g2",
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        edgecolor="#ced4da",
        linewidth=0.3,
        legend=True,
        legend_kwds={
            "label": "Focos Previstos pelo G2 (XGBoost Recorrência)",
            "orientation": "horizontal",
            "shrink": 0.65,
            "pad": 0.03
        },
        zorder=2
    )

    gdf_m_cells[gdf_m_cells["top_10pct"]].plot(
        ax=ax,
        facecolor="none",
        edgecolor="#004085",
        linewidth=1.2,
        zorder=3
    )

    if len(gdf_focos_pts) > 0:
        gdf_focos_pts.plot(ax=ax, color="#212529", markersize=3.5, alpha=0.65, zorder=4)

    texto_info = (
        f"{nm_mun} — {mes_exemplo}\n"
        f"• Total de focos observados: {focos_obs:,}\n"
        f"• Top 10% da área: {gdf_m_cells['top_10pct'].sum()} células (borda azul)\n"
        f"• Focos capturados no top 10%: {focos_cap10:,.0f} ({pct_cap:.1f}%)\n"
        f"• Pontos pretos: focos reais de satélite VIIRS"
    )
    ax.text(
        0.03, 0.03, texto_info,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#adb5bd", alpha=0.92),
        zorder=5
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Mapa salvo em '{out_png_path}'.")

def main():
    logger.info("=== AVALIAÇÃO DOS MODELOS CONGELADOS EM 2026 (T25 e T23) ===")
    t_start = time.time()

    processed_dir = Path("data/processed_2026")
    res_dir = processed_dir / "resultados"
    interim_dir = Path("data/interim_2026")
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar Bases
    logger.info("Carregando bases municipais e de grade...")
    df_mun = pd.read_parquet(processed_dir / "aphim_modelagem_v2.parquet")
    df_grade = pd.read_parquet(processed_dir / "grade_5000m_modelagem.parquet")
    df_deter = pd.read_parquet(processed_dir / "grade_5000m_deter.parquet")
    df_recorr = pd.read_parquet(processed_dir / "grade_5000m_recorrencia.parquet")

    gdf_mun = gpd.read_file(processed_dir / "municipios_pa.gpkg")
    gdf_cells = gpd.read_file(processed_dir / "grade_5000m_celulas.gpkg")
    focos_parquet_path = interim_dir / "firms_focos_pa.parquet"

    # Merge grade com deter e recorrência
    cols_det = ["deter_desmat_cel_3m", "deter_desmat_cel_12m", "deter_degrad_cel_3m", "deter_degrad_cel_12m", "deter_desmat_viz_12m"]
    cols_rec = [
        "meses_desde_ultimo_foco", "anos_com_fogo_3a", "frac_meses_fogo_24m",
        "focos_janela_retorno", "taxa_retorno_emp_final", "taxa_retorno_emp_t25",
        "frac_meses_fogo_24m_viz"
    ]
    df_grade_full = df_grade.merge(df_deter[["cell_id", "mes_emissao"] + cols_det], on=["cell_id", "mes_emissao"])
    df_grade_full = df_grade_full.merge(df_recorr[["cell_id", "mes_emissao"] + cols_rec], on=["cell_id", "mes_emissao"])
    cell_area_map = dict(zip(gdf_cells["cell_id"].astype(np.int32), gdf_cells["area_km2_dentro_pa"].astype(float)))
    df_grade_full["area_km2"] = df_grade_full["cell_id"].map(cell_area_map)

    # Meses avaliados em 2026
    meses_emissao_2026 = sorted(df_mun[df_mun["conjunto"] == "teste_2026"]["mes_emissao"].unique())
    meses_alvo_2026 = [get_next_month(m) for m in meses_emissao_2026]
    logger.info(f"Meses de emissão 2026 avaliados ({len(meses_emissao_2026)} meses): {meses_emissao_2026}")
    logger.info(f"Meses alvo 2026 ({len(meses_alvo_2026)} meses): {meses_alvo_2026}")

    # Subconjunto de teste 2026
    test_mun = df_mun[df_mun["mes_emissao"].isin(meses_emissao_2026)].copy()
    test_grade = df_grade_full[df_grade_full["mes_emissao"].isin(meses_emissao_2026)].copy()

    # Baselines municipais (usando dados até 2025)
    df_mun_hist = df_mun[df_mun["mes_emissao"] <= "2025-11"].copy()
    df_mun_hist["cal_m"] = df_mun_hist["mes_alvo"].str.slice(5, 7).astype(int)
    clima_mun_dict = df_mun_hist.groupby(["cd_mun", "cal_m"])["y"].mean().to_dict()
    hist_fixo_mun_dict = df_mun_hist.groupby("cd_mun")["y"].sum().to_dict()

    # Baselines celulares (usando dados até 2025)
    df_grade_hist = df_grade_full[df_grade_full["mes_emissao"] <= "2025-11"].copy()
    df_grade_hist["cal_m"] = df_grade_hist["mes_alvo"].str.slice(5, 7).astype(int)
    clima_cell_dict = df_grade_hist.groupby(["cell_id", "cal_m"])["y"].mean().to_dict()
    hist_fixo_cell_dict = df_grade_hist.groupby("cell_id")["y"].sum().to_dict()

    # Regimes de Treino: T25 e T23
    regimes = {
        "T25": {
            "tr_mun": df_mun[(df_mun["mes_emissao"] >= "2020-01") & (df_mun["mes_emissao"] <= "2025-11")],
            "tr_grade": df_grade_full[(df_grade_full["mes_emissao"] >= "2020-01") & (df_grade_full["mes_emissao"] <= "2025-11")],
            "col_taxa": "taxa_retorno_emp_t25"
        },
        "T23": {
            "tr_mun": df_mun[(df_mun["mes_emissao"] >= "2020-01") & (df_mun["mes_emissao"] <= "2023-11")],
            "tr_grade": df_grade_full[(df_grade_full["mes_emissao"] >= "2020-01") & (df_grade_full["mes_emissao"] <= "2023-11")],
            "col_taxa": "taxa_retorno_emp_final"
        }
    }

    res_mensal_mun_list = []
    res_mensal_grade_list = []
    predicoes_mun_t25 = None
    predicoes_grade_t25 = None
    predicoes_grade_t23 = None

    for regime_nome, reg_info in regimes.items():
        logger.info(f"\n=================== EXECUTANDO REGIME {regime_nome} ===================")
        tr_m = reg_info["tr_mun"]
        tr_g = reg_info["tr_grade"]
        col_taxa = reg_info["col_taxa"]

        # 1. Treinar Nível 1: D0
        logger.info(f"Treinando D0 ({regime_nome}) com {len(tr_m):,} registros...")
        m_d0 = treinar_modelo_xgb(PARAMS_D0, tr_m[FEATURES_D0], tr_m["y"].values)
        preds_d0 = m_d0.predict(test_mun[FEATURES_D0])

        df_eval_mun = test_mun[["cd_mun", "nm_mun", "mes_emissao", "mes_alvo", "y", "pct_floresta"]].copy()
        df_eval_mun["pred_d0"] = preds_d0
        df_eval_mun["focos_t"] = test_mun["focos_t"].values
        df_eval_mun["cal_m"] = df_eval_mun["mes_alvo"].str.slice(5, 7).astype(int)
        df_eval_mun["pred_clima"] = [clima_mun_dict.get((c, m), 0.0) for c, m in zip(df_eval_mun["cd_mun"], df_eval_mun["cal_m"])]
        df_eval_mun["pred_hist_fixo"] = [hist_fixo_mun_dict.get(c, 0.0) for c in df_eval_mun["cd_mun"]]
        df_eval_mun["pred_persistencia"] = df_eval_mun["focos_t"]

        if regime_nome == "T25":
            predicoes_mun_t25 = df_eval_mun.copy()

        # Avaliar Captura@15 mensal no município
        top15_muns_por_mes = {}
        for m_alvo in meses_alvo_2026:
            sub_m = df_eval_mun[df_eval_mun["mes_alvo"] == m_alvo]
            # Identificar top 15 do D0 para a grade
            sub_sort_d0 = sub_m.sort_values(by=["pred_d0", "cd_mun"], ascending=[False, True])
            top15_muns = sub_sort_d0.head(15)["cd_mun"].tolist()
            top15_muns_por_mes[m_alvo] = top15_muns

            met_d0, _ = calcular_metricas_mes(sub_m, col_score="pred_d0", k_valores=(15,))
            met_cli, _ = calcular_metricas_mes(sub_m, col_score="pred_clima", k_valores=(15,))
            met_pers, _ = calcular_metricas_mes(sub_m, col_score="pred_persistencia", k_valores=(15,))
            met_hist, _ = calcular_metricas_mes(sub_m, col_score="pred_hist_fixo", k_valores=(15,))

            is_seca = int(m_alvo[5:7]) >= 7
            is_nrt = m_alvo >= "2026-07"
            mes_parcial = (m_alvo == "2026-10")
            res_mensal_mun_list.append({
                "regime": regime_nome,
                "mes_alvo": m_alvo,
                "estacao": "seca" if is_seca else "chuva",
                "is_nrt": is_nrt,
                "mes_parcial": mes_parcial,
                "cap15_d0": met_d0["captura_15"],
                "cap15_clima": met_cli["captura_15"],
                "cap15_persistencia": met_pers["captura_15"],
                "cap15_hist_fixo": met_hist["captura_15"]
            })

        # Comparação de áreas para 2026-01 na primeira iteração (T25)
        if regime_nome == "T25":
            top15_01 = top15_muns_por_mes["2026-01"]
            cells_top15_01 = gdf_cells[gdf_cells["cd_mun"].isin(top15_01)]
            soma_dentro_pa = cells_top15_01["area_km2_dentro_pa"].sum()
            soma_25km = len(cells_top15_01) * 25.0
            logger.info("\n=== COMPARAÇÃO DE ÁREAS DOS 15 MUNICÍPIOS DO TOPO (2026-01) ===")
            logger.info(f"Municípios Top 15 D0: {top15_01}")
            logger.info(f"Total de células: {len(cells_top15_01):,}")
            logger.info(f"Soma com area_km2_dentro_pa: {soma_dentro_pa:,.2f} km²")
            logger.info(f"Soma com área cheia (25 km²): {soma_25km:,.2f} km²")
            logger.info(f"Diferença absoluta: {soma_25km - soma_dentro_pa:,.2f} km² ({(soma_25km - soma_dentro_pa)/soma_dentro_pa*100:.2f}%)")
            logger.info("===============================================================\n")

        # 2. Treinar Nível 2: G0, G1, G2
        feats_g2 = FEATURES_G1 + FEATURES_RECORR_BASE + [col_taxa]
        logger.info(f"Treinando G0, G1, G2 ({regime_nome}) na grade ({len(tr_g):,} registros)...")

        m_g0 = treinar_modelo_xgb(PARAMS_G0, tr_g[FEATURES_G0], tr_g["y"].values)
        m_g1 = treinar_modelo_xgb(PARAMS_G1, tr_g[FEATURES_G1], tr_g["y"].values)

        # Para G2 renomeamos col_taxa para taxa_retorno_emp
        X_tr_g2 = tr_g[feats_g2].rename(columns={col_taxa: "taxa_retorno_emp"})
        m_g2 = treinar_modelo_xgb(PARAMS_G2, X_tr_g2, tr_g["y"].values)

        preds_g0 = m_g0.predict(test_grade[FEATURES_G0])
        preds_g1 = m_g1.predict(test_grade[FEATURES_G1])
        X_test_g2 = test_grade[feats_g2].rename(columns={col_taxa: "taxa_retorno_emp"})
        preds_g2 = m_g2.predict(X_test_g2)

        df_eval_grade = test_grade[["cell_id", "cd_mun", "mes_emissao", "mes_alvo", "y", "focos_t", "area_km2"]].copy()
        df_eval_grade["pred_g0"] = preds_g0
        df_eval_grade["pred_g1"] = preds_g1
        df_eval_grade["pred_g2"] = preds_g2
        df_eval_grade["cal_m"] = df_eval_grade["mes_alvo"].str.slice(5, 7).astype(int)
        df_eval_grade["pred_clima"] = [clima_cell_dict.get((c, m), 0.0) for c, m in zip(df_eval_grade["cell_id"], df_eval_grade["cal_m"])]
        df_eval_grade["pred_hist_fixo"] = [hist_fixo_cell_dict.get(c, 0.0) for c in df_eval_grade["cell_id"]]
        df_eval_grade["pred_persistencia"] = df_eval_grade["focos_t"]
        df_eval_grade["pred_oraculo"] = df_eval_grade["y"]

        if regime_nome == "T25":
            predicoes_grade_t25 = df_eval_grade.copy()
        elif regime_nome == "T23":
            predicoes_grade_t23 = df_eval_grade.copy()

        # Avaliar métricas no Top 15 municípios do D0 (separadamente por município)
        for m_alvo in meses_alvo_2026:
            top_muns = top15_muns_por_mes[m_alvo]
            sub_m_grade = df_eval_grade[df_eval_grade["mes_alvo"] == m_alvo]
            total_focos_pa = float(sub_m_grade["y"].sum())
            sub_top15_cells = sub_m_grade[sub_m_grade["cd_mun"].isin(top_muns)]
            focos_top15 = float(sub_top15_cells["y"].sum())

            cap_area_5_g2, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_g2", top_muns, pct_area_k=5.0)
            cap_area_10_g2, f_cap10_g2, _ = calcular_captura_area_mes(sub_m_grade, "pred_g2", top_muns, pct_area_k=10.0)
            cap_area_20_g2, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_g2", top_muns, pct_area_k=20.0)

            cap_area_10_g1, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_g1", top_muns, pct_area_k=10.0)
            cap_area_10_g0, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_g0", top_muns, pct_area_k=10.0)
            cap_area_10_hist, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_hist_fixo", top_muns, pct_area_k=10.0)
            cap_area_10_cli, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_clima", top_muns, pct_area_k=10.0)
            cap_area_10_pers, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_persistencia", top_muns, pct_area_k=10.0)
            cap_area_10_ora, _, _ = calcular_captura_area_mes(sub_m_grade, "pred_oraculo", top_muns, pct_area_k=10.0)

            # Ponta a ponta: focos capturados nos 10% de cada município divididos pelo total do Pará
            ponta_a_ponta_g2 = (f_cap10_g2 / total_focos_pa) if total_focos_pa > 0 else 0.0

            is_seca = int(m_alvo[5:7]) >= 7
            is_nrt = m_alvo >= "2026-07"
            mes_parcial = (m_alvo == "2026-10")
            res_mensal_grade_list.append({
                "regime": regime_nome,
                "mes_alvo": m_alvo,
                "estacao": "seca" if is_seca else "chuva",
                "is_nrt": is_nrt,
                "mes_parcial": mes_parcial,
                "total_focos_pa": total_focos_pa,
                "focos_top15_muns": focos_top15,
                "cap_area_5_g2": cap_area_5_g2,
                "cap_area_10_g2": cap_area_10_g2,
                "cap_area_20_g2": cap_area_20_g2,
                "ponta_a_ponta_g2": ponta_a_ponta_g2,
                "cap_area_10_g1": cap_area_10_g1,
                "cap_area_10_g0": cap_area_10_g0,
                "cap_area_10_hist_fixo": cap_area_10_hist,
                "cap_area_10_clima": cap_area_10_cli,
                "cap_area_10_persistencia": cap_area_10_pers,
                "cap_area_10_oraculo": cap_area_10_ora
            })

    # =========================================================================
    # TABELAS FINAIS E RESUMOS ESTATÍSTICOS
    # =========================================================================
    df_res_mun_mensal = pd.DataFrame(res_mensal_mun_list)
    df_res_grade_mensal = pd.DataFrame(res_mensal_grade_list)

    df_res_mun_mensal.to_csv(res_dir / "teste2026_municipio_mensal.csv", index=False)
    df_res_grade_mensal.to_csv(res_dir / "teste2026_grade_mensal.csv", index=False)
    logger.info("teste2026_municipio_mensal.csv e teste2026_grade_mensal.csv salvos.")

    # Filtros de período para resumos (com e sem outubro parcial)
    filtros_periodo = [
        ("todos_10_meses", lambda df: df),
        ("sem_outubro_9_meses", lambda df: df[df["mes_alvo"] != "2026-10"])
    ]

    # Resumos Estatísticos: Nível 1
    resumo_mun_rows = []
    comparadores_mun = [("climatologia", "cap15_clima"), ("persistencia", "cap15_persistencia"), ("historico_fixo", "cap15_hist_fixo")]

    for reg in ["T25", "T23"]:
        sub_reg = df_res_mun_mensal[df_res_mun_mensal["regime"] == reg]
        for nome_filtro, func_f in filtros_periodo:
            sub_m = func_f(sub_reg)
            n_meses = len(sub_m)
            c_d0 = sub_m["cap15_d0"].values
            p_vals = []
            rows_temp = []

            for comp_nome, col_comp in comparadores_mun:
                c_comp = sub_m[col_comp].values
                diff = c_d0 - c_comp
                diff_med = float(np.mean(diff))
                ic_inf, ic_sup = bootstrap_ci_diff(diff)
                w_stat, p_val = wilcoxon(diff, alternative="two-sided") if not np.all(diff == 0) else (0.0, 1.0)
                p_vals.append(p_val)
                venceu = int(np.sum(diff > 0))

                rows_temp.append({
                    "regime": reg,
                    "filtro_meses": nome_filtro,
                    "n_meses": n_meses,
                    "comparador": comp_nome,
                    "cap15_d0": float(np.mean(c_d0)),
                    "cap15_comparador": float(np.mean(c_comp)),
                    "diff_media": diff_med,
                    "ic95_inf": ic_inf,
                    "ic95_sup": ic_sup,
                    "p_wilcoxon": p_val,
                    "meses_venceu": venceu
                })

            p_adj = holm_bonferroni(np.array(p_vals))
            for r, pa in zip(rows_temp, p_adj):
                r["p_holm"] = pa
                resumo_mun_rows.append(r)

    df_resumo_mun = pd.DataFrame(resumo_mun_rows)
    df_resumo_mun.to_csv(res_dir / "teste2026_municipio_resumo.csv", index=False)

    # Resumos Estatísticos: Nível 2
    resumo_grade_rows = []
    comparadores_grade = [
        ("G1_xgb_deter", "cap_area_10_g1"),
        ("G0_xgb_base", "cap_area_10_g0"),
        ("historico_fixo_celula", "cap_area_10_hist_fixo"),
        ("climatologia_celula", "cap_area_10_clima"),
        ("persistencia_celula", "cap_area_10_persistencia"),
        ("oraculo_celula", "cap_area_10_oraculo")
    ]

    for reg in ["T25", "T23"]:
        sub_reg = df_res_grade_mensal[df_res_grade_mensal["regime"] == reg]
        for nome_filtro, func_f in filtros_periodo:
            sub_g = func_f(sub_reg)
            n_meses = len(sub_g)
            c_g2 = sub_g["cap_area_10_g2"].values
            p_vals = []
            rows_temp = []

            for comp_nome, col_comp in comparadores_grade:
                c_comp = sub_g[col_comp].values
                diff = c_g2 - c_comp
                diff_med = float(np.mean(diff))
                ic_inf, ic_sup = bootstrap_ci_diff(diff)
                w_stat, p_val = wilcoxon(diff, alternative="two-sided") if not np.all(diff == 0) else (0.0, 1.0)
                p_vals.append(p_val)
                venceu = int(np.sum(diff > 0))

                rows_temp.append({
                    "regime": reg,
                    "filtro_meses": nome_filtro,
                    "n_meses": n_meses,
                    "comparador": comp_nome,
                    "cap_area_10_g2": float(np.mean(c_g2)),
                    "cap_area_10_comparador": float(np.mean(c_comp)),
                    "diff_media": diff_med,
                    "ic95_inf": ic_inf,
                    "ic95_sup": ic_sup,
                    "p_wilcoxon": p_val,
                    "meses_venceu": venceu
                })

            p_adj = holm_bonferroni(np.array(p_vals))
            for r, pa in zip(rows_temp, p_adj):
                r["p_holm"] = pa
                resumo_grade_rows.append(r)

    df_resumo_grade = pd.DataFrame(resumo_grade_rows)
    df_resumo_grade.to_csv(res_dir / "teste2026_grade_resumo.csv", index=False)
    logger.info("teste2026_municipio_resumo.csv e teste2026_grade_resumo.csv salvos.")

    # Comparar predições T25 com arquivo anterior (verificação de identidade)
    p_t25_grade = res_dir / "teste2026_predicoes_grade.parquet"
    if p_t25_grade.exists():
        df_old_t25 = pd.read_parquet(p_t25_grade)
        diff_max_g2 = float(np.max(np.abs(predicoes_grade_t25["pred_g2"].values - df_old_t25["pred_g2"].values)))
        diff_max_g1 = float(np.max(np.abs(predicoes_grade_t25["pred_g1"].values - df_old_t25["pred_g1"].values)))
        diff_max_g0 = float(np.max(np.abs(predicoes_grade_t25["pred_g0"].values - df_old_t25["pred_g0"].values)))
        logger.info("\n=== VERIFICAÇÃO DE IDENTIDADE DAS PREVISÕES T25 ===")
        logger.info(f"Diferença máxima G2: {diff_max_g2:.10e}")
        logger.info(f"Diferença máxima G1: {diff_max_g1:.10e}")
        logger.info(f"Diferença máxima G0: {diff_max_g0:.10e}")
        logger.info("===================================================\n")

    # Salvar predições
    predicoes_mun_t25.to_parquet(res_dir / "teste2026_predicoes_municipio.parquet", index=False)
    predicoes_grade_t25.to_parquet(p_t25_grade, index=False)
    logger.info("Predições T25 salvas em parquet.")

    if predicoes_grade_t23 is not None:
        p_t23_grade = res_dir / "teste2026_predicoes_grade_T23.parquet"
        predicoes_grade_t23.to_parquet(p_t23_grade, index=False)
        logger.info(f"Predições T23 salvas em '{p_t23_grade}'.")

    # Identificar mês com maior total de focos no Pará em 2026
    sub_t25_grade = df_res_grade_mensal[df_res_grade_mensal["regime"] == "T25"]
    max_idx = sub_t25_grade["total_focos_pa"].idxmax()
    mes_max_fogo = sub_t25_grade.loc[max_idx, "mes_alvo"]
    focos_max = sub_t25_grade.loc[max_idx, "total_focos_pa"]
    logger.info(f"Mês de 2026 com mais focos no Pará: {mes_max_fogo} ({focos_max:,.0f} focos).")

    # Gerar mapas de Altamira e São Félix do Xingu
    cd_altamira = "1500602"
    cd_sao_felix = "1507300"

    gerar_mapa_municipio(
        cd_mun=cd_altamira,
        nm_mun="Altamira",
        mes_exemplo=mes_max_fogo,
        gdf_mun=gdf_mun,
        gdf_cells=gdf_cells,
        df_pred_grade=predicoes_grade_t25,
        focos_parquet_path=focos_parquet_path,
        out_png_path=res_dir / f"mapa_altamira_{mes_max_fogo}.png"
    )

    gerar_mapa_municipio(
        cd_mun=cd_sao_felix,
        nm_mun="São Félix do Xingu",
        mes_exemplo=mes_max_fogo,
        gdf_mun=gdf_mun,
        gdf_cells=gdf_cells,
        df_pred_grade=predicoes_grade_t25,
        focos_parquet_path=focos_parquet_path,
        out_png_path=res_dir / f"mapa_sao_felix_xingu_{mes_max_fogo}.png"
    )

    t_fim = time.time() - t_start
    logger.info(f"=== AVALIAÇÃO EM 2026 CONCLUÍDA EM {t_fim:.1f} SEGUNDOS ===")

if __name__ == "__main__":
    main()
