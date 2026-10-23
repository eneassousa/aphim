"""
Módulo 10b — Reprodução das Predições de Teste do Modelo G1 (DETER Grade 5 km)
Dissertação de Mestrado: Predição e Priorização Espacial na Grade de 5 km do Pará.

Objetivo:
- Refazer a previsão do G1 no conjunto de teste (2024-01 a 2025-12) com os mesmos hiperparâmetros
  escolhidos na validação (deter_validacao_grade.csv) e mesma semente (42).
- Validar se a CapturaÁrea@10% mensal bate rigorosamente com deter_grade_mensal.csv (diferença <= 0,01 ponto).
- Salvar data/processed_convlstm/g1_predicoes_teste.parquet com as colunas:
  cell_id, cd_mun, mes_emissao, mes_alvo, y, pred_g1.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_10b_g1_predicoes_teste")

FEATURES_CELULA_BASE = [
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

FEATURES_CELULA_DETER = [
    "deter_desmat_cel_3m",
    "deter_desmat_cel_12m",
    "deter_degrad_cel_3m",
    "deter_degrad_cel_12m",
    "deter_desmat_viz_12m"
]

def calcular_captura_area_mes(
    df_mes: pd.DataFrame,
    col_score: str,
    top15_muns: List[str],
    pct_area_k: float = 10.0
) -> float:
    """Calcula a CapturaÁrea@k% para um único mês dentro dos 15 municípios."""
    sub_top15 = df_mes[df_mes["cd_mun"].isin(top15_muns)]
    focos_capturados = 0.0
    focos_top15 = 0.0

    for _, grp in sub_top15.groupby("cd_mun"):
        f_mun = grp["y"].sum()
        focos_top15 += f_mun

        grp_sort = grp.sort_values(by=[col_score, "cell_id"], ascending=[False, True])
        cell_areas = grp_sort["area_km2"].values
        cum_areas = np.cumsum(cell_areas)
        total_mun_area = cum_areas[-1] if len(cum_areas) > 0 else 0.0
        target_area = (pct_area_k / 100.0) * total_mun_area

        mask = (cum_areas - cell_areas) < target_area
        focos_capturados += grp_sort.iloc[mask]["y"].sum()

    cap = (focos_capturados / focos_top15) if focos_top15 > 0 else 0.0
    return float(cap)

def main():
    logger.info("=== REPRODUZINDO PREDIÇÕES DE TESTE DO MODELO G1 (PARTE 0) ===")
    
    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = Path("data/processed_convlstm")
    out_res_dir = out_dir / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem da grade
    grade_mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    grade_det_path = processed_deter_dir / "grade_5000m_deter.parquet"
    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"

    logger.info("Carregando grade_5000m_modelagem.parquet...")
    df_mod = pd.read_parquet(grade_mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    logger.info("Carregando grade_5000m_deter.parquet...")
    df_det = pd.read_parquet(grade_det_path)

    logger.info("Mesclando atributos DETER...")
    df_mod = df_mod.merge(df_det, on=["cell_id", "mes_emissao"], how="left")
    for col in FEATURES_CELULA_DETER:
        df_mod[col] = df_mod[col].fillna(0.0).astype(np.float32)

    # Carregar áreas de célula
    gdf_cells = gpd.read_file(gpkg_path)
    cell_area_map = dict(zip(gdf_cells["cell_id"].astype(np.int32), gdf_cells["area_km2_dentro_pa"].astype(float)))
    df_mod["area_km2"] = df_mod["cell_id"].map(cell_area_map).astype(np.float32)

    # 2. Carregar hiperparâmetros de G1
    val_csv_path = processed_deter_dir / "resultados" / "deter_validacao_grade.csv"
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Arquivo '{val_csv_path}' não encontrado.")
    df_val = pd.read_csv(val_csv_path)
    row_g1 = df_val[df_val["configuracao"] == "G1"].iloc[0]
    params_g1 = json.loads(row_g1["hiperparametros"])
    logger.info(f"Hiperparâmetros de G1 carregados: {params_g1}")

    # 3. Treinar modelo G1 em todo o treino
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    features_g1 = FEATURES_CELULA_BASE + FEATURES_CELULA_DETER

    logger.info(f"Treinando modelo XGBoost G1 em {len(df_treino)} células...")
    m_final = xgb.XGBRegressor(
        max_depth=params_g1["max_depth"],
        learning_rate=params_g1["learning_rate"],
        n_estimators=params_g1["n_estimators"],
        min_child_weight=params_g1["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        random_state=42, n_jobs=-1
    )
    m_final.fit(df_treino[features_g1], df_treino["y"])
    logger.info("Modelo G1 treinado. Gerando predições de teste...")

    preds_g1 = np.maximum(0.0, m_final.predict(df_teste[features_g1]))
    df_teste["pred_g1"] = preds_g1

    # 4. Salvar g1_predicoes_teste.parquet
    out_parquet = out_dir / "g1_predicoes_teste.parquet"
    cols_salvar = ["cell_id", "cd_mun", "mes_emissao", "mes_alvo", "y", "pred_g1"]
    df_pred_out = df_teste[cols_salvar].copy()
    df_pred_out.to_parquet(out_parquet, index=False)
    logger.info(f"Predições de teste do G1 salvas em '{out_parquet}' ({len(df_pred_out)} linhas).")

    # 5. Conferir CapturaÁrea@10% mensal contra deter_grade_mensal.csv
    l1_path = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
    df_pred_l1 = pd.read_parquet(l1_path)
    df_pred_l1["cd_mun"] = df_pred_l1["cd_mun"].astype(str).str.zfill(7)

    ref_mensal_path = processed_deter_dir / "resultados" / "deter_grade_mensal.csv"
    df_ref_mensal = pd.read_csv(ref_mensal_path)
    df_ref_g1 = df_ref_mensal[df_ref_mensal["metodo"] == "G1_xgb_deter"].set_index("mes_alvo")["captura_area_10"]

    meses_alvo = sorted(df_teste["mes_alvo"].unique())
    logger.info(f"Calculando CapturaÁrea@10% para os {len(meses_alvo)} meses de teste...")

    comparacoes = []
    max_diff = 0.0

    for m_alvo in meses_alvo:
        sub_l1 = df_pred_l1[df_pred_l1["mes_alvo"] == m_alvo].sort_values(
            by=["pred_vencedor", "cd_mun"], ascending=[False, True]
        )
        top15_muns = sub_l1["cd_mun"].head(15).tolist()

        df_m = df_teste[df_teste["mes_alvo"] == m_alvo]
        cap_calc = calcular_captura_area_mes(df_m, "pred_g1", top15_muns, pct_area_k=10.0)
        cap_ref = float(df_ref_g1.loc[m_alvo])
        diff = abs(cap_calc - cap_ref)
        max_diff = max(max_diff, diff)

        comparacoes.append({
            "mes_alvo": m_alvo,
            "cap10_reproduzido": cap_calc,
            "cap10_referencia": cap_ref,
            "diferenca_pp": diff * 100.0
        })

    df_comp = pd.DataFrame(comparacoes)
    print("\n" + "=" * 80)
    print("      CONFERÊNCIA DA CAPTURAÁREA@10% DO G1 NO TESTE (PARTE 0)")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)

    logger.info(f"Diferença máxima observada: {max_diff*100:.6f} p.p. (limite estrito: 0,0100 p.p.)")
    if max_diff > 0.0001:
        msg = f"ERRO NA PARTE 0: A diferença máxima de {max_diff*100:.6f} p.p. excede o limite de 0,01 p.p."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("PARTE 0 CONCLUÍDA COM SUCESSO: G1 reproduzido com perfeição em todos os 24 meses de teste!")

if __name__ == "__main__":
    main()
