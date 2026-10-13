"""
Módulo 05 — Módulo de Vulnerabilidade (AHP) e Fusão Multicritério ELECTRE Tri-B (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos e Ajustes Metodológicos:
1. Módulo de Vulnerabilidade (AHP):
   - Pesos calculados pelo método do autovetor principal de Saaty a partir de config.yaml.
   - Cálculo da Razão de Consistência (CR < 0,10).
   - Normalização [0, 1] dos critérios de vulnerabilidade com min/max calculados ESTRITAMENTE no treino.
   - Geração do índice de vulnerabilidade V = soma(peso_nivel2 * criterio_normalizado).
2. Fusão Perigo + Vulnerabilidade via ELECTRE Tri-B no Espaço de Percentis (ECDF do Treino):
   - Alternativas: 3.456 observações do teste (144 municípios x 24 meses).
   - Transformação empírica ECDF estritamente calibrada no treino para cada critério:
     * perigo: ECDF do y observado no treino, aplicada às previsões de teste.
     * desmatamento e floresta: ECDF dos valores do treino.
     * renda: ECDF de -renda do treino (-renda_pc, critério de custo).
   - Perfis de fronteira fixados nos percentis b1 = 0,50, b2 = 0,75 e b3 = 0,90.
   - Limiares em unidades de percentil a partir de config.yaml: q = 0,02 e p = 0,05. Sem veto.
   - Procedimento de atribuição pessimista com corte lambda = 0,75.
   - Bateria de 5 testes unitários automatizados (incluindo dominância de perigo e perfis intermediários).
3. Avaliação de Rankings Comparados no Teste:
   - hibrido_B (ordena por classe ELECTRE decrescente e desempata por perigo).
   - vulnerabilidade_isolada (ordena só por V).
   - perigo_isolado (modelo preditivo vencedor).
   - Baselines de referência (climatologia_focos, persistencia, historico_fixo, rf_inpe, oraculo).
4. Métricas e Estatística:
   - Captura@k, Captura*@15, Captura ponderada@15, CapturaExcesso@15, Spearman e Rotatividade.
   - Versão agregada de todas as capturas (focos totais capturados / focos totais no período).
   - Distribuição de classes comparada com a referência do WCAMA.
   - Teste de sanidade em setembro de 2024 (Top 15 com classe, perigo e V).
   - Testes de Wilcoxon pareado (24 meses) + Bootstrap 95% + Correção de Holm para Captura@15 e Captura ponderada@15.
   - Análise de sensibilidade dos pesos (+-20%, 8 cenários).

Saídas (em data/processed/resultados/):
- vulnerabilidade_municipio_ano.csv
- electre_classes_teste.parquet
- hibrido_mensal.csv
- hibrido_resumo.csv
- hibrido_testes_estatisticos.csv
- hibrido_sensibilidade.csv
- capturas_agregadas.csv
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import (
    calcular_metricas_mes,
    adicionar_captura_star_15
)

logger = setup_logger("processamento_05_vulnerabilidade_electre")

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Aplica a correção de Holm-Bonferroni (step-down) para controle do FWER."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    
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

# ==============================================================================
# 1. CÁLCULO DOS PESOS AHP E RAZÃO DE CONSISTÊNCIA
# ==============================================================================

def calcular_ahp(config_mc: Dict[str, Any]) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    logger.info("Calculando pesos AHP em dois níveis a partir de config.yaml...")
    w_n1 = config_mc["nivel_1"]
    w_perigo_n1 = float(w_n1["perigo"])
    w_vuln_n1 = float(w_n1["vulnerabilidade"])

    saaty_info = config_mc["nivel_2_saaty"]
    criterios_vuln = saaty_info["criterios"]
    matriz_saaty = np.array(saaty_info["matriz"], dtype=float)

    n = len(criterios_vuln)
    if matriz_saaty.shape != (n, n):
        raise ValueError(f"Dimensão da matriz de Saaty {matriz_saaty.shape} incompatível com {n} critérios.")

    # Cálculo do autovetor principal
    autovalores, autovetores = np.linalg.eig(matriz_saaty)
    idx_max = np.argmax(np.real(autovalores))
    lambda_max = float(np.real(autovalores[idx_max]))
    autovetor_principal = np.real(autovetores[:, idx_max])
    
    # Normalização dos pesos de nível 2
    w_n2_arr = autovetor_principal / np.sum(autovetor_principal)
    w_n2 = {crit: float(w_n2_arr[i]) for i, crit in enumerate(criterios_vuln)}

    # Razão de consistência (Saaty)
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = ri_table.get(n, 1.49)
    cr = float(ci / ri) if ri > 0 else 0.0

    logger.info(f"AHP Nível 2: lambda_max = {lambda_max:.4f}, CI = {ci:.4f}, CR = {cr:.4f}")
    if cr >= 0.10:
        raise ValueError(f"FALHA DE CONSISTÊNCIA AHP: Razão de Consistência (CR = {cr:.4f}) >= 0,10!")
    logger.info(f"Consistência AHP aprovada (CR = {cr:.4f} < 0,10).")

    # Pesos finais (Nível 1 x Nível 2)
    w_finais = {
        "perigo": w_perigo_n1,
        "desmat_pct_area_prodes": w_vuln_n1 * w_n2["desmat_pct_area_prodes"],
        "pct_floresta": w_vuln_n1 * w_n2["pct_floresta"],
        "renda_pc": w_vuln_n1 * w_n2["renda_pc"]
    }
    soma_w = sum(w_finais.values())
    assert np.isclose(soma_w, 1.0), f"Soma dos pesos finais ({soma_w}) diferente de 1.0!"

    logger.info(f"Pesos AHP Nível 2 (Vulnerabilidade): {w_n2}")
    logger.info(f"Pesos AHP Finais (Fusão): {w_finais}")
    return w_finais, cr, w_n2

# ==============================================================================
# 2. ÍNDICE DE VULNERABILIDADE V
# ==============================================================================

def calcular_indice_vulnerabilidade(
    df_mod: pd.DataFrame,
    w_n2: Dict[str, float],
    resultados_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    logger.info("Normalizando critérios de vulnerabilidade [0, 1] estritamente no treino...")
    df_treino = df_mod[df_mod["conjunto"] == "treino"]

    escalas_vuln = {
        "desmat_pct_area_prodes": (float(df_treino["desmat_pct_area_prodes"].min()), float(df_treino["desmat_pct_area_prodes"].max())),
        "pct_floresta": (float(df_treino["pct_floresta"].min()), float(df_treino["pct_floresta"].max())),
        "renda_pc": (float(df_treino["renda_pc"].min()), float(df_treino["renda_pc"].max()))
    }

    df_out = df_mod.copy()
    for col, (vmin, vmax) in escalas_vuln.items():
        norm_col = np.clip((df_out[col] - vmin) / (vmax - vmin), 0.0, 1.0)
        if col == "renda_pc":
            df_out["renda_norm_inv"] = 1.0 - norm_col
        else:
            df_out[f"{col}_norm"] = norm_col

    df_out["V"] = (
        w_n2["desmat_pct_area_prodes"] * df_out["desmat_pct_area_prodes_norm"] +
        w_n2["pct_floresta"] * df_out["pct_floresta_norm"] +
        w_n2["renda_pc"] * df_out["renda_norm_inv"]
    )

    df_out["ano"] = df_out["mes_emissao"].str[:4].astype(int)
    cols_vuln_ano = [
        "cd_mun", "ano",
        "desmat_pct_area_prodes_norm", "pct_floresta_norm", "renda_norm_inv", "V"
    ]
    df_vuln_ano = df_out[cols_vuln_ano].drop_duplicates(subset=["cd_mun", "ano"]).sort_values(["ano", "cd_mun"]).reset_index(drop=True)
    out_csv = resultados_dir / "vulnerabilidade_municipio_ano.csv"
    df_vuln_ano.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Índice de vulnerabilidade exportado para '{out_csv}' ({len(df_vuln_ano)} linhas).")

    return df_out, escalas_vuln

# ==============================================================================
# 3. MOTOR ELECTRE TRI-B NO ESPAÇO DE PERCENTIS (ECDF)
# ==============================================================================

def concordancia_parcial(ga: float, gb: float, q: float, p: float) -> float:
    """
    Função de concordância parcial para critério com limiares de indiferença (q) e preferência (p).
    Sentido: maximização (quanto maior ga em relação a gb, maior a concordância de ga S gb).
    """
    diff = ga - gb
    if diff >= -q:
        return 1.0
    elif diff <= -p:
        return 0.0
    else:
        return float((diff + p) / (p - q))

def classificar_alternativa_electre_tri_b(
    crit_vals: Dict[str, float],
    weights: Dict[str, float],
    perfis: Dict[str, Tuple[float, float, float]],
    q: float,
    p: float,
    classes_info: List[Tuple[int, str, int]],
    lambda_cut: float = 0.75
) -> Tuple[str, int]:
    """
    Atribuição pessimista do ELECTRE Tri-B no espaço de percentis:
    Compara a alternativa sucessivamente de cima para baixo com os perfis b_3, b_2, b_1.
    A alternativa é atribuída à classe correspondente ao primeiro perfil b_h que ela sobreclassifica (sigma >= lambda).
    Caso não sobreclassifique nenhum perfil, vai para a classe mais baixa (Baixo).
    """
    for h_idx, cl_nome, cl_ord in classes_info:
        sigma = 0.0
        for c_nome, w in weights.items():
            ga = crit_vals[c_nome]
            gb = perfis[c_nome][h_idx]
            c_val = concordancia_parcial(ga, gb, q, p)
            sigma += w * c_val
        if sigma >= lambda_cut:
            return cl_nome, cl_ord
    return "Baixo", 1

def executar_testes_unitarios_electre(
    weights: Dict[str, float],
    perfis: Dict[str, Tuple[float, float, float]],
    q: float,
    p: float,
    classes_info: List[Tuple[int, str, int]],
    lambda_cut: float = 0.75
):
    logger.info("Executando bateria de testes unitários do ELECTRE Tri-B no espaço de percentis...")
    
    # Teste 1: alternativa estritamente acima de b3 em todos os critérios deve ir para Crítico
    alt_max = {c: 0.99 for c in weights}
    cl_max, ord_max = classificar_alternativa_electre_tri_b(alt_max, weights, perfis, q, p, classes_info, lambda_cut)
    if cl_max != "Critico" or ord_max != 4:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO 1: Esperado 'Critico', obtido '{cl_max}'.")

    # Teste 2: alternativa estritamente abaixo de b1 - p em todos os critérios deve ir para Baixo
    alt_min = {c: 0.10 for c in weights}
    cl_min, ord_min = classificar_alternativa_electre_tri_b(alt_min, weights, perfis, q, p, classes_info, lambda_cut)
    if cl_min != "Baixo" or ord_min != 1:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO 2: Esperado 'Baixo', obtido '{cl_min}'.")

    # Teste 3: perigo no percentil 0.99 e demais critérios em 0.95 -> Crítico
    alt_3 = {"perigo": 0.99, "desmat_pct_area_prodes": 0.95, "pct_floresta": 0.95, "renda_pc": 0.95}
    cl_3, ord_3 = classificar_alternativa_electre_tri_b(alt_3, weights, perfis, q, p, classes_info, lambda_cut)
    if cl_3 != "Critico" or ord_3 != 4:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO 3: Esperado 'Critico', obtido '{cl_3}'.")

    # Teste 4: perigo em 0.10 e demais em 0.95 -> NÃO pode ser Crítico (pois perigo pesa 0.5 e corte é 0.75)
    alt_4 = {"perigo": 0.10, "desmat_pct_area_prodes": 0.95, "pct_floresta": 0.95, "renda_pc": 0.95}
    cl_4, ord_4 = classificar_alternativa_electre_tri_b(alt_4, weights, perfis, q, p, classes_info, lambda_cut)
    if cl_4 == "Critico":
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO 4: Alternativa com perigo baixo foi indevidamente classificada como 'Critico'.")

    # Teste 5: todos os critérios em 0.60 -> Moderado (acima de b1=0.50 e abaixo de b2=0.75)
    alt_5 = {c: 0.60 for c in weights}
    cl_5, ord_5 = classificar_alternativa_electre_tri_b(alt_5, weights, perfis, q, p, classes_info, lambda_cut)
    if cl_5 != "Moderado" or ord_5 != 2:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO 5: Esperado 'Moderado', obtido '{cl_5}'.")

    logger.info("Bateria de 5 testes unitários do ELECTRE Tri-B aprovada com 100% de sucesso!")

# ==============================================================================
# 4. AVALIAÇÃO DE RANKINGS E MÉTRICAS COMPARADAS
# ==============================================================================

def executar_etapa_6():
    logger.info("=== INICIANDO ETAPA 6: MÓDULO DE VULNERABILIDADE E FUSÃO ELECTRE TRI-B (VERSÃO ECDF) ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    config_mc = config["multicriterio"]

    # 1. Pesos AHP
    w_finais, cr_ahp, w_n2 = calcular_ahp(config_mc)

    # 2. Carregar dados de modelagem e predições de teste
    mod_v2_path = processed_dir / "aphim_modelagem_v2.parquet"
    mod_path = mod_v2_path if mod_v2_path.exists() else processed_dir / "aphim_modelagem.parquet"
    base_raw_path = processed_dir / "aphim_base_municipio_mes.parquet"
    df_mod = pd.read_parquet(mod_path)
    df_base_raw = pd.read_parquet(base_raw_path)

    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_base_raw["cd_mun"] = df_base_raw["cd_mun"].astype(str).str.zfill(7)

    # Predições de perigo: prioriza vencedor de 03b se existir, senão rf_A
    preds_v2_path = resultados_dir / "features_v2_predicoes_teste.parquet"
    preds_e5_path = resultados_dir / "perigo_predicoes_teste.parquet"

    if preds_v2_path.exists():
        logger.info(f"Carregando predições do vencedor da Rodada 2 de atributos de '{preds_v2_path}'...")
        df_p = pd.read_parquet(preds_v2_path)
        col_perigo = "pred_vencedor"
        nome_modelo_perigo = "perigo_vencedor_v2"
        df_preds_teste = df_p[["cd_mun", "mes_alvo", col_perigo]].copy()
    elif preds_e5_path.exists():
        logger.info(f"Carregando predições de '{preds_e5_path}' (fallback rf_A)...")
        df_p = pd.read_parquet(preds_e5_path)
        col_perigo = "pred_rf_A"
        nome_modelo_perigo = "perigo_rf_A"
        df_preds_teste = df_p[["cd_mun", "mes_alvo", col_perigo]].copy()
    else:
        raise FileNotFoundError("Nenhum arquivo de predições de teste encontrado.")

    df_preds_teste["cd_mun"] = df_preds_teste["cd_mun"].astype(str).str.zfill(7)

    # 3. Índice de Vulnerabilidade V
    df_mod, _ = calcular_indice_vulnerabilidade(df_mod, w_n2, resultados_dir)

    # 4. Construção das ECDFs Estritamente no Treino
    logger.info("Construindo ECDFs empíricas no treino para todos os critérios...")
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    df_ancora = df_mod[df_mod["mes_emissao"] == "2023-11"].copy() # mes_alvo = 2023-12

    # Incorporar perigo previsto
    df_teste = df_teste.merge(df_preds_teste, on=["cd_mun", "mes_alvo"], how="left")
    df_ancora = df_ancora.merge(df_preds_teste, on=["cd_mun", "mes_alvo"], how="left")
    if df_ancora[col_perigo].isnull().any():
        df_ancora[col_perigo] = df_ancora[col_perigo].fillna(df_ancora["focos_media_hist_mes"])

    # Score correto do histórico fixo (2019-2023)
    logger.info("Calculando score exato do histórico fixo (2019-2023 por município)...")
    df_hist_fixo_treino = df_base_raw[df_base_raw["ano"].between(2019, 2023)]
    focos_hist_fixo_dict = df_hist_fixo_treino.groupby("cd_mun")["focos_count"].sum().to_dict()
    df_teste["score_hist_fixo"] = df_teste["cd_mun"].map(focos_hist_fixo_dict)
    df_ancora["score_hist_fixo"] = df_ancora["cd_mun"].map(focos_hist_fixo_dict)

    # ECDFs do treino
    train_arrays = {
        "perigo": np.sort(df_treino["y"].values.astype(float)),
        "desmat_pct_area_prodes": np.sort(df_treino["desmat_pct_area_prodes"].values.astype(float)),
        "pct_floresta": np.sort(df_treino["pct_floresta"].values.astype(float)),
        "renda_pc": np.sort(-df_treino["renda_pc"].values.astype(float)) # -renda
    }

    def criar_ecdf(sorted_arr: np.ndarray):
        n = len(sorted_arr)
        return lambda v: np.searchsorted(sorted_arr, v, side="right") / float(n)

    ecdfs = {k: criar_ecdf(arr) for k, arr in train_arrays.items()}

    # Parâmetros ELECTRE no espaço de percentis
    quantis_cfg = config_mc["electre"]["quantis_perfis"]
    limiar_q = float(config_mc["electre"]["limiar_q_percentil"])
    limiar_p = float(config_mc["electre"]["limiar_p_percentil"])
    lambda_cut = float(config_mc["electre"]["corte_lambda"])

    # Perfis de fronteira fixados no espaço de percentil
    perfis = {
        k: (float(quantis_cfg[0]), float(quantis_cfg[1]), float(quantis_cfg[2]))
        for k in train_arrays
    }

    classes_info = [(2, "Critico", 4), (1, "Alto", 3), (0, "Moderado", 2)]

    logger.info(f"Limiares de percentil carregados: q = {limiar_q:.4f}, p = {limiar_p:.4f}, lambda = {lambda_cut:.2f}")
    logger.info(f"Perfis de fronteira em percentil: b1 = {quantis_cfg[0]}, b2 = {quantis_cfg[1]}, b3 = {quantis_cfg[2]}")

    # Executar bateria de testes unitários
    executar_testes_unitarios_electre(w_finais, perfis, limiar_q, limiar_p, classes_info, lambda_cut)

    # 5. Classificação do Teste via ELECTRE Tri-B
    logger.info("Classificando as 3.456 alternativas de teste pelo ELECTRE Tri-B (espaço de percentis)...")
    
    def classificar_df(df_target):
        res = []
        for _, r in df_target.iterrows():
            c_vals = {
                "perigo": float(ecdfs["perigo"](r[col_perigo])),
                "desmat_pct_area_prodes": float(ecdfs["desmat_pct_area_prodes"](r["desmat_pct_area_prodes"])),
                "pct_floresta": float(ecdfs["pct_floresta"](r["pct_floresta"])),
                "renda_pc": float(ecdfs["renda_pc"](-r["renda_pc"]))
            }
            cl_nome, cl_ord = classificar_alternativa_electre_tri_b(
                c_vals, w_finais, perfis, limiar_q, limiar_p, classes_info, lambda_cut
            )
            res.append((cl_nome, cl_ord))
        return [x[0] for x in res], [x[1] for x in res]

    df_teste["classe"], df_teste["classe_ordem"] = classificar_df(df_teste)
    df_ancora["classe"], df_ancora["classe_ordem"] = classificar_df(df_ancora)

    # Exportar electre_classes_teste.parquet
    out_parquet = resultados_dir / "electre_classes_teste.parquet"
    df_electre_export = df_teste[["cd_mun", "mes_alvo", "classe", "V", col_perigo]].rename(columns={col_perigo: "perigo"})
    df_electre_export.to_parquet(out_parquet, index=False)
    logger.info(f"Classes do ELECTRE exportadas com sucesso para '{out_parquet}'.")

    # Distribuição por classe no teste
    dist_classes = df_teste["classe"].value_counts(normalize=True) * 100.0
    ref_wcama = {"Baixo": 15.3, "Moderado": 73.6, "Alto": 10.4, "Critico": 0.7}

    # 6. Avaliação dos Rankings Comparados
    logger.info("Avaliando métricas dos rankings comparados mês a mês...")
    df_teste["score_hibrido_B"] = df_teste["classe_ordem"] * 1e9 + df_teste[col_perigo]
    df_ancora["score_hibrido_B"] = df_ancora["classe_ordem"] * 1e9 + df_ancora[col_perigo]

    meses_alvo = sorted(df_teste["mes_alvo"].unique())
    focos_totais_periodo = float(df_teste["y"].sum())
    meses_seca = [m for m in meses_alvo if m[5:7] in ["07", "08", "09", "10", "11"]]

    def avaliar_metodo(df_t, col_score, nome_metodo, df_anc=None):
        prev_top15 = None
        if df_anc is not None:
            anc_s = df_anc.sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            prev_top15 = set(anc_s.iloc[:15]["cd_mun"])

        linhas_m = []
        focos_top15_acum = 0.0
        for mes in meses_alvo:
            sub = df_t[df_t["mes_alvo"] == mes].copy()
            met, prev_top15 = calcular_metricas_mes(sub, col_score, prev_top15=prev_top15, k_valores=(10, 15, 20, 25, 30))
            sub_s = sub.sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            
            # CapturaExcesso@15
            e_total = np.maximum(0.0, sub["y"] - sub["focos_media_hist_mes"]).sum()
            e_top15 = np.maximum(0.0, sub_s.iloc[:15]["y"] - sub_s.iloc[:15]["focos_media_hist_mes"]).sum()
            met["captura_excesso_15"] = float(e_top15 / e_total) if e_total > 0 else np.nan
            met["mes_alvo"] = mes
            met["metodo"] = nome_metodo
            linhas_m.append(met)

            focos_top15_acum += float(sub_s.iloc[:15]["y"].sum())

        df_res_m = pd.DataFrame(linhas_m)
        agregada_15 = float(focos_top15_acum / focos_totais_periodo)
        return df_res_m, agregada_15

    metodos_avaliar = [
        ("hibrido_B", "score_hibrido_B"),
        ("vulnerabilidade_isolada", "V"),
        ("perigo_isolado", col_perigo),
        ("climatologia_focos", "focos_media_hist_mes"),
        ("persistencia", "focos_t"),
        ("oraculo", "y")
    ]

    dfs_mensais = []
    for nome_m, col_s in metodos_avaliar:
        df_m, _ = avaliar_metodo(df_teste, col_s, nome_m, df_ancora)
        dfs_mensais.append(df_m)

    df_hibrido_mensal = pd.concat(dfs_mensais, ignore_index=True)
    df_hibrido_mensal = adicionar_captura_star_15(df_hibrido_mensal, col_metodo="metodo", nome_oraculo="oraculo")

    # Montagem da tabela resumo
    resumos_lista = []
    for nome_m, col_s in metodos_avaliar:
        sub_m = df_hibrido_mensal[df_hibrido_mensal["metodo"] == nome_m]
        sub_sec = sub_m[sub_m["mes_alvo"].isin(meses_seca)]
        
        sub_agr = 0.0
        for mes in meses_alvo:
            sub = df_teste[df_teste["mes_alvo"] == mes].sort_values(by=[col_s, "cd_mun"], ascending=[False, True])
            sub_agr += float(sub.iloc[:15]["y"].sum())
        agr_val = float(sub_agr / focos_totais_periodo)

        resumos_lista.append({
            "metodo": nome_m,
            "captura_10": sub_m["captura_10"].mean(),
            "captura_15": sub_m["captura_15"].mean(),
            "captura_20": sub_m["captura_20"].mean(),
            "captura_25": sub_m["captura_25"].mean(),
            "captura_30": sub_m["captura_30"].mean(),
            "captura_star_15": sub_m["captura_star_15"].mean(),
            "captura_pond_15": sub_m["captura_pond_15"].mean(),
            "captura_excesso_15": sub_m["captura_excesso_15"].mean(),
            "captura_agregada_15": agr_val,
            "captura_15_seca": sub_sec["captura_15"].mean(),
            "spearman": sub_m["spearman"].mean(),
            "rotatividade": sub_m["rotatividade"].mean()
        })

    df_hibrido_resumo = pd.DataFrame(resumos_lista)

    out_mensal = resultados_dir / "hibrido_mensal.csv"
    out_resumo = resultados_dir / "hibrido_resumo.csv"
    df_hibrido_mensal.to_csv(out_mensal, index=False, encoding="utf-8")
    df_hibrido_resumo.to_csv(out_resumo, index=False, encoding="utf-8")
    logger.info(f"Tabelas hibrido_mensal.csv e hibrido_resumo.csv salvas com sucesso.")

    # 7. Testes Estatísticos Pareados (Wilcoxon + Bootstrap + Holm)
    logger.info("Executando testes pareados de Wilcoxon e Bootstrap 95% com correção de Holm...")
    hib_caps15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "hibrido_B"].set_index("mes_alvo")["captura_15"].loc[meses_alvo].values
    hib_pond15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "hibrido_B"].set_index("mes_alvo")["captura_pond_15"].loc[meses_alvo].values

    per_caps15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "perigo_isolado"].set_index("mes_alvo")["captura_15"].loc[meses_alvo].values
    per_pond15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "perigo_isolado"].set_index("mes_alvo")["captura_pond_15"].loc[meses_alvo].values

    clima_caps15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "climatologia_focos"].set_index("mes_alvo")["captura_15"].loc[meses_alvo].values
    clima_pond15 = df_hibrido_mensal[df_hibrido_mensal["metodo"] == "climatologia_focos"].set_index("mes_alvo")["captura_pond_15"].loc[meses_alvo].values

    comparacoes = [
        ("hibrido_B", "perigo_isolado", "captura_15", hib_caps15, per_caps15),
        ("hibrido_B", "climatologia_focos", "captura_15", hib_caps15, clima_caps15),
        ("hibrido_B", "perigo_isolado", "captura_pond_15", hib_pond15, per_pond15),
        ("hibrido_B", "climatologia_focos", "captura_pond_15", hib_pond15, clima_pond15)
    ]

    rng_boot = np.random.RandomState(42)
    testes_res = []
    p_brutos = []

    for mod, comp, metrica, s_hib, s_comp in comparacoes:
        diffs = s_hib - s_comp
        dif_media = float(np.mean(diffs))

        indices = rng_boot.randint(0, len(diffs), size=(10000, len(diffs)))
        boot_means = np.mean(diffs[indices], axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        res_w = wilcoxon(s_hib, s_comp, alternative="two-sided")
        pval = float(res_w.pvalue)
        p_brutos.append(pval)

        testes_res.append({
            "modelo": mod,
            "comparado_com": comp,
            "metrica": metrica,
            "diferenca_media": dif_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": pval
        })

    p_holm = holm_bonferroni(np.array(p_brutos))
    for i, p_h in enumerate(p_holm):
        testes_res[i]["p_holm"] = float(p_h)

    df_testes_stat = pd.DataFrame(testes_res)
    out_testes = resultados_dir / "hibrido_testes_estatisticos.csv"
    df_testes_stat.to_csv(out_testes, index=False, encoding="utf-8")
    logger.info(f"Testes estatísticos salvos em '{out_testes}'.")

    # 8. Análise de Sensibilidade dos Pesos (+-20%, 8 cenários) no Espaço de Percentis
    logger.info("Executando análise de sensibilidade dos pesos (+-20%, 8 cenários) no espaço de percentis...")
    top15_base_mensal = {}
    for mes in meses_alvo:
        sub = df_teste[df_teste["mes_alvo"] == mes].sort_values(by=["score_hibrido_B", "cd_mun"], ascending=[False, True])
        top15_base_mensal[mes] = set(sub.iloc[:15]["cd_mun"])

    criterios_nomes = ["perigo", "desmat_pct_area_prodes", "pct_floresta", "renda_pc"]
    linhas_sensibilidade = []

    for c_var in criterios_nomes:
        for var_label, fator in [("+20%", 1.2), ("-20%", 0.8)]:
            w_pert = w_finais.copy()
            w_pert[c_var] = w_finais[c_var] * fator
            soma_pert = sum(w_pert.values())
            w_pert_norm = {k: v / soma_pert for k, v in w_pert.items()}

            def classificar_pert(row):
                c_vals = {
                    "perigo": float(ecdfs["perigo"](row[col_perigo])),
                    "desmat_pct_area_prodes": float(ecdfs["desmat_pct_area_prodes"](row["desmat_pct_area_prodes"])),
                    "pct_floresta": float(ecdfs["pct_floresta"](row["pct_floresta"])),
                    "renda_pc": float(ecdfs["renda_pc"](-row["renda_pc"]))
                }
                _, cl_ord = classificar_alternativa_electre_tri_b(
                    c_vals, w_pert_norm, perfis, limiar_q, limiar_p, classes_info, lambda_cut
                )
                return cl_ord

            ords_pert = [classificar_pert(r) for _, r in df_teste.iterrows()]
            df_teste_pert = df_teste.copy()
            df_teste_pert["score_pert"] = np.array(ords_pert) * 1e9 + df_teste_pert[col_perigo]

            caps15_pert = []
            cpond15_pert = []
            coincidencias = []
            for mes in meses_alvo:
                sub = df_teste_pert[df_teste_pert["mes_alvo"] == mes].sort_values(by=["score_pert", "cd_mun"], ascending=[False, True])
                top15_muns = set(sub.iloc[:15]["cd_mun"])
                
                tot_y = sub["y"].sum()
                tot_pond = (sub["y"] * sub["pct_floresta"] / 100.0).sum()
                soma15 = sub.iloc[:15]["y"].sum()
                soma_pond15 = (sub.iloc[:15]["y"] * sub.iloc[:15]["pct_floresta"] / 100.0).sum()

                caps15_pert.append(soma15 / tot_y if tot_y > 0 else 0.0)
                cpond15_pert.append(soma_pond15 / tot_pond if tot_pond > 0 else 0.0)
                coincidencias.append(len(top15_muns.intersection(top15_base_mensal[mes])))

            linhas_sensibilidade.append({
                "criterio_variado": c_var,
                "variacao": var_label,
                "peso_perigo": w_pert_norm["perigo"],
                "peso_desmat": w_pert_norm["desmat_pct_area_prodes"],
                "peso_floresta": w_pert_norm["pct_floresta"],
                "peso_renda": w_pert_norm["renda_pc"],
                "captura_15": float(np.mean(caps15_pert)),
                "captura_pond_15": float(np.mean(cpond15_pert)),
                "coincidencia_top15_media": float(np.mean(coincidencias))
            })

    df_sensibilidade = pd.DataFrame(linhas_sensibilidade)
    out_sens = resultados_dir / "hibrido_sensibilidade.csv"
    df_sensibilidade.to_csv(out_sens, index=False, encoding="utf-8")
    logger.info(f"Análise de sensibilidade salva em '{out_sens}'.")

    # 9. Tabela Consolidada de Capturas Agregadas (Todos os Métodos Corrigida)
    logger.info("Gerando tabela unificada de capturas agregadas corrigida para todos os métodos...")
    metodos_todos = []
    
    # Modelos da Etapa 5
    if preds_e5_path.exists():
        df_pe5 = pd.read_parquet(preds_e5_path)
        colunas_preds_e5 = [c for c in df_pe5.columns if c.startswith("pred_")]
        for c in colunas_preds_e5:
            nome_m = c.replace("pred_", "")
            df_t_temp = df_teste.copy()
            df_t_temp[c] = df_pe5[c].values
            metodos_todos.append((nome_m, "modelo_etapa5", c, df_t_temp))

    # Vencedor da Etapa 03b
    if preds_v2_path.exists():
        metodos_todos.append(("xgb_A_v2_sel", "modelo_vencedor_v2", "pred_vencedor", df_teste))

    # Baselines
    metodos_todos.append(("oraculo", "baseline", "y", df_teste))
    metodos_todos.append(("climatologia_focos", "baseline", "focos_media_hist_mes", df_teste))
    metodos_todos.append(("persistencia", "baseline", "focos_t", df_teste))
    # Correção do historico_fixo: utiliza a contagem total de focos em 2019-2023
    metodos_todos.append(("historico_fixo", "baseline", "score_hist_fixo", df_teste))
    metodos_todos.append(("rf_inpe", "baseline", "base_rf_inpe_media_t", df_teste))

    # Métodos da Etapa 6
    metodos_todos.append(("hibrido_B", "fusao_multicriterio", "score_hibrido_B", df_teste))
    metodos_todos.append(("vulnerabilidade_isolada", "vulnerabilidade_isolada", "V", df_teste))

    linhas_agregadas = []
    vistos = set()
    for nome_m, tipo_m, col_s, df_data in metodos_todos:
        if nome_m in vistos:
            continue
        vistos.add(nome_m)

        focos_capt = 0.0
        caps_mensais = []
        for mes in meses_alvo:
            sub = df_data[df_data["mes_alvo"] == mes].sort_values(by=[col_s, "cd_mun"], ascending=[False, True])
            soma15 = float(sub.iloc[:15]["y"].sum())
            tot_y = float(sub["y"].sum())
            focos_capt += soma15
            caps_mensais.append(soma15 / tot_y if tot_y > 0 else 0.0)

        linhas_agregadas.append({
            "metodo": nome_m,
            "tipo": tipo_m,
            "focos_totais_teste": int(focos_totais_periodo),
            "focos_capturados_top15": int(focos_capt),
            "captura_agregada_15": float(focos_capt / focos_totais_periodo),
            "captura_media_mensal_15": float(np.mean(caps_mensais))
        })

    df_agregadas = pd.DataFrame(linhas_agregadas).sort_values("captura_agregada_15", ascending=False).reset_index(drop=True)
    out_agregadas = resultados_dir / "capturas_agregadas.csv"
    df_agregadas.to_csv(out_agregadas, index=False, encoding="utf-8")
    logger.info(f"Tabela capturas_agregadas.csv salva com sucesso em '{out_agregadas}'.")

    # 10. Teste de Sanidade em Setembro de 2024
    sub_set24 = df_teste[df_teste["mes_alvo"] == "2024-09"].sort_values(
        by=["score_hibrido_B", "cd_mun"], ascending=[False, True]
    ).reset_index(drop=True)
    cols_sanidade = ["cd_mun", "nm_mun", "classe", col_perigo, "y", "V"]
    df_sanidade_set24 = sub_set24.iloc[:15][cols_sanidade].rename(columns={col_perigo: "perigo_previsto", "y": "focos_reais"})

    # Exibição final dos resultados solicitados
    print("\n" + "=" * 90)
    print("        ETAPA 6 — MÓDULO DE VULNERABILIDADE E FUSÃO ELECTRE TRI-B CONCLUÍDOS")
    print("=" * 90)
    print("\n1. DISTRIBUIÇÃO POR CLASSE DO HÍBRIDO B vs WCAMA:")
    for cl in ["Baixo", "Moderado", "Alto", "Critico"]:
        pct_teste = dist_classes.get(cl, 0.0)
        pct_wcama = ref_wcama.get(cl, 0.0)
        print(f"{cl:<10}: Teste APHIM = {pct_teste:>6.2f}% | WCAMA Publicado = {pct_wcama:>6.2f}%")

    print("\n2. TESTE DE SANIDADE EM SETEMBRO DE 2024 (TOP 15 DO HÍBRIDO B):")
    print(df_sanidade_set24.to_string(index=False))

    print("\n3. TABELA-RESUMO: HÍBRIDO B, VULNERABILIDADE E REFERÊNCIAS:")
    print(df_hibrido_resumo.to_string(index=False))

    print("\n4. TESTES ESTATÍSTICOS PAREADOS (WILCOXON + BOOTSTRAP + HOLM):")
    print(df_testes_stat.to_string(index=False))

    print("\n5. ANÁLISE DE SENSIBILIDADE DOS PESOS (+-20%):")
    print(df_sensibilidade.to_string(index=False))

    print("\n6. TABELA CONSOLIDADA DE CAPTURAS AGREGADAS CORRIGIDA:")
    print(df_agregadas.to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    executar_etapa_6()
