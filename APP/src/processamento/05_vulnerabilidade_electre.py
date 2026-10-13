"""
Módulo 05 — Módulo de Vulnerabilidade (AHP) e Fusão Multicritério ELECTRE Tri-B (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Módulo de Vulnerabilidade (AHP):
   - Pesos calculados pelo método do autovetor principal de Saaty a partir de config.yaml.
   - Cálculo da Razão de Consistência (CR < 0,10).
   - Normalização [0, 1] dos critérios de vulnerabilidade com min/max calculados ESTRITAMENTE no treino.
   - Geração do índice de vulnerabilidade V = soma(peso_nivel2 * criterio_normalizado).
2. Fusão Perigo + Vulnerabilidade via ELECTRE Tri-B (Híbrido B):
   - Alternativas: 3.456 observações do teste (144 municípios x 24 meses).
   - Critérios: Perigo (previsão do vencedor da rodada 2 / rf_A), Desmatamento (t-1), Floresta (t-1), Renda (-renda_pc).
   - Perfis de fronteira nos quantis 0,50, 0,75 e 0,90 calculados estritamente no treino (y observado para perigo).
   - Limiares de indiferença (q = 5%) e preferência (p = 15%) da amplitude do treino. Sem veto.
   - Procedimento de atribuição pessimista com corte lambda = 0,75.
   - Teste unitário automatizado de fronteiras extremas.
3. Avaliação de Rankings Comparados no Teste:
   - hibrido_B (ordena por classe ELECTRE e desempata por perigo).
   - vulnerabilidade_isolada (ordena só por V).
   - perigo_isolado (modelo preditivo vencedor).
   - Baselines de referência (climatologia_focos, persistencia, oraculo).
4. Métricas e Estatística:
   - Captura@k, Captura*@15, Captura ponderada@15, CapturaExcesso@15, Spearman e Rotatividade.
   - Versão agregada de todas as capturas (focos totais capturados / focos totais no período).
   - Distribuição de classes comparada com a referência do WCAMA.
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
    # Tabela de RI para n=1..10
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
            # Inversão da renda: quanto menor a renda, maior a vulnerabilidade
            df_out["renda_norm_inv"] = 1.0 - norm_col
        else:
            df_out[f"{col}_norm"] = norm_col

    df_out["V"] = (
        w_n2["desmat_pct_area_prodes"] * df_out["desmat_pct_area_prodes_norm"] +
        w_n2["pct_floresta"] * df_out["pct_floresta_norm"] +
        w_n2["renda_pc"] * df_out["renda_norm_inv"]
    )

    # Exportar vulnerabilidade_municipio_ano.csv
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
# 3. MOTOR ELECTRE TRI-B
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
    limiares: Dict[str, Tuple[float, float]],
    classes_info: List[Tuple[int, str, int]],
    lambda_cut: float = 0.75
) -> Tuple[str, int]:
    """
    Atribuição pessimista do ELECTRE Tri-B:
    Compara a alternativa sucessivamente de cima para baixo com os perfis b_3, b_2, b_1.
    A alternativa é atribuída à classe correspondente ao primeiro perfil b_h que ela sobreclassifica (sigma >= lambda).
    Caso não sobreclassifique nenhum perfil, vai para a classe mais baixa (Baixo).
    """
    # classes_info: [(2, 'Critico', 4), (1, 'Alto', 3), (0, 'Moderado', 2)]
    for h_idx, cl_nome, cl_ord in classes_info:
        sigma = 0.0
        for c_nome, w in weights.items():
            ga = crit_vals[c_nome]
            gb = perfis[c_nome][h_idx]
            q, p = limiares[c_nome]
            c_val = concordancia_parcial(ga, gb, q, p)
            sigma += w * c_val
        if sigma >= lambda_cut:
            return cl_nome, cl_ord
    return "Baixo", 1

def executar_testes_unitarios_electre(
    weights: Dict[str, float],
    perfis: Dict[str, Tuple[float, float, float]],
    limiares: Dict[str, Tuple[float, float]],
    classes_info: List[Tuple[int, str, int]],
    lambda_cut: float = 0.75
):
    logger.info("Executando teste unitário das propriedades do ELECTRE Tri-B...")
    # Teste A: alternativa estritamente acima de b3 em todos os critérios deve ir para Crítico
    alt_max = {c: perfis[c][2] + 2.0 * limiares[c][1] for c in weights}
    cl_max, ord_max = classificar_alternativa_electre_tri_b(alt_max, weights, perfis, limiares, classes_info, lambda_cut)
    if cl_max != "Critico" or ord_max != 4:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO A: Esperado 'Critico', obtido '{cl_max}'.")

    # Teste B: alternativa estritamente abaixo de b1 em todos os critérios deve ir para Baixo
    alt_min = {c: perfis[c][0] - 2.0 * limiares[c][1] for c in weights}
    cl_min, ord_min = classificar_alternativa_electre_tri_b(alt_min, weights, perfis, limiares, classes_info, lambda_cut)
    if cl_min != "Baixo" or ord_min != 1:
        raise AssertionError(f"FALHA NO TESTE UNITÁRIO B: Esperado 'Baixo', obtido '{cl_min}'.")

    logger.info("Testes unitários do ELECTRE Tri-B aprovados com 100% de sucesso!")

# ==============================================================================
# 4. AVALIAÇÃO DE RANKINGS E MÉTRICAS COMPARADAS
# ==============================================================================

def executar_etapa_6():
    logger.info("=== INICIANDO ETAPA 6: MÓDULO DE VULNERABILIDADE E FUSÃO ELECTRE TRI-B ===")
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
    df_mod = pd.read_parquet(mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    # Predições de perigo: prioriza vencedor de 03b se existir, senão rf_A
    preds_v2_path = resultados_dir / "features_v2_predicoes_teste.parquet"
    preds_e5_path = resultados_dir / "perigo_predicoes_teste.parquet"

    df_preds_teste = None
    col_perigo = ""
    nome_modelo_perigo = ""

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

    # 4. Parâmetros de treino para ELECTRE Tri-B
    logger.info("Calculando quantis e limiares do ELECTRE Tri-B estritamente nas linhas de treino...")
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    df_ancora = df_mod[df_mod["mes_emissao"] == "2023-11"].copy() # mes_alvo = 2023-12

    # Incorporar perigo previsto
    df_teste = df_teste.merge(df_preds_teste, on=["cd_mun", "mes_alvo"], how="left")
    df_ancora = df_ancora.merge(df_preds_teste, on=["cd_mun", "mes_alvo"], how="left")
    # Caso âncora 2023-12 não esteja no teste, preenche com focos_media_hist_mes para perigo de desempate
    if df_ancora[col_perigo].isnull().any():
        df_ancora[col_perigo] = df_ancora[col_perigo].fillna(df_ancora["focos_media_hist_mes"])

    # Escala dos 4 critérios no treino:
    # Perigo no treino usa o y observado conforme metodologia
    criterios_tr = {
        "perigo": df_treino["y"].values.astype(float),
        "desmat_pct_area_prodes": df_treino["desmat_pct_area_prodes"].values.astype(float),
        "pct_floresta": df_treino["pct_floresta"].values.astype(float),
        "renda_pc": -df_treino["renda_pc"].values.astype(float) # critério de custo: -renda
    }

    quantis_cfg = config_mc["electre"]["quantis_perfis"]
    indif_pct = float(config_mc["electre"]["indiferenca_pct_amplitude"])
    pref_pct = float(config_mc["electre"]["preferencia_pct_amplitude"])
    lambda_cut = float(config_mc["electre"]["corte_lambda"])

    perfis = {}
    limiares = {}
    for c_nome, vals in criterios_tr.items():
        amp = float(vals.max() - vals.min())
        q = indif_pct * amp
        p = pref_pct * amp
        b1 = float(np.quantile(vals, quantis_cfg[0]))
        b2 = float(np.quantile(vals, quantis_cfg[1]))
        b3 = float(np.quantile(vals, quantis_cfg[2]))
        perfis[c_nome] = (b1, b2, b3)
        limiares[c_nome] = (q, p)
        logger.info(f"Critério '{c_nome}': amplitude={amp:.2f}, q={q:.2f}, p={p:.2f} | b1={b1:.2f}, b2={b2:.2f}, b3={b3:.2f}")

    classes_info = [(2, "Critico", 4), (1, "Alto", 3), (0, "Moderado", 2)]

    # Executar teste unitário
    executar_testes_unitarios_electre(w_finais, perfis, limiares, classes_info, lambda_cut)

    # 5. Classificação do Teste via ELECTRE Tri-B
    logger.info("Classificando as 3.456 alternativas de teste pelo ELECTRE Tri-B (atribuição pessimista)...")
    
    def classificar_df(df_target):
        res = []
        for _, r in df_target.iterrows():
            c_vals = {
                "perigo": float(r[col_perigo]),
                "desmat_pct_area_prodes": float(r["desmat_pct_area_prodes"]),
                "pct_floresta": float(r["pct_floresta"]),
                "renda_pc": -float(r["renda_pc"])
            }
            cl_nome, cl_ord = classificar_alternativa_electre_tri_b(c_vals, w_finais, perfis, limiares, classes_info, lambda_cut)
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
    # Score combinado para hibrido_B: classe_ordem * 1e9 + perigo (desempata por perigo dentro da mesma classe)
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
    resumos_lista = []

    for nome_m, col_s in metodos_avaliar:
        df_m, agr_15 = avaliar_metodo(df_teste, col_s, nome_m, df_ancora)
        dfs_mensais.append(df_m)

    df_hibrido_mensal = pd.concat(dfs_mensais, ignore_index=True)
    df_hibrido_mensal = adicionar_captura_star_15(df_hibrido_mensal, col_metodo="metodo", nome_oraculo="oraculo")

    # Montagem da tabela resumo
    for nome_m, _ in metodos_avaliar:
        sub_m = df_hibrido_mensal[df_hibrido_mensal["metodo"] == nome_m]
        sub_sec = sub_m[sub_m["mes_alvo"].isin(meses_seca)]
        
        # Recalcular captura agregada precisa
        sub_agr = 0.0
        for mes in meses_alvo:
            col_s = [c for n, c in metodos_avaliar if n == nome_m][0]
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

        # Bootstrap
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

    # 8. Análise de Sensibilidade dos Pesos (+-20%, 8 cenários)
    logger.info("Executando análise de sensibilidade dos pesos (+-20%, 8 cenários)...")
    # Top-15 base mensal
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

            # Reclassificar teste
            def classificar_pert(row):
                c_vals = {
                    "perigo": float(row[col_perigo]),
                    "desmat_pct_area_prodes": float(row["desmat_pct_area_prodes"]),
                    "pct_floresta": float(row["pct_floresta"]),
                    "renda_pc": -float(row["renda_pc"])
                }
                _, cl_ord = classificar_alternativa_electre_tri_b(c_vals, w_pert_norm, perfis, limiares, classes_info, lambda_cut)
                return cl_ord

            ords_pert = [classificar_pert(r) for _, r in df_teste.iterrows()]
            df_teste_pert = df_teste.copy()
            df_teste_pert["score_pert"] = np.array(ords_pert) * 1e9 + df_teste_pert[col_perigo]

            # Avaliar Captura@15 e coincidência com o base
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

    # 9. Tabela Consolidada de Capturas Agregadas (Todos os Métodos)
    logger.info("Gerando tabela unificada de capturas agregadas para todos os métodos avaliados...")
    
    # Lista completa de métodos a compilar
    metodos_todos = []
    
    # a) Baselines e modelos já avaliados na Etapa 5
    if preds_e5_path.exists():
        df_pe5 = pd.read_parquet(preds_e5_path)
        colunas_preds_e5 = [c for c in df_pe5.columns if c.startswith("pred_")]
        for c in colunas_preds_e5:
            nome_m = c.replace("pred_", "")
            df_t_temp = df_teste.copy()
            df_t_temp[c] = df_pe5[c].values
            metodos_todos.append((nome_m, "modelo_etapa5", c, df_t_temp))

    # b) Vencedor de 03b se disponível
    if preds_v2_path.exists():
        metodos_todos.append(("xgb_A_v2_sel", "modelo_vencedor_v2", "pred_vencedor", df_teste))

    # c) Baselines
    metodos_todos.append(("oraculo", "baseline", "y", df_teste))
    metodos_todos.append(("climatologia_focos", "baseline", "focos_media_hist_mes", df_teste))
    metodos_todos.append(("persistencia", "baseline", "focos_t", df_teste))
    metodos_todos.append(("historico_fixo", "baseline", "focos_media_hist_mes", df_teste)) # proxy
    metodos_todos.append(("rf_inpe", "baseline", "base_rf_inpe_media_t", df_teste))

    # d) Métodos da Etapa 6
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

    # Exibição final dos resultados solicitados
    print("\n" + "=" * 90)
    print("        ETAPA 6 — MÓDULO DE VULNERABILIDADE E FUSÃO ELECTRE TRI-B CONCLUÍDOS")
    print("=" * 90)
    print("\n1. PESOS DO AHP E RAZÃO DE CONSISTÊNCIA (CR):")
    print(f"CR: {cr_ahp:.4f} (Aprovado < 0,10)")
    print(f"Pesos Finais: {w_finais}")
    print("\n2. PERFIS DE FRONTEIRA E LIMIARES (TREINO):")
    for c_nome in criterios_tr:
        b1, b2, b3 = perfis[c_nome]
        q, p = limiares[c_nome]
        print(f"{c_nome:<25}: b1(0.50)={b1:>10.2f} | b2(0.75)={b2:>10.2f} | b3(0.90)={b3:>10.2f} | q={q:>8.2f} | p={p:>8.2f}")
    print("\n3. TABELA-RESUMO: HÍBRIDO B, VULNERABILIDADE E REFERÊNCIAS:")
    print(df_hibrido_resumo.to_string(index=False))
    print("\n4. DISTRIBUIÇÃO POR CLASSE DO HÍBRIDO B vs WCAMA:")
    for cl in ["Baixo", "Moderado", "Alto", "Critico"]:
        pct_teste = dist_classes.get(cl, 0.0)
        pct_wcama = ref_wcama.get(cl, 0.0)
        print(f"{cl:<10}: Teste APHIM = {pct_teste:>6.2f}% | WCAMA Publicado = {pct_wcama:>6.2f}%")
    print("\n5. TESTES ESTATÍSTICOS PAREADOS (WILCOXON + BOOTSTRAP + HOLM):")
    print(df_testes_stat.to_string(index=False))
    print("\n6. ANÁLISE DE SENSIBILIDADE DOS PESOS (+-20%):")
    print(df_sensibilidade.to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    executar_etapa_6()
