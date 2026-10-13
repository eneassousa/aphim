"""
Módulo 04 — Captura do Fogo Acima do Normal e Rankings de Anomalia (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Avaliar a capacidade dos métodos e modelos em capturar o fogo excedente (anomalia observada):
   - normal_iτ = focos_media_hist_mes (média histórica do mês de calendário do alvo usando apenas anos anteriores).
   - e_iτ = max(0, y_iτ - normal_iτ) (fogo observado acima da normal climatológica).
2. Parte 1 — Avaliação dos rankings originais de fogo (mesmo ordenamento da Captura@15) sob a métrica:
   - CapturaExcesso@k(τ) = soma de e nos k primeiros ÷ soma de e dos 144 municípios (k = 10 a 30).
3. Parte 2 — Avaliação de rankings de anomalia (ordenando pelo excesso previsto em vez de contagem total):
   - Modelos: excesso_previsto = pred - normal_iτ.
   - Baseline persistencia_anomalia: focos_t - normal_t (excesso do mês de emissão em relação ao mesmo mês de calendário em anos anteriores).
   - Teto oraculo_excesso: ordena pelo próprio e_iτ.
   - CapturaExcesso*@15 = CapturaExcesso@15 do método ÷ CapturaExcesso@15(oraculo_excesso).
4. Comparações estatísticas pareadas na Parte 2:
   - Teste de Wilcoxon pareado (24 meses) contra persistencia_anomalia.
   - Intervalo de confiança de 95% por bootstrap (10.000 reamostragens, semente 42).
   - Correção de Holm nos p-valores dentro da família de comparações dos 6 modelos.
5. Correção de Holm retroativa aplicada à tabela perigo_testes_estatisticos.csv,
   gerando o arquivo perigo_testes_estatisticos_holm.csv sem sobrescrever o original.

Saídas:
- data/processed/resultados/excesso_mensal.csv
- data/processed/resultados/excesso_resumo.csv
- data/processed/resultados/excesso_testes_estatisticos.csv
- data/processed/resultados/perigo_testes_estatisticos_holm.csv
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

logger = setup_logger("processamento_04_excesso")

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """
    Aplica a correção de Holm-Bonferroni (step-down) para controle da taxa de erro familiar (FWER).
    p_adj_(k) = min(1.0, max_{j <= k} ((m - j + 1) * p_(j)))
    """
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

def executar_analise_excesso():
    logger.info("=== INICIANDO MÓDULO 04: CAPTURA DO FOGO ACIMA DO NORMAL ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento dos dados
    mod_path = processed_dir / "aphim_modelagem.parquet"
    base_path = processed_dir / "aphim_base_municipio_mes.parquet"
    preds_path = resultados_dir / "perigo_predicoes_teste.parquet"
    stat_path = resultados_dir / "perigo_testes_estatisticos.csv"

    if not mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{mod_path}' não encontrado.")
    if not base_path.exists():
        raise FileNotFoundError(f"Arquivo '{base_path}' não encontrado.")
    if not preds_path.exists():
        raise FileNotFoundError(f"Arquivo de previsões '{preds_path}' não encontrado.")
    if not stat_path.exists():
        raise FileNotFoundError(f"Arquivo de testes estatísticos '{stat_path}' não encontrado.")

    df_mod = pd.read_parquet(mod_path)
    df_base = pd.read_parquet(base_path)
    df_preds = pd.read_parquet(preds_path)

    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_base["cd_mun"] = df_base["cd_mun"].astype(str).str.zfill(7)
    df_preds["cd_mun"] = df_preds["cd_mun"].astype(str).str.zfill(7)

    # Filtrar conjunto de teste (24 meses alvo: 2024-01 a 2025-12)
    df_test = df_mod[df_mod["conjunto"] == "teste"].copy()
    assert len(df_test) == 3456, f"Esperado 3.456 linhas no conjunto de teste, obtido {len(df_test)}."

    # Incorporar previsões dos modelos
    cols_preds = [
        "pred_glm_A", "pred_glm_C",
        "pred_rf_A", "pred_rf_C",
        "pred_xgb_A", "pred_xgb_C"
    ]
    for c in cols_preds:
        df_test[c] = df_preds[c].values

    # Adicionar score do histórico fixo (2019 a 2023)
    df_treino_hist = df_base[df_base["ano"].between(2019, 2023)]
    focos_hist_fixo = df_treino_hist.groupby("cd_mun")["focos_count"].sum().to_dict()
    df_test["score_hist_fixo"] = df_test["cd_mun"].map(focos_hist_fixo)

    # 2. Definição da Normal e do Excesso Observado (e)
    # normal_iτ = focos_media_hist_mes do mês alvo
    df_test["normal"] = df_test["focos_media_hist_mes"]
    df_test["e"] = np.maximum(0.0, df_test["y"] - df_test["normal"])

    # Validações estritas
    assert (df_test["e"] >= 0.0).all(), "ERRO CRÍTICO: Detectado valor negativo de excesso e!"
    assert not df_test["e"].isna().any(), "ERRO CRÍTICO: Detectado valor NaN em e!"
    logger.info("Validação aprovada: e >= 0 em 100% das 3.456 linhas do teste.")

    # 3. Sumário mensal do excesso observado
    meses_alvo_teste = sorted(df_test["mes_alvo"].unique())
    logger.info("\n=== DISTRIBUIÇÃO MENSAL DO FOGO ACIMA DO NORMAL (TESTE 2024-2025) ===")
    linhas_distribuicao = []
    for mes in meses_alvo_teste:
        sub = df_test[df_test["mes_alvo"] == mes]
        soma_y = int(sub["y"].sum())
        soma_norm = float(sub["normal"].sum())
        soma_e = float(sub["e"].sum())
        n_mun_e = int((sub["e"] > 0).sum())
        linhas_distribuicao.append({
            "mes_alvo": mes,
            "focos_reais": soma_y,
            "focos_normal": soma_norm,
            "focos_excesso": soma_e,
            "pct_excesso_sobre_real": (soma_e / soma_y * 100) if soma_y > 0 else 0.0,
            "mun_com_excesso": n_mun_e
        })

    df_dist = pd.DataFrame(linhas_distribuicao)
    print("\n" + "=" * 90)
    print("                 DISTRIBUIÇÃO MENSAL DO FOGO ACIMA DO NORMAL (TESTE)")
    print("=" * 90)
    print(f"{'Mês Alvo':<10} | {'Focos Reais':<12} | {'Normal Hist.':<13} | {'Excesso (e)':<13} | {'% Excesso':<11} | {'Muns c/ Excesso'}")
    print("-" * 90)
    for _, r in df_dist.iterrows():
        print(f"{r['mes_alvo']:<10} | {r['focos_reais']:<12,d} | {r['focos_normal']:<13,.1f} | {r['focos_excesso']:<13,.1f} | {r['pct_excesso_sobre_real']:<10.1f}% | {r['mun_com_excesso']:<2d} / 144")
    print("=" * 90)
    print(f"TOTAL GERAL: Focos Reais = {df_test['y'].sum():,d} | Normal = {df_test['normal'].sum():,.1f} | Excesso = {df_test['e'].sum():,.1f} ({df_test['e'].sum()/df_test['y'].sum()*100:.1f}%)")
    print("=" * 90 + "\n")

    # 4. Cálculo da normal_t para a baseline persistencia_anomalia (mês de emissão t)
    logger.info("Calculando climatologia da emissão normal_t para o baseline persistencia_anomalia...")
    hist_fire = df_base[["cd_mun", "ano", "mes_num", "focos_count"]].copy()
    hist_dict_t = {}
    for (c, m_num), group in hist_fire.groupby(["cd_mun", "mes_num"]):
        group = group.sort_values("ano")
        anos = group["ano"].values
        counts = group["focos_count"].values
        for target_year in range(2020, 2026):
            mask = anos < target_year
            if mask.any():
                hist_dict_t[(c, m_num, target_year)] = float(np.mean(counts[mask]))
            else:
                hist_dict_t[(c, m_num, target_year)] = np.nan

    df_test["ano_emissao"] = df_test["mes_emissao"].apply(lambda x: int(x[:4]))
    df_test["mes_emissao_num"] = df_test["mes_emissao"].apply(lambda x: int(x[5:7]))
    df_test["normal_t"] = [
        hist_dict_t[(c, mn, y)]
        for c, mn, y in zip(df_test["cd_mun"], df_test["mes_emissao_num"], df_test["ano_emissao"])
    ]
    df_test["score_persistencia_anomalia"] = df_test["focos_t"] - df_test["normal_t"]

    # Excesso previsto para cada modelo (pred - normal)
    df_test["score_glm_A_excesso"] = df_test["pred_glm_A"] - df_test["normal"]
    df_test["score_glm_C_excesso"] = df_test["pred_glm_C"] - df_test["normal"]
    df_test["score_rf_A_excesso"] = df_test["pred_rf_A"] - df_test["normal"]
    df_test["score_rf_C_excesso"] = df_test["pred_rf_C"] - df_test["normal"]
    df_test["score_xgb_A_excesso"] = df_test["pred_xgb_A"] - df_test["normal"]
    df_test["score_xgb_C_excesso"] = df_test["pred_xgb_C"] - df_test["normal"]
    df_test["score_oraculo_excesso"] = df_test["e"]

    k_valores = [10, 15, 20, 25, 30]

    # ==============================================================================
    # 5. PARTE 1 — OS RANKINGS ATUAIS CAPTURAM O EXCESSO?
    # ==============================================================================
    logger.info("Calculando Parte 1: Avaliação dos rankings originais sobre a captura de excesso...")
    metodos_p1 = {
        "oraculo": "y",
        "climatologia_focos": "focos_media_hist_mes",
        "persistencia": "focos_t",
        "historico_fixo": "score_hist_fixo",
        "rf_inpe": "base_rf_inpe_media_t",
        "glm_A": "pred_glm_A",
        "glm_C": "pred_glm_C",
        "rf_A": "pred_rf_A",
        "rf_C": "pred_rf_C",
        "xgb_A": "pred_xgb_A",
        "xgb_C": "pred_xgb_C"
    }

    registros_mensais_p1 = []
    for nome, col_score in metodos_p1.items():
        for mes in meses_alvo_teste:
            sub = df_test[df_test["mes_alvo"] == mes].copy()
            soma_e_mes = float(sub["e"].sum())
            if soma_e_mes == 0.0:
                logger.warning(f"[Parte 1] Mês alvo {mes} possui soma de excesso zero! Registrando NaN.")

            sub_sorted = sub.sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            reg = {
                "parte": "parte_1",
                "tipo_ranking": "ranking_original_focos",
                "metodo": nome,
                "mes_alvo": mes,
                "soma_excesso_mes": soma_e_mes,
                "mun_com_excesso": int((sub["e"] > 0).sum())
            }
            for k in k_valores:
                val_k = float(sub_sorted.iloc[:k]["e"].sum() / soma_e_mes) if soma_e_mes > 0 else np.nan
                reg[f"captura_excesso_{k}"] = val_k
            registros_mensais_p1.append(reg)

    df_mensal_p1 = pd.DataFrame(registros_mensais_p1)

    # CapturaExcesso*@15 da Parte 1 em relação ao oráculo de fogo da Parte 1
    oraculo_p1_caps = df_mensal_p1[df_mensal_p1["metodo"] == "oraculo"].set_index("mes_alvo")["captura_excesso_15"].to_dict()
    df_mensal_p1["captura_excesso_star_15"] = df_mensal_p1.apply(
        lambda r: float(r["captura_excesso_15"] / oraculo_p1_caps[r["mes_alvo"]]) if oraculo_p1_caps.get(r["mes_alvo"], 0) > 0 else np.nan,
        axis=1
    )

    # ==============================================================================
    # 6. PARTE 2 — UM RANKING DE ANOMALIA
    # ==============================================================================
    logger.info("Calculando Parte 2: Avaliação dos rankings de anomalia (excesso previsto)...")
    metodos_p2 = {
        "oraculo_excesso": "score_oraculo_excesso",
        "persistencia_anomalia": "score_persistencia_anomalia",
        "glm_A": "score_glm_A_excesso",
        "glm_C": "score_glm_C_excesso",
        "rf_A": "score_rf_A_excesso",
        "rf_C": "score_rf_C_excesso",
        "xgb_A": "score_xgb_A_excesso",
        "xgb_C": "score_xgb_C_excesso"
    }

    registros_mensais_p2 = []
    for nome, col_score in metodos_p2.items():
        for mes in meses_alvo_teste:
            sub = df_test[df_test["mes_alvo"] == mes].copy()
            soma_e_mes = float(sub["e"].sum())
            if soma_e_mes == 0.0:
                logger.warning(f"[Parte 2] Mês alvo {mes} possui soma de excesso zero! Registrando NaN.")

            sub_sorted = sub.sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            reg = {
                "parte": "parte_2",
                "tipo_ranking": "ranking_anomalia",
                "metodo": nome,
                "mes_alvo": mes,
                "soma_excesso_mes": soma_e_mes,
                "mun_com_excesso": int((sub["e"] > 0).sum())
            }
            for k in k_valores:
                val_k = float(sub_sorted.iloc[:k]["e"].sum() / soma_e_mes) if soma_e_mes > 0 else np.nan
                reg[f"captura_excesso_{k}"] = val_k
            registros_mensais_p2.append(reg)

    df_mensal_p2 = pd.DataFrame(registros_mensais_p2)

    # CapturaExcesso*@15 da Parte 2 em relação ao oraculo_excesso
    oraculo_p2_caps = df_mensal_p2[df_mensal_p2["metodo"] == "oraculo_excesso"].set_index("mes_alvo")["captura_excesso_15"].to_dict()
    df_mensal_p2["captura_excesso_star_15"] = df_mensal_p2.apply(
        lambda r: float(r["captura_excesso_15"] / oraculo_p2_caps[r["mes_alvo"]]) if oraculo_p2_caps.get(r["mes_alvo"], 0) > 0 else np.nan,
        axis=1
    )

    # Unificação de excesso_mensal.csv
    colunas_mensal_final = [
        "parte", "tipo_ranking", "metodo", "mes_alvo",
        "captura_excesso_10", "captura_excesso_15", "captura_excesso_20", "captura_excesso_25", "captura_excesso_30",
        "captura_excesso_star_15", "soma_excesso_mes", "mun_com_excesso"
    ]
    df_mensal_unificado = pd.concat([df_mensal_p1, df_mensal_p2], ignore_index=True)[colunas_mensal_final]
    out_excesso_mensal = resultados_dir / "excesso_mensal.csv"
    df_mensal_unificado.to_csv(out_excesso_mensal, index=False, encoding="utf-8")
    logger.info(f"Resultados mensais salvos em '{out_excesso_mensal}'.")

    # ==============================================================================
    # 7. TABELA RESUMO DE EXCESSO (MÉDIA 24 MESES + ESTAÇÃO SECA)
    # ==============================================================================
    meses_seca = [m for m in meses_alvo_teste if m[5:7] in ["07", "08", "09", "10", "11"]]
    cols_k = [f"captura_excesso_{k}" for k in k_valores] + ["captura_excesso_star_15"]

    # Agregação geral
    df_resumo_total = df_mensal_unificado.groupby(["parte", "tipo_ranking", "metodo"])[cols_k].mean().reset_index()

    # Agregação na estação seca
    df_seca_mensal = df_mensal_unificado[df_mensal_unificado["mes_alvo"].isin(meses_seca)]
    resumo_seca = df_seca_mensal.groupby(["parte", "tipo_ranking", "metodo"]).agg(
        captura_excesso_15_seca=("captura_excesso_15", "mean"),
        captura_excesso_star_15_seca=("captura_excesso_star_15", "mean")
    ).reset_index()

    df_resumo = df_resumo_total.merge(resumo_seca, on=["parte", "tipo_ranking", "metodo"], how="left")

    out_excesso_resumo = resultados_dir / "excesso_resumo.csv"
    df_resumo.to_csv(out_excesso_resumo, index=False, encoding="utf-8")
    logger.info(f"Tabela resumo de excesso salva em '{out_excesso_resumo}'.")

    # ==============================================================================
    # 8. TESTES ESTATÍSTICOS PAREADOS DA PARTE 2 COM CORREÇÃO DE HOLM
    # ==============================================================================
    logger.info("Executando testes pareados de Wilcoxon e Bootstrap (10.000 amostras) na Parte 2...")
    modelos_p2 = ["glm_A", "glm_C", "rf_A", "rf_C", "xgb_A", "xgb_C"]
    baseline_ref_p2 = "persistencia_anomalia"

    sub_ref = df_mensal_p2[df_mensal_p2["metodo"] == baseline_ref_p2].sort_values("mes_alvo")
    c_ref = sub_ref["captura_excesso_15"].values

    rng = np.random.RandomState(42)
    n_boot = 10000
    linhas_testes_p2 = []

    for mod in modelos_p2:
        sub_mod = df_mensal_p2[df_mensal_p2["metodo"] == mod].sort_values("mes_alvo")
        c_mod = sub_mod["captura_excesso_15"].values

        diff = c_mod - c_ref
        diff_media = float(np.mean(diff))

        res_w = wilcoxon(c_mod, c_ref, alternative="two-sided")
        p_val = float(res_w.pvalue)

        boot_idx = rng.choice(len(diff), size=(n_boot, len(diff)), replace=True)
        boot_means = diff[boot_idx].mean(axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        linhas_testes_p2.append({
            "modelo": mod,
            "comparado_com": baseline_ref_p2,
            "diferenca_media": diff_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": p_val
        })

    df_testes_p2 = pd.DataFrame(linhas_testes_p2)
    # Aplicação da correção de Holm dentro da família dos 6 modelos
    df_testes_p2["p_holm"] = holm_bonferroni(df_testes_p2["p_valor"].values)

    out_excesso_testes = resultados_dir / "excesso_testes_estatisticos.csv"
    df_testes_p2.to_csv(out_excesso_testes, index=False, encoding="utf-8")
    logger.info(f"Testes estatísticos de excesso com Holm salvos em '{out_excesso_testes}'.")

    # ==============================================================================
    # 9. CORREÇÃO DE HOLM RETROATIVA NA TABELA perigo_testes_estatisticos.csv
    # ==============================================================================
    logger.info("Aplicando correção de Holm retroativa em perigo_testes_estatisticos.csv...")
    df_perigo_stat = pd.read_csv(stat_path)
    df_perigo_holm = df_perigo_stat.copy()
    df_perigo_holm["p_holm"] = 0.0

    # Aplicação de Holm dentro de cada família de comparação (agrupado por comparado_com)
    for comp_target, grp in df_perigo_holm.groupby("comparado_com"):
        p_adj = holm_bonferroni(grp["p_valor"].values)
        df_perigo_holm.loc[grp.index, "p_holm"] = p_adj

    out_perigo_holm = resultados_dir / "perigo_testes_estatisticos_holm.csv"
    df_perigo_holm.to_csv(out_perigo_holm, index=False, encoding="utf-8")
    logger.info(f"Tabela retroativa perigo_testes_estatisticos_holm.csv salva com sucesso em '{out_perigo_holm}'.")

    # ==============================================================================
    # 10. EXIBIÇÃO FORMATADA NO TERMINAL
    # ==============================================================================
    imprimir_relatorios_excesso(df_resumo, df_testes_p2, df_perigo_holm)
    logger.info("Módulo 04 finalizado com sucesso!")
    return df_resumo, df_testes_p2, df_perigo_holm

def imprimir_relatorios_excesso(
    df_resumo: pd.DataFrame,
    df_testes_p2: pd.DataFrame,
    df_perigo_holm: pd.DataFrame
):
    """Exibe no terminal os relatórios de excesso solicitados."""
    # 1. PARTE 1
    res_p1 = df_resumo[df_resumo["parte"] == "parte_1"].copy()
    ordem_p1 = [
        "oraculo", "persistencia", "rf_A", "xgb_A", "rf_C", "xgb_C",
        "glm_A", "glm_C", "climatologia_focos", "historico_fixo", "rf_inpe"
    ]
    res_p1["_ordem"] = res_p1["metodo"].map(lambda m: ordem_p1.index(m) if m in ordem_p1 else 99)
    res_p1 = res_p1.sort_values("_ordem")

    print("\n" + "=" * 105)
    print("      PARTE 1 — OS RANKINGS ORIGINAIS DE FOCOS CAPTURAM O EXCESSO? (24 MESES DE TESTE)")
    print("=" * 105)
    header_p1 = f"{'Método':<22} | {'CapExcesso@15':<15} | {'CapExcesso*@15':<17} | {'CapExc@15 Seca':<16} | {'CapExc@10':<11} | {'CapExc@30'}"
    print(header_p1)
    print("-" * 105)
    for _, r in res_p1.iterrows():
        m_name = r["metodo"]
        ce15 = f"{r['captura_excesso_15']*100:.2f}%"
        ce_star = f"{r['captura_excesso_star_15']*100:.2f}%"
        ce_seca = f"{r['captura_excesso_15_seca']*100:.2f}%"
        ce10 = f"{r['captura_excesso_10']*100:.2f}%"
        ce30 = f"{r['captura_excesso_30']*100:.2f}%"
        print(f"{m_name:<22} | {ce15:<15} | {ce_star:<17} | {ce_seca:<16} | {ce10:<11} | {ce30}")
    print("=" * 105)

    # 2. PARTE 2
    res_p2 = df_resumo[df_resumo["parte"] == "parte_2"].copy()
    ordem_p2 = [
        "oraculo_excesso", "glm_C", "persistencia_anomalia", "glm_A",
        "xgb_C", "xgb_A", "rf_C", "rf_A"
    ]
    res_p2["_ordem"] = res_p2["metodo"].map(lambda m: ordem_p2.index(m) if m in ordem_p2 else 99)
    res_p2 = res_p2.sort_values("_ordem")

    print("\n" + "=" * 105)
    print("           PARTE 2 — RANKINGS DE ANOMALIA (ORDENADOS PELO EXCESSO PREVISTO)")
    print("=" * 105)
    header_p2 = f"{'Ranking Anomalia':<24} | {'CapExcesso@15':<15} | {'CapExcesso*@15':<17} | {'CapExc@15 Seca':<16} | {'CapExc*@15 Seca'}"
    print(header_p2)
    print("-" * 105)
    for _, r in res_p2.iterrows():
        m_name = r["metodo"]
        ce15 = f"{r['captura_excesso_15']*100:.2f}%"
        ce_star = f"{r['captura_excesso_star_15']*100:.2f}%"
        ce_seca = f"{r['captura_excesso_15_seca']*100:.2f}%"
        ce_star_seca = f"{r['captura_excesso_star_15_seca']*100:.2f}%"
        print(f"{m_name:<24} | {ce15:<15} | {ce_star:<17} | {ce_seca:<16} | {ce_star_seca}")
    print("=" * 105)

    # 3. TESTES ESTATÍSTICOS DA PARTE 2 (WILCOXON & HOLM)
    print("\n" + "=" * 115)
    print("      TESTES ESTATÍSTICOS DA PARTE 2: RANKINGS DE ANOMALIA vs. persistencia_anomalia")
    print("=" * 115)
    header_stat_p2 = f"{'Modelo':<10} | {'Referência':<23} | {'Dif. Média':<12} | {'IC 95% Bootstrap':<25} | {'p-valor':<10} | {'p_holm':<10} | {'Significância'}"
    print(header_stat_p2)
    print("-" * 115)
    for _, r in df_testes_p2.iterrows():
        mod = r["modelo"]
        comp = r["comparado_com"]
        dm = f"{r['diferenca_media']*100:+.2f} p.p."
        ic_str = f"[{r['ic_95_inf']*100:+.2f}%, {r['ic_95_sup']*100:+.2f}%]"
        pval = f"{r['p_valor']:.4f}"
        pholm = f"{r['p_holm']:.4f}"
        sig = "p_holm < 0.05 (*)" if r['p_holm'] < 0.05 else "não sig."
        print(f"{mod:<10} | {comp:<23} | {dm:<12} | {ic_str:<25} | {pval:<10} | {pholm:<10} | {sig}")
    print("=" * 115)

    # 4. TABELA RETROATIVA COM P_HOLM (perigo_testes_estatisticos_holm.csv)
    print("\n" + "=" * 115)
    print("     TESTES DO MÓDULO DE PERIGO COM CORREÇÃO DE HOLM (perigo_testes_estatisticos_holm.csv)")
    print("=" * 115)
    header_stat_perigo = f"{'Modelo':<10} | {'Referência':<20} | {'Dif. Média':<12} | {'IC 95% Bootstrap':<25} | {'p-valor':<10} | {'p_holm':<10} | {'Signif. Holm'}"
    print(header_stat_perigo)
    print("-" * 115)
    for _, r in df_perigo_holm.iterrows():
        mod = r["modelo"]
        comp = r["comparado_com"]
        dm = f"{r['diferenca_media']*100:+.2f} p.p."
        ic_str = f"[{r['ic_95_inf']*100:+.2f}%, {r['ic_95_sup']*100:+.2f}%]"
        pval = f"{r['p_valor']:.4f}"
        pholm = f"{r['p_holm']:.4f}"
        sig = "p_holm < 0.05 (*)" if r['p_holm'] < 0.05 else "não sig."
        print(f"{mod:<10} | {comp:<20} | {dm:<12} | {ic_str:<25} | {pval:<10} | {pholm:<10} | {sig}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    executar_analise_excesso()
