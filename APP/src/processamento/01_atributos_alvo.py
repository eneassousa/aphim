"""
Módulo 01 — Engenharia de Atributos, Variável Alvo e Partição Cronológica (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Regra Metodológica Central:
- No fim do mês t (mês de emissão), o sistema gera a priorização para o mês t+1 (mês alvo).
- Cada linha da tabela de modelagem representa o par (município, t).
- Atributos preditivos utilizam ESTRITAMENTE informações disponíveis até o fim de t (defasagem temporal estrita, sem vazamento).
- Variável alvo (y): número de focos de calor observados no mês t+1.
- t = 2025-12 não possui alvo em 2026-01 e é descartado da modelagem.

Partição Cronológica:
- aquecimento: t in [2019-01, 2019-12] (1.728 linhas) - cálculo de defasagens e acumulados; não entra no treino.
- treino: t in [2020-01, 2023-11] (6.768 linhas) - alvos t+1 de 2020-02 a 2023-12.
- teste: t in [2023-12, 2025-11] (3.456 linhas) - alvos t+1 de 2024-01 a 2025-12.
Total de linhas: 144 municípios x 83 meses = 11.952 linhas.

Saídas:
- data/processed/aphim_modelagem.parquet
- data/processed/dicionario_modelagem.md
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_01_atributos_alvo")

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_next_month(mes_str: str) -> str:
    """Calcula o mês t+1 no formato AAAA-MM."""
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    if mes == 12:
        return f"{ano + 1:04d}-01"
    return f"{ano:04d}-{mes + 1:02d}"

def get_prev_month(mes_str: str, steps: int = 1) -> str:
    """Calcula o mês t-steps no formato AAAA-MM."""
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    for _ in range(steps):
        if mes == 1:
            ano -= 1
            mes = 12
        else:
            mes -= 1
    return f"{ano:04d}-{mes:02d}"

def classificar_conjunto(mes_emissao: str) -> str:
    """Define a partição temporal cronológica estrita da linha (município, t)."""
    if "2019-01" <= mes_emissao <= "2019-12":
        return "aquecimento"
    elif "2020-01" <= mes_emissao <= "2023-11":
        return "treino"
    elif "2023-12" <= mes_emissao <= "2025-11":
        return "teste"
    raise ValueError(f"Mês de emissão {mes_emissao} fora do escopo de modelagem.")

def construir_tabela_modelagem(processed_dir: Path) -> pd.DataFrame:
    """
    Executa a montagem determinística de atributos, alvo e conjuntos cronológicos.
    """
    logger.info("Carregando bases processadas de entrada...")
    base_path = processed_dir / "aphim_base_municipio_mes.parquet"
    clima_path = processed_dir / "clima_municipio_mes.parquet"
    normal_path = processed_dir / "clima_normal_1991_2020.parquet"

    if not base_path.exists():
        raise FileNotFoundError(f"Base consolidada '{base_path}' não encontrada.")
    if not clima_path.exists():
        raise FileNotFoundError(f"Base de clima '{clima_path}' não encontrada.")
    if not normal_path.exists():
        raise FileNotFoundError(f"Base de normais '{normal_path}' não encontrada.")

    df_base = pd.read_parquet(base_path)
    df_clima = pd.read_parquet(clima_path)
    df_normal = pd.read_parquet(normal_path)

    # Padronização de tipos de identificadores
    df_base["cd_mun"] = df_base["cd_mun"].astype(str).str.zfill(7)
    df_base["mes"] = df_base["mes"].astype(str)
    df_clima["cd_mun"] = df_clima["cd_mun"].astype(str).str.zfill(7)
    df_clima["mes"] = df_clima["mes"].astype(str)
    df_normal["cd_mun"] = df_normal["cd_mun"].astype(str).str.zfill(7)
    df_normal["mes_calendario"] = df_normal["mes_calendario"].astype(int)

    logger.info("Construindo tabelas auxiliares e dicionários de busca rápida...")
    # 1. Lookups de Focos por (cd_mun, mes)
    focos_lookup = df_base.set_index(["cd_mun", "mes"])["focos_count"].to_dict()

    # 2. Lookups de Clima Acumulado (prec_3m) usando clima_municipio_mes (que contém 2018 para o início de 2019)
    df_clima = df_clima.sort_values(["cd_mun", "mes"]).reset_index(drop=True)
    df_clima["prec_t1"] = df_clima.groupby("cd_mun")["prec_mm"].shift(1)
    df_clima["prec_t2"] = df_clima.groupby("cd_mun")["prec_mm"].shift(2)
    df_clima["prec_3m"] = df_clima["prec_mm"] + df_clima["prec_t1"] + df_clima["prec_t2"]
    prec3m_lookup = df_clima.set_index(["cd_mun", "mes"])["prec_3m"].to_dict()

    # 3. Lookups de Normais Climatológicas 1991-2020 por (cd_mun, mes_calendario)
    normal_prec = df_normal.set_index(["cd_mun", "mes_calendario"])["prec_mm"].to_dict()
    normal_t2m = df_normal.set_index(["cd_mun", "mes_calendario"])["t2m_media_c"].to_dict()
    normal_ur = df_normal.set_index(["cd_mun", "mes_calendario"])["ur_pct"].to_dict()
    normal_umid = df_normal.set_index(["cd_mun", "mes_calendario"])["umid_solo"].to_dict()
    normal_lst = df_normal.set_index(["cd_mun", "mes_calendario"])["lst_c"].to_dict()

    # 4. Lookup de Média Histórica de Focos no mesmo mês do calendário de mes_alvo (somente anos < ano_alvo)
    # Garante estritamente que nenhum ano >= ano(mes_alvo) é considerado.
    logger.info("Pré-computando climatologia histórica de focos por município e mês de calendário...")
    hist_fire = df_base[["cd_mun", "ano", "mes_num", "focos_count"]].copy()
    hist_dict = {}
    for (c, m_num), group in hist_fire.groupby(["cd_mun", "mes_num"]):
        group = group.sort_values("ano")
        anos = group["ano"].values
        counts = group["focos_count"].values
        for target_year in range(2019, 2026):
            mask = anos < target_year
            if mask.any():
                hist_dict[(c, m_num, target_year)] = float(np.mean(counts[mask]))
            else:
                hist_dict[(c, m_num, target_year)] = np.nan

    logger.info("Filtrando e estruturando a grade de modelagem (83 meses: 2019-01 a 2025-11)...")
    # Descartar t = 2025-12 pois não tem alvo em 2026-01
    df_mod = df_base[df_base["mes"] != "2025-12"].copy()
    df_mod = df_mod.rename(columns={"mes": "mes_emissao"})
    df_mod["mes_alvo"] = df_mod["mes_emissao"].apply(get_next_month)

    # Identificação do Alvo y (focos em t+1)
    df_mod["y"] = [focos_lookup.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_alvo"])]
    # Garantir tipo inteiro no alvo onde não nulo
    df_mod["y"] = df_mod["y"].astype(int)

    # Conjunto cronológico
    df_mod["conjunto"] = df_mod["mes_emissao"].apply(classificar_conjunto)

    # 1. Atributos de Histórico de Focos
    logger.info("Calculando atributos de histórico de focos e densidades espaciais...")
    df_mod["focos_t"] = df_mod["focos_count"]
    df_mod["focos_t1"] = [focos_lookup.get((c, get_prev_month(m, 1)), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao"])]
    df_mod["focos_t2"] = [focos_lookup.get((c, get_prev_month(m, 2)), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao"])]
    df_mod["focos_mm3"] = (df_mod["focos_t"] + df_mod["focos_t1"] + df_mod["focos_t2"]) / 3.0

    # focos em mes_alvo menos 12 meses = t - 11
    df_mod["focos_mesmo_mes_ano_ant"] = [focos_lookup.get((c, get_prev_month(m, 11)), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao"])]

    # focos_media_hist_mes
    df_mod["mes_alvo_num"] = df_mod["mes_alvo"].apply(lambda x: int(x[5:7]))
    df_mod["mes_alvo_ano"] = df_mod["mes_alvo"].apply(lambda x: int(x[:4]))
    df_mod["focos_media_hist_mes"] = [hist_dict.get((c, mn, y), np.nan) 
                                      for c, mn, y in zip(df_mod["cd_mun"], df_mod["mes_alvo_num"], df_mod["mes_alvo_ano"])]

    # Densidades (focos / area_km2)
    df_mod["dens_t"] = df_mod["focos_t"] / df_mod["area_km2"]
    df_mod["dens_t1"] = df_mod["focos_t1"] / df_mod["area_km2"]
    df_mod["dens_t2"] = df_mod["focos_t2"] / df_mod["area_km2"]
    df_mod["dens_mm3"] = df_mod["focos_mm3"] / df_mod["area_km2"]
    df_mod["dens_mesmo_mes_ano_ant"] = df_mod["focos_mesmo_mes_ano_ant"] / df_mod["area_km2"]
    df_mod["dens_media_hist_mes"] = df_mod["focos_media_hist_mes"] / df_mod["area_km2"]

    # 2. Clima em t (já presente em df_mod): prec_mm, t2m_max_c, ur_pct, dpv_kpa, umid_solo, lst_c, vento_ms

    # 3. Anomalias climáticas em t
    logger.info("Calculando anomalias climáticas em relação à normal 1991-2020...")
    df_mod["mes_emissao_num"] = df_mod["mes_emissao"].apply(lambda x: int(x[5:7]))
    df_mod["norm_prec_t"] = [normal_prec.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]
    df_mod["norm_t2m_t"] = [normal_t2m.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]
    df_mod["norm_ur_t"] = [normal_ur.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]
    df_mod["norm_umid_t"] = [normal_umid.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]
    df_mod["norm_lst_t"] = [normal_lst.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]

    df_mod["anom_prec_mm"] = df_mod["prec_mm"] - df_mod["norm_prec_t"]
    df_mod["anom_t2m_media_c"] = df_mod["t2m_media_c"] - df_mod["norm_t2m_t"]
    df_mod["anom_ur_pct"] = df_mod["ur_pct"] - df_mod["norm_ur_t"]
    df_mod["anom_umid_solo"] = df_mod["umid_solo"] - df_mod["norm_umid_t"]
    df_mod["anom_lst_c"] = df_mod["lst_c"] - df_mod["norm_lst_t"]

    # 4. Acumulados de 3 meses
    logger.info("Calculando acumulado de precipitação de 3 meses e sua respectiva anomalia...")
    df_mod["prec_3m"] = [prec3m_lookup.get((c, m), np.nan) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao"])]

    def soma_normal_prec_3m(c_mun, m_num):
        m_t = m_num
        m_t1 = 12 if m_t == 1 else m_t - 1
        m_t2 = 12 if m_t1 == 1 else m_t1 - 1
        return normal_prec[(c_mun, m_t)] + normal_prec[(c_mun, m_t1)] + normal_prec[(c_mun, m_t2)]

    df_mod["norm_prec_3m"] = [soma_normal_prec_3m(c, m) for c, m in zip(df_mod["cd_mun"], df_mod["mes_emissao_num"])]
    df_mod["anom_prec_3m"] = df_mod["prec_3m"] - df_mod["norm_prec_3m"]

    # 5. Sazonalidade do Mês Alvo (t+1)
    logger.info("Calculando codificação senoidal/cossenoidal do mês alvo...")
    df_mod["mes_sin"] = np.sin(2.0 * np.pi * df_mod["mes_alvo_num"] / 12.0)
    df_mod["mes_cos"] = np.cos(2.0 * np.pi * df_mod["mes_alvo_num"] / 12.0)

    # 6. Estruturais (da linha t, onde defasagem anual t-1 já foi aplicada):
    # area_km2, pct_floresta, pct_pastagem, pct_agricultura, pct_solo_exposto,
    # desmat_km2_prodes, desmat_pct_area_prodes, renda_pc

    # 7. Baseline: rf_inpe_media_t mantida sob prefixo base_
    df_mod["base_rf_inpe_media_t"] = df_mod["rf_inpe_media"]

    # Seleção canônica ordenada das colunas finais
    colunas_finais = [
        # Identificação e Partição
        "cd_mun", "nm_mun", "mes_emissao", "mes_alvo", "conjunto",
        # Variável Alvo
        "y",
        # Baseline de Referência (não entra como atributo nos modelos)
        "base_rf_inpe_media_t",
        # Atributos de Histórico de Focos
        "focos_t", "focos_t1", "focos_t2", "focos_mm3", "focos_mesmo_mes_ano_ant", "focos_media_hist_mes",
        "dens_t", "dens_t1", "dens_t2", "dens_mm3", "dens_mesmo_mes_ano_ant", "dens_media_hist_mes",
        # Atributos de Clima em t
        "prec_mm", "t2m_max_c", "ur_pct", "dpv_kpa", "umid_solo", "lst_c", "vento_ms",
        # Atributos de Anomalias Climáticas
        "anom_prec_mm", "anom_t2m_media_c", "anom_ur_pct", "anom_umid_solo", "anom_lst_c",
        # Atributos de Acumulados
        "prec_3m", "anom_prec_3m",
        # Atributos de Sazonalidade
        "mes_sin", "mes_cos",
        # Atributos Estruturais (uso restrito ao híbrido C)
        "area_km2", "pct_floresta", "pct_pastagem", "pct_agricultura", "pct_solo_exposto",
        "desmat_km2_prodes", "desmat_pct_area_prodes", "renda_pc"
    ]

    df_final = df_mod[colunas_finais].sort_values(["mes_emissao", "cd_mun"]).reset_index(drop=True)
    return df_final

def validar_tabela_modelagem(df: pd.DataFrame, processed_dir: Path):
    """
    Validações estritas da etapa 4:
    1. Linhas: 144 x 83 = 11.952 (aquecimento=1.728, treino=6.768, teste=3.456);
    2. Sem duplicatas de (cd_mun, mes_emissao);
    3. Sem nulos nos atributos de treino e teste (nulos só no aquecimento);
    4. Soma de y no teste igual a 492.267 (414.486 em 2024 + 77.781 em 2025);
    5. Teste automático de vazamento temporal em 50 linhas aleatórias.
    """
    logger.info("=== INICIANDO VALIDAÇÕES ESTRITAS DA TABELA DE MODELAGEM ===")

    # 1. Contagens de Linhas
    n_linhas = len(df)
    if n_linhas != 11952:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Esperado 11.952 linhas no total, obtido {n_linhas}.")
    
    contagens = df["conjunto"].value_counts().to_dict()
    logger.info(f"Contagem por conjunto: {contagens}")
    if contagens.get("aquecimento", 0) != 1728:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Aquecimento com {contagens.get('aquecimento')} linhas (esperado 1.728).")
    if contagens.get("treino", 0) != 6768:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Treino com {contagens.get('treino')} linhas (esperado 6.768).")
    if contagens.get("teste", 0) != 3456:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Teste com {contagens.get('teste')} linhas (esperado 3.456).")
    logger.info("Validação 1 aprovada: exatamente 11.952 linhas (aquecimento: 1.728, treino: 6.768, teste: 3.456).")

    # 2. Duplicatas
    dups = df.duplicated(subset=["cd_mun", "mes_emissao"]).sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO 2: Encontradas {dups} duplicatas de (cd_mun, mes_emissao).")
    logger.info("Validação 2 aprovada: zero duplicatas de (cd_mun, mes_emissao).")

    # 3. Integridade de Nulos
    cols_atributos = [
        "focos_t", "focos_t1", "focos_t2", "focos_mm3", "focos_mesmo_mes_ano_ant", "focos_media_hist_mes",
        "dens_t", "dens_t1", "dens_t2", "dens_mm3", "dens_mesmo_mes_ano_ant", "dens_media_hist_mes",
        "prec_mm", "t2m_max_c", "ur_pct", "dpv_kpa", "umid_solo", "lst_c", "vento_ms",
        "anom_prec_mm", "anom_t2m_media_c", "anom_ur_pct", "anom_umid_solo", "anom_lst_c",
        "prec_3m", "anom_prec_3m",
        "mes_sin", "mes_cos",
        "area_km2", "pct_floresta", "pct_pastagem", "pct_agricultura", "pct_solo_exposto",
        "desmat_km2_prodes", "desmat_pct_area_prodes", "renda_pc"
    ]
    df_treino_teste = df[df["conjunto"].isin(["treino", "teste"])]
    nulos_treino_teste = df_treino_teste[cols_atributos].isnull().sum()
    nulos_proibidos = nulos_treino_teste.sum()
    if nulos_proibidos > 0:
        cols_problematicas = nulos_treino_teste[nulos_treino_teste > 0].to_dict()
        raise ValueError(f"FALHA NA VALIDAÇÃO 3: Nulos encontrados em atributos de treino/teste: {cols_problematicas}")
    logger.info("Validação 3 aprovada: zero nulos nos 36 atributos das partições de treino e teste.")

    # 4. Soma de y no conjunto de teste
    df_teste = df[df["conjunto"] == "teste"]
    soma_y_teste = int(df_teste["y"].sum())
    esperado_y_teste = 492267
    if soma_y_teste != esperado_y_teste:
        raise ValueError(f"FALHA NA VALIDAÇÃO 4: Soma de y no teste = {soma_y_teste:,} (esperado {esperado_y_teste:,}).")
    logger.info(f"Validação 4 aprovada: soma de y no teste idêntica a 2024+2025 ({soma_y_teste:,} focos).")

    # 5. Teste automático de vazamento temporal
    logger.info("Executando teste automático de vazamento temporal em 50 linhas aleatórias...")
    df_raw = pd.read_parquet(processed_dir / "aphim_base_municipio_mes.parquet")
    df_raw["cd_mun"] = df_raw["cd_mun"].astype(str).str.zfill(7)
    df_raw["mes"] = df_raw["mes"].astype(str)

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(df), size=50, replace=False)
    sample_rows = df.iloc[sample_indices]

    for idx, row in sample_rows.iterrows():
        c_mun = row["cd_mun"]
        t = row["mes_emissao"]
        t_alvo = row["mes_alvo"]
        ano_alvo = int(t_alvo[:4])
        mes_alvo_num = int(t_alvo[5:7])

        # a) focos_t deve bater com o mês de emissão na base bruta
        raw_t = df_raw[(df_raw["cd_mun"] == c_mun) & (df_raw["mes"] == t)]
        if len(raw_t) != 1 or raw_t["focos_count"].values[0] != row["focos_t"]:
            raise ValueError(f"VAZAMENTO/DIVERGÊNCIA: focos_t em ({c_mun}, {t}) não coincide com base bruta!")

        # b) y deve bater com o mês alvo na base bruta
        raw_alvo = df_raw[(df_raw["cd_mun"] == c_mun) & (df_raw["mes"] == t_alvo)]
        if len(raw_alvo) != 1 or raw_alvo["focos_count"].values[0] != row["y"]:
            raise ValueError(f"VAZAMENTO/DIVERGÊNCIA: y em ({c_mun}, {t_alvo}) não coincide com base bruta!")

        # c) focos_media_hist_mes não pode usar nenhum ano >= ano(mes_alvo)
        raw_hist = df_raw[(df_raw["cd_mun"] == c_mun) & (df_raw["mes_num"] == mes_alvo_num)]
        # anos que de fato foram usados:
        valid_raw = raw_hist[raw_hist["ano"] < ano_alvo]
        if len(valid_raw) > 0:
            media_esperada = float(valid_raw["focos_count"].mean())
            if not np.isclose(media_esperada, row["focos_media_hist_mes"]):
                raise ValueError(f"VAZAMENTO/DIVERGÊNCIA: focos_media_hist_mes diverge da média calculada para {c_mun}, {t_alvo}!")
            # Checar expressamente se algum ano >= ano_alvo contaminou
            anos_usados = valid_raw["ano"].values
            if any(a >= ano_alvo for a in anos_usados):
                raise ValueError(f"VAZAMENTO GRAVE: ano >= {ano_alvo} detectado no cálculo histórico de {c_mun}!")
        else:
            if not np.isnan(row["focos_media_hist_mes"]):
                raise ValueError(f"VAZAMENTO: valor não-nulo encontrado quando não havia histórico anterior a {ano_alvo}!")

    logger.info("Validação 5 aprovada: teste de vazamento em 50 linhas passou com 100% de precisão e integridade!")

def gerar_dicionario_dados(df: pd.DataFrame, processed_dir: Path):
    """
    Gera o dicionário de modelagem detalhado em formato Markdown.
    """
    logger.info("Gerando data/processed/dicionario_modelagem.md...")
    out_dict = processed_dir / "dicionario_modelagem.md"

    # Definição dos metadados das colunas
    metadados = [
        # Identificação e Partição
        ("cd_mun", "Código IBGE oficial do município (7 dígitos)", "Identificador geográfico", "Identificação", "Todas"),
        ("nm_mun", "Nome oficial do município", "IBGE Malha Municipal 2024", "Identificação", "Todas"),
        ("mes_emissao", "Mês t de fechamento e emissão da priorização (AAAA-MM)", "Data limite de observação", "Identificação", "Todas"),
        ("mes_alvo", "Mês t+1 de previsão e ação preventiva (AAAA-MM)", "Mês subsequente a mes_emissao", "Identificação", "Todas"),
        ("conjunto", "Partição cronológica da modelagem ('aquecimento', 'treino', 'teste')", "aquecimento: 2019; treino: 2020-01 a 2023-11; teste: 2023-12 a 2025-11", "Identificação", "Todas"),
        # Alvo
        ("y", "Contagem real de focos de calor observados no mês alvo t+1", "focos_count do município em mes_alvo (NASA FIRMS)", "Alvo", "Todas"),
        # Baseline
        ("base_rf_inpe_media_t", "Média zonal do Risco de Fogo do INPE no mês de emissão t", "rf_inpe_media em t (não usado como feature)", "Baseline", "Baselines"),
        # Histórico de Focos
        ("focos_t", "Contagem de focos de calor observados no mês de emissão t", "focos_count em t", "Atributo", "Todos os Modelos"),
        ("focos_t1", "Contagem de focos de calor no mês anterior t-1", "focos_count em t-1", "Atributo", "Todos os Modelos"),
        ("focos_t2", "Contagem de focos de calor em t-2", "focos_count em t-2", "Atributo", "Todos os Modelos"),
        ("focos_mm3", "Média móvel de focos dos últimos 3 meses (t, t-1, t-2)", "(focos_t + focos_t1 + focos_t2) / 3", "Atributo", "Todos os Modelos"),
        ("focos_mesmo_mes_ano_ant", "Focos de calor no mesmo mês do ano anterior ao alvo (t-11)", "focos_count em (mes_alvo - 12 meses) = t-11", "Atributo", "Todos os Modelos"),
        ("focos_media_hist_mes", "Média histórica de focos no mesmo mês de calendário do alvo", "Média de focos no mês calendário do alvo para anos estritamente < ano(alvo)", "Atributo", "Todos os Modelos"),
        ("dens_t", "Densidade de focos em t por km²", "focos_t / area_km2", "Atributo", "Todos os Modelos"),
        ("dens_t1", "Densidade de focos em t-1 por km²", "focos_t1 / area_km2", "Atributo", "Todos os Modelos"),
        ("dens_t2", "Densidade de focos em t-2 por km²", "focos_t2 / area_km2", "Atributo", "Todos os Modelos"),
        ("dens_mm3", "Densidade da média móvel de 3 meses por km²", "focos_mm3 / area_km2", "Atributo", "Todos os Modelos"),
        ("dens_mesmo_mes_ano_ant", "Densidade de focos no mesmo mês do ano anterior por km²", "focos_mesmo_mes_ano_ant / area_km2", "Atributo", "Todos os Modelos"),
        ("dens_media_hist_mes", "Densidade da média histórica do mês alvo por km²", "focos_media_hist_mes / area_km2", "Atributo", "Todos os Modelos"),
        # Clima em t
        ("prec_mm", "Precipitação mensal acumulada no mês de emissão t (mm)", "ERA5-Land total_precipitation em t", "Atributo", "Todos os Modelos"),
        ("t2m_max_c", "Temperatura máxima diária média a 2m no mês t (°C)", "ERA5-Land temperature_2m_max em t", "Atributo", "Todos os Modelos"),
        ("ur_pct", "Umidade relativa média do ar a 2m no mês t (%)", "ERA5-Land deduzida por Magnus-Tetens em t", "Atributo", "Todos os Modelos"),
        ("dpv_kpa", "Déficit de pressão de vapor saturado no mês t (kPa)", "ERA5-Land es(T) - ea(Td) em t", "Atributo", "Todos os Modelos"),
        ("umid_solo", "Umidade volumétrica do solo na camada superficial 0-7 cm (m³/m³)", "ERA5-Land volumetric_soil_water_layer_1 em t", "Atributo", "Todos os Modelos"),
        ("lst_c", "Temperatura da superfície terrestre diurna no mês t (°C)", "ERA5-Land skin_temperature em t", "Atributo", "Todos os Modelos"),
        ("vento_ms", "Velocidade média do vento a 10m no mês t (m/s)", "ERA5-Land sqrt(u10² + v10²) em t", "Atributo", "Todos os Modelos"),
        # Anomalias Climáticas
        ("anom_prec_mm", "Anomalia de precipitação em t em relação à normal 1991-2020", "prec_mm(t) - normal_prec(mês calendário)", "Atributo", "Todos os Modelos"),
        ("anom_t2m_media_c", "Anomalia de temperatura média a 2m em t em relação à normal", "t2m_media_c(t) - normal_t2m(mês calendário)", "Atributo", "Todos os Modelos"),
        ("anom_ur_pct", "Anomalia de umidade relativa em t em relação à normal", "ur_pct(t) - normal_ur(mês calendário)", "Atributo", "Todos os Modelos"),
        ("anom_umid_solo", "Anomalia de umidade do solo em t em relação à normal", "umid_solo(t) - normal_umid(mês calendário)", "Atributo", "Todos os Modelos"),
        ("anom_lst_c", "Anomalia de LST em t em relação à normal", "lst_c(t) - normal_lst(mês calendário)", "Atributo", "Todos os Modelos"),
        # Acumulados
        ("prec_3m", "Precipitação acumulada dos últimos 3 meses (t, t-1, t-2)", "prec_mm(t) + prec_mm(t-1) + prec_mm(t-2)", "Atributo", "Todos os Modelos"),
        ("anom_prec_3m", "Anomalia de precipitação acumulada de 3 meses", "prec_3m - soma das normais dos 3 meses de calendário", "Atributo", "Todos os Modelos"),
        # Sazonalidade
        ("mes_sin", "Componente seno da sazonalidade do mês alvo t+1", "sin(2 * pi * mes_alvo_num / 12)", "Atributo", "Todos os Modelos"),
        ("mes_cos", "Componente cosseno da sazonalidade do mês alvo t+1", "cos(2 * pi * mes_alvo_num / 12)", "Atributo", "Todos os Modelos"),
        # Estruturais
        ("area_km2", "Área territorial oficial do município em km²", "IBGE Malha Municipal 2024 / SIRGAS 2000", "Estrutural", "Só para o híbrido C"),
        ("pct_floresta", "Cobertura florestal (%) com defasagem anual t-1", "MapBiomas Col. 10 (ano = ano(t) - 1)", "Estrutural", "Só para o híbrido C"),
        ("pct_pastagem", "Área de pastagem (%) com defasagem anual t-1", "MapBiomas Col. 10 (ano = ano(t) - 1)", "Estrutural", "Só para o híbrido C"),
        ("pct_agricultura", "Área agrícola consolidada (%) com defasagem anual t-1", "MapBiomas Col. 10 (ano = ano(t) - 1)", "Estrutural", "Só para o híbrido C"),
        ("pct_solo_exposto", "Área de solo exposto/degradado (%) com defasagem anual t-1", "MapBiomas Col. 10 (ano = ano(t) - 1)", "Estrutural", "Só para o híbrido C"),
        ("desmat_km2_prodes", "Desmatamento anual acumulado em km² com defasagem anual t-1", "PRODES Amazônia (ano = ano(t) - 1)", "Estrutural", "Só para o híbrido C"),
        ("desmat_pct_area_prodes", "Desmatamento anual em relação à área total (%) com defasagem t-1", "PRODES Amazônia / area_km2 * 100", "Estrutural", "Só para o híbrido C"),
        ("renda_pc", "Renda domiciliar per capita mediana em R$", "Censo Demográfico 2022 (IBGE SIDRA)", "Estrutural", "Só para o híbrido C")
    ]

    linhas_md = [
        "# Dicionário de Dados da Base de Modelagem APHIM (`aphim_modelagem.parquet`)\n",
        "Este documento descreve detalhadamente cada uma das 43 colunas presentes na tabela final de modelagem do projeto **APHIM**.\n",
        "## Resumo da Estrutura Temporal",
        "- **Unidade de análise:** Par (município, mês de emissão $t$).",
        "- **Mês de emissão ($t$):** 2019-01 a 2025-11 (83 meses x 144 municípios = **11.952 linhas**).",
        "- **Mês alvo ($t+1$):** 2019-02 a 2025-12.",
        "- **Partições:**",
        "  - `aquecimento`: 2019-01 a 2019-12 (1.728 linhas) - utilizado para cálculo de defasagens; descartado do treino.",
        "  - `treino`: 2020-01 a 2023-11 (6.768 linhas) - alvos de 2020-02 a 2023-12.",
        "  - `teste`: 2023-12 a 2025-11 (3.456 linhas) - alvos de 2024-01 a 2025-12.",
        "- **Regra de ouro temporal:** Nenhuma variável preditiva utiliza dado posterior ao fechamento do mês de emissão $t$.\n",
        "## Tabela de Metadados\n",
        "| Coluna | Tipo | Descrição | Fórmula / Origem | Escopo do Modelo |",
        "| :--- | :---: | :--- | :--- | :--- |"
    ]

    for col, desc, form, tipo, escopo in metadados:
        linhas_md.append(f"| `{col}` | **{tipo}** | {desc} | `{form}` | {escopo} |")

    with open(out_dict, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md) + "\n")

    logger.info(f"Dicionário de modelagem salvo com sucesso em '{out_dict}'.")

def executar_processamento_atributos_alvo():
    logger.info("Iniciando Módulo 01: Engenharia de Atributos, Alvo e Divisão Temporal...")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])

    out_parquet = processed_dir / "aphim_modelagem.parquet"

    df_modelagem = construir_tabela_modelagem(processed_dir)
    validar_tabela_modelagem(df_modelagem, processed_dir)
    gerar_dicionario_dados(df_modelagem, processed_dir)

    logger.info(f"Exportando base de modelagem para Parquet: '{out_parquet}'...")
    df_modelagem.to_parquet(out_parquet, index=False)

    logger.info(f"Base de modelagem gerada com sucesso! Dimensões: {df_modelagem.shape[0]} linhas x {df_modelagem.shape[1]} colunas.")
    return df_modelagem

if __name__ == "__main__":
    executar_processamento_atributos_alvo()
