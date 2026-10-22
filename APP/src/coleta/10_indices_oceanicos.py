"""
Módulo 10 — Coleta e Processamento de Índices Oceânicos (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Fontes de Dados Públicas (NOAA):
1. Niño 3.4 (TSM Anomalia Mensal): NOAA Climate Prediction Center (CPC)
   URL: https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices
   Coluna: ANOM após NINO3.4 (NÃO usa ONI centrado para evitar vazamento temporal).
2. TNA (Tropical Northern Atlantic Index): NOAA Physical Sciences Laboratory (PSL)
   URL: https://psl.noaa.gov/data/correlation/tna.data

Saída:
- {processed_dir}/indices_oceanicos_mes.parquet (colunas: mes, nino34_anom, tna)
"""

import os
import sys
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_10_indices_oceanicos")

URL_NINO34 = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
URL_TNA = "https://psl.noaa.gov/data/correlation/tna.data"

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def baixar_linhas_url(url: str) -> list:
    """Faz download do texto de uma URL pública com User-Agent apropriado."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
    return content.strip().split("\n")

def processar_nino34(raw_lines: list) -> pd.DataFrame:
    """
    Processa o arquivo sstoi.indices da NOAA CPC.
    Colunas: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
    Retorna DataFrame com colunas ['mes', 'nino34_anom'].
    """
    logger.info("--- PRIMEIRAS E ÚLTIMAS LINHAS LIDAS: NIÑO 3.4 (NOAA CPC) ---")
    print("\n[NOAA CPC - sstoi.indices] Primeiras 5 linhas:")
    for l in raw_lines[:5]:
        print(f"  {l}")
    print("[NOAA CPC - sstoi.indices] Últimas 5 linhas:")
    for l in raw_lines[-5:]:
        print(f"  {l}")

    registros = []
    for l in raw_lines[1:]:
        parts = l.split()
        if len(parts) >= 10:
            try:
                yr = int(parts[0])
                mon = int(parts[1])
                anom_nino34 = float(parts[9])
                mes_str = f"{yr:04d}-{mon:02d}"
                registros.append({"mes": mes_str, "nino34_anom": anom_nino34})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(registros).sort_values("mes").reset_index(drop=True)
    return df

def processar_tna(raw_lines: list) -> pd.DataFrame:
    """
    Processa o arquivo tna.data da NOAA PSL.
    Formato: Ano seguido de 12 valores mensais (jan-dez). Valores <= -90.0 são faltantes.
    Retorna DataFrame com colunas ['mes', 'tna'].
    """
    logger.info("--- PRIMEIRAS E ÚLTIMAS LINHAS LIDAS: TNA (NOAA PSL) ---")
    print("\n[NOAA PSL - tna.data] Primeiras 5 linhas:")
    for l in raw_lines[:5]:
        print(f"  {l}")
    print("[NOAA PSL - tna.data] Últimas 5 linhas:")
    for l in raw_lines[-5:]:
        print(f"  {l}")

    registros = []
    for l in raw_lines[1:]:
        parts = l.split()
        if len(parts) == 13:
            try:
                yr = int(parts[0])
                for m in range(1, 13):
                    val = float(parts[m])
                    if val <= -90.0:
                        val = np.nan
                    mes_str = f"{yr:04d}-{m:02d}"
                    registros.append({"mes": mes_str, "tna": val})
            except ValueError:
                continue

    df = pd.DataFrame(registros).sort_values("mes").reset_index(drop=True)
    return df

def validar_indices(df: pd.DataFrame):
    """
    Validações obrigatórias especificadas:
    1. Não há meses faltando de 2014-01 a 2025-12 (144 meses contínuos sem nulos).
    2. O Niño 3.4 de 2015-11 ou 2015-12 fica acima de +2,0 °C (El Niño forte).
    3. O Niño 3.4 de 2023-12 fica acima de +1,5 °C.
    """
    logger.info("=== INICIANDO VALIDAÇÕES ESTRITAS DOS ÍNDICES OCEÂNICOS ===")
    
    # Período de interesse para defasagens e modelagem: 2014-01 a 2025-12
    meses_esperados = [d.strftime("%Y-%m") for d in pd.date_range("2014-01-01", "2025-12-31", freq="MS")]
    df_eval = df[df["mes"].isin(meses_esperados)].copy().reset_index(drop=True)
    
    # 1. Contagem e Nulos
    n_meses = len(df_eval)
    if n_meses != 144:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Esperado 144 meses entre 2014-01 e 2025-12, obtido {n_meses}.")
    
    nulos = df_eval.isnull().sum().to_dict()
    print(f"\n[VALIDAÇÃO] Contagem de nulos no período 2014-2025: {nulos}")
    if nulos["nino34_anom"] > 0 or nulos["tna"] > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO 1: Nulos encontrados nos índices oceânicos de 2014 a 2025: {nulos}")
    logger.info("Validação 1 aprovada: 144 meses contínuos sem nenhum valor ausente entre 2014 e 2025.")

    # 2. El Niño 2015-2016
    v_15_11 = df_eval[df_eval["mes"] == "2015-11"]["nino34_anom"].values[0]
    v_15_12 = df_eval[df_eval["mes"] == "2015-12"]["nino34_anom"].values[0]
    print(f"[VALIDAÇÃO] Niño 3.4 2015-11: {v_15_11:+.2f} °C | 2015-12: {v_15_12:+.2f} °C (critério: > +2.0 °C)")
    if not (v_15_11 > 2.0 or v_15_12 > 2.0):
        raise ValueError(f"FALHA NA VALIDAÇÃO 2: Niño 3.4 de 2015-11 ({v_15_11}) e 2015-12 ({v_15_12}) não superaram +2.0 °C.")
    logger.info(f"Validação 2 aprovada: El Niño extremo 2015 confirmado (2015-11 = {v_15_11:+.2f} °C, 2015-12 = {v_15_12:+.2f} °C).")

    # 3. El Niño 2023
    v_23_12 = df_eval[df_eval["mes"] == "2023-12"]["nino34_anom"].values[0]
    print(f"[VALIDAÇÃO] Niño 3.4 2023-12: {v_23_12:+.2f} °C (critério: > +1.5 °C)")
    if not (v_23_12 > 1.5):
        raise ValueError(f"FALHA NA VALIDAÇÃO 3: Niño 3.4 de 2023-12 ({v_23_12}) não superou +1.5 °C.")
    logger.info(f"Validação 3 aprovada: El Niño 2023 confirmado (2023-12 = {v_23_12:+.2f} °C).")
    
    print("\nTODAS AS VALIDAÇÕES DOS ÍNDICES OCEÂNICOS PASSARAM COM SUCESSO!\n")

def executar_coleta_indices():
    logger.info("=== INICIANDO COLETA DE ÍNDICES OCEÂNICOS (NOAA) ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Baixando Niño 3.4 de {URL_NINO34}...")
    lines_nino = baixar_linhas_url(URL_NINO34)
    df_nino = processar_nino34(lines_nino)
    
    logger.info(f"Baixando TNA de {URL_TNA}...")
    lines_tna = baixar_linhas_url(URL_TNA)
    df_tna = processar_tna(lines_tna)
    
    # Junção externa ordenada
    df_indices = pd.merge(df_nino, df_tna, on="mes", how="inner").sort_values("mes").reset_index(drop=True)
    
    # Filtrar a partir de 2010 para ter histórico prévio suficiente
    df_indices = df_indices[df_indices["mes"] >= "2010-01"].reset_index(drop=True)
    
    # Validar
    validar_indices(df_indices)
    
    out_parquet = processed_dir / "indices_oceanicos_mes.parquet"
    logger.info(f"Salvando índices oceânicos em '{out_parquet}'...")
    df_indices.to_parquet(out_parquet, index=False)
    logger.info(f"Concluído com sucesso! Base salva com {len(df_indices)} meses ({df_indices['mes'].min()} a {df_indices['mes'].max()}).")

if __name__ == "__main__":
    executar_coleta_indices()
