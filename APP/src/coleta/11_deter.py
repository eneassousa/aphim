"""
Módulo 11 — DETER Amazônia: Coleta e Validação dos Alertas Mensais no Pará (Projeto APHIM)
Dissertação de Mestrado: Predição e Classificação de Risco de Queimadas no Pará.

Fonte Oficial:
INPE / TerraBrasilis — DETER Amazônia
WFS Endpoint: https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/ows
Camada WFS: deter-amz:deter_amz

Objetivo:
1. Executar DescribeFeatureType e mapear os campos da camada.
2. Baixar alertas do DETER para o Pará com view_date entre 2018-01-01 e 2025-12-31.
3. Salvar polígonos brutos em data/raw/deter/deter_pa_<ano>.parquet (GeoParquet).
4. Agrupar classes (desmat, degrad, cicatriz) e verificar ocorrência de classes não previstas.
5. Validação contra o PRODES: calcular a razão DETER/PRODES por ano PRODES (agosto a julho).
6. Verificar e registrar a regra de disponibilidade temporal (anti-vazamento).
"""

import os
import sys
import io
import time
import calendar
from pathlib import Path
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_11_deter")

WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/ows"
LAYER_NAME = "deter-amz:deter_amz"

# Taxas Oficiais publicadas pelo INPE para o Pará (km² por ano PRODES - 01/ago a 31/jul)
# Fonte: INPE / TerraBrasilis (mesmo dicionário do 06_prodes.py)
TAXAS_OFICIAIS_INPE_PA = {
    2014: 1887.0,
    2015: 2153.0,
    2016: 2992.0,
    2017: 2433.0,
    2018: 2744.0,
    2019: 4172.0,
    2020: 4899.0,
    2021: 5238.0,
    2022: 4141.0,
    2023: 3272.0,
    2024: 2736.0,
    2025: 2450.0
}

# Grupos de classes oficiais solicitados no projeto
CLASSES_GRUPO = {
    "DESMATAMENTO_CR": "desmat",
    "DESMATAMENTO_VEG": "desmat",
    "MINERACAO": "desmat",
    "DEGRADACAO": "degrad",
    "CS_DESORDENADO": "degrad",
    "CS_GEOMETRICO": "degrad",
    "CICATRIZ_DE_QUEIMADA": "cicatriz"
}

def descrever_tipo_camada() -> Dict[str, str]:
    """
    Executa DescribeFeatureType e imprime todos os campos e seus tipos XML.
    """
    logger.info(f"Executando DescribeFeatureType para a camada '{LAYER_NAME}'...")
    url = f"{WFS_URL}?service=WFS&version=1.0.0&request=DescribeFeatureType&typeName={LAYER_NAME}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha ao executar DescribeFeatureType: status {r.status_code}")
    
    root = ET.fromstring(r.content)
    campos = {}
    for elem in root.iter():
        if elem.tag.endswith("element") and "name" in elem.attrib:
            nome = elem.attrib.get("name")
            tipo = elem.attrib.get("type", "desconhecido")
            campos[nome] = tipo

    logger.info("=== CAMPOS ENCONTRADOS NO DescribeFeatureType ===")
    for k, v in campos.items():
        logger.info(f" - {k:<15} : {v}")

    return campos

def fetch_wfs_csv(url: str, max_retries: int = 5, timeout: int = 120) -> pd.DataFrame:
    """
    Faz requisição HTTP GET para o WFS esperando formato CSV.
    Implementa retentativas exponenciais em caso de erro 503, timeout ou erro temporário.
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text.startswith("FID"):
                return pd.read_csv(io.StringIO(r.text))
            elif r.status_code == 200 and len(r.text.strip()) == 0:
                # CSV vazio
                return pd.DataFrame()
            else:
                snippet = r.text[:120].replace("\n", " ").replace("\r", " ")
                logger.warning(f"Tentativa {attempt+1}/{max_retries} retornou status {r.status_code}. Trecho: {snippet}")
        except Exception as e:
            logger.warning(f"Tentativa {attempt+1}/{max_retries} falhou com exceção: {e}")
        
        sleep_time = 3 * (attempt + 1)
        time.sleep(sleep_time)

    raise RuntimeError(f"Não foi possível obter dados do WFS após {max_retries} tentativas: {url}")

def baixar_alertas_periodo(d_ini: str, d_fim: str) -> pd.DataFrame:
    """
    Baixa alertas para o estado do Pará (uf='PA') entre d_ini e d_fim.
    Se a janela mensal falhar ou demorar, divide recursivamente pela metade.
    """
    cql = f"uf='PA' AND view_date>='{d_ini}' AND view_date<='{d_fim}'"
    url = f"{WFS_URL}?service=WFS&version=1.0.0&request=GetFeature&typeName={LAYER_NAME}&cql_filter={cql}&outputFormat=csv"
    
    try:
        df = fetch_wfs_csv(url, max_retries=4, timeout=120)
        return df
    except Exception as e:
        logger.warning(f"Janela {d_ini} a {d_fim} falhou ({e}). Subdividindo período...")
        t_ini = pd.to_datetime(d_ini)
        t_fim = pd.to_datetime(d_fim)
        if t_ini == t_fim:
            raise RuntimeError(f"Falha definitiva ao baixar dia {d_ini}: {e}")
        
        t_mid = t_ini + (t_fim - t_ini) // 2
        d_mid1 = t_mid.strftime("%Y-%m-%d")
        d_mid2 = (t_mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        df1 = baixar_alertas_periodo(d_ini, d_mid1)
        df2 = baixar_alertas_periodo(d_mid2, d_fim)
        return pd.concat([df1, df2], ignore_index=True)

def baixar_deter_ano(ano: int, raw_dir: Path) -> gpd.GeoDataFrame:
    """
    Baixa todos os alertas do DETER para o Pará no ano especificado,
    iterando mês a mês para garantir consultas leves e estáveis no GeoServer.
    Salva em GeoParquet em data/raw/deter/deter_pa_<ano>.parquet.
    """
    out_parquet = raw_dir / f"deter_pa_{ano}.parquet"
    if out_parquet.exists():
        logger.info(f"Arquivo anual '{out_parquet}' já existe. Carregando...")
        gdf = gpd.read_parquet(out_parquet)
        logger.info(f"Ano {ano}: {len(gdf)} alertas carregados.")
        return gdf

    logger.info(f"Baixando alertas DETER de {ano} para o Pará mês a mês...")
    dfs_meses = []
    for mes in range(1, 13):
        ultimo_dia = calendar.monthrange(ano, mes)[1]
        d_ini = f"{ano:04d}-{mes:02d}-01"
        d_fim = f"{ano:04d}-{mes:02d}-{ultimo_dia:02d}"
        
        df_mes = baixar_alertas_periodo(d_ini, d_fim)
        if not df_mes.empty:
            dfs_meses.append(df_mes)
            logger.info(f"  {ano:04d}-{mes:02d}: {len(df_mes)} alertas obtidos.")
        else:
            logger.info(f"  {ano:04d}-{mes:02d}: 0 alertas.")

    if dfs_meses:
        df_ano = pd.concat(dfs_meses, ignore_index=True)
    else:
        df_ano = pd.DataFrame()

    if df_ano.empty:
        logger.warning(f"Nenhum alerta encontrado para o ano {ano}.")
        gdf = gpd.GeoDataFrame()
        return gdf

    # Conversão de geometria WKT
    logger.info(f"Convertendo {len(df_ano)} geometrias WKT para GeoDataFrame (EPSG:4674)...")
    df_ano["geometry"] = df_ano["geom"].apply(wkt.loads)
    df_ano.drop(columns=["geom"], inplace=True)
    gdf = gpd.GeoDataFrame(df_ano, geometry="geometry", crs="EPSG:4674")

    # Salva em GeoParquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_parquet, index=False)
    logger.info(f"Alertas de {ano} salvos com sucesso em '{out_parquet}' ({len(gdf)} registros).")
    return gdf

def executar_coleta_deter():
    logger.info("=== INICIANDO COLETA DO DETER AMAZÔNIA (2018–2025) ===")
    
    # 1. DescribeFeatureType
    campos = descrever_tipo_camada()
    
    # Validações dos campos solicitados no prompt:
    # - campo da classe: classname
    # - data da imagem: view_date
    # - data de publicação: publish_month
    # - município e geocódigo: municipality, mun_geocod
    # - UF: uf
    # - áreas: areauckm, areamunkm
    logger.info(f"Camada pública utilizada: {LAYER_NAME} no endpoint {WFS_URL}")
    logger.info(f"Campo de classe: 'classname' ({campos.get('classname')})")
    logger.info(f"Campo de data da imagem: 'view_date' ({campos.get('view_date')})")
    logger.info(f"Campo de publicação existente: 'publish_month' ({campos.get('publish_month')})")
    logger.info(f"Campos municipais: municipality='{campos.get('municipality')}', mun_geocod='{campos.get('mun_geocod')}'")
    logger.info(f"Campo UF: 'uf' ({campos.get('uf')})")
    logger.info(f"Campos de área: areauckm='{campos.get('areauckm')}', areamunkm='{campos.get('areamunkm')}'")

    raw_dir = Path("data/raw/deter")
    raw_dir.mkdir(parents=True, exist_ok=True)

    gdfs_anos = []
    for ano in range(2018, 2026):
        gdf_ano = baixar_deter_ano(ano, raw_dir)
        if not gdf_ano.empty:
            gdfs_anos.append(gdf_ano)

    gdf_total = pd.concat(gdfs_anos, ignore_index=True)
    logger.info(f"Total consolidado de alertas DETER (2018–2025): {len(gdf_total):,} alertas.")

    # Datas mínima e máxima
    data_min = gdf_total["view_date"].min()
    data_max = gdf_total["view_date"].max()
    logger.info(f"Período temporal coberto: view_date mínima = {data_min} | máxima = {data_max}")

    # Verificar classes encontradas
    classes_encontradas = sorted(gdf_total["classname"].dropna().unique().tolist())
    logger.info(f"\nLista de classes encontradas no DETER Amazônia ({len(classes_encontradas)} classes):")
    for cl in classes_encontradas:
        logger.info(f" - {cl}")

    classes_desconhecidas = [c for c in classes_encontradas if c not in CLASSES_GRUPO]
    if classes_desconhecidas:
        msg = f"ATENÇÃO: Classes fora da lista esperada encontradas: {classes_desconhecidas}"
        logger.error(msg)
        raise ValueError(msg)

    # Mapear grupo
    gdf_total["grupo"] = gdf_total["classname"].map(CLASSES_GRUPO)
    gdf_total["ano"] = pd.to_datetime(gdf_total["view_date"]).dt.year

    # Tabela de alertas e área total (km²) por ano e por classe
    # Usar areamunkm ou calcular área via projeção de área igual EPSG:5880
    logger.info("Calculando área de cada polígono em EPSG:5880 (km²)...")
    gdf_total["area_calc_km2"] = gdf_total.to_crs(epsg=5880).geometry.area / 1e6

    tabela_classe_ano = gdf_total.groupby(["ano", "classname"]).agg(
        qtd_alertas=("gid", "count"),
        area_km2=("area_calc_km2", "sum")
    ).reset_index()

    logger.info("\n=== TABELA DE ALERTAS E ÁREA POR ANO E CLASSE ===")
    print("\nTABELA 1 — ALERTAS E ÁREA TOTAL (KM²) POR ANO E CLASSE:")
    print(tabela_classe_ano.to_string(index=False))

    tabela_grupo_ano = gdf_total.groupby(["ano", "grupo"]).agg(
        qtd_alertas=("gid", "count"),
        area_km2=("area_calc_km2", "sum")
    ).reset_index()
    print("\nTABELA 2 — RESUMO POR GRUPO E ANO:")
    print(tabela_grupo_ano.to_string(index=False))

    # Primeiras linhas
    print("\nPRIMEIRAS LINHAS DA BASE BRUTA:")
    print(gdf_total[["gid", "classname", "grupo", "view_date", "publish_month", "municipality", "area_calc_km2"]].head(5))

    # Parte B: Análise de data de publicação e anti-vazamento
    logger.info("\n=== PARTE B: AUDITORIA DE PUBLICAÇÃO E ANTI-VAZAMENTO ===")
    gdf_total["vd_dt"] = pd.to_datetime(gdf_total["view_date"])
    gdf_total["pm_dt"] = pd.to_datetime(gdf_total["publish_month"])
    
    # Contagem com publish_month < view_date como data diária
    pub_antes_view_diario = (gdf_total["pm_dt"] < gdf_total["vd_dt"]).sum()
    pub_antes_view_mes = (gdf_total["pm_dt"].dt.to_period("M") < gdf_total["vd_dt"].dt.to_period("M")).sum()
    
    logger.info(f"Contagem de alertas com publish_month < view_date (nível diário): {pub_antes_view_diario:,}")
    logger.info(f"Contagem de alertas com publish_month < view_date (nível mensal): {pub_antes_view_mes:,}")

    # Explicitação da regra:
    # Como o campo publish_month é gerado pelo TerraBrasilis como YYYY-MM-01 (primeiro dia do mês da imagem),
    # no nível diário ele ocorre antes da view_date para todas as imagens do dia 02 em diante.
    # No nível mensal, o mês da publish_month é rigorosamente igual ao mês da view_date.
    # Para garantir rigor anti-vazamento absoluto ('no mês de emissão t só entram alertas com view_date até o fim de t-1'),
    # a regra de disponibilidade temporal adotada é:
    # O alerta observado no mês m fica disponível para modelagem no mês seguinte (m + 1).
    regra_usada = "mes_seguinte_view_date"
    descricao_regra = (
        "Alerta com view_date no mês m fica disponível a partir do mês m+1. "
        "No mês de emissão t, utilizam-se estritamente alertas com view_date até o fim de t-1."
    )
    logger.info(f"Regra de disponibilidade temporal adotada: {regra_usada}")
    logger.info(f"Descrição da regra: {descricao_regra}")

    # Validação contra o PRODES
    logger.info("\n=== VALIDAÇÃO CONTRA O PRODES AMAZÔNIA ===")
    # Ano PRODES: 01 de agosto de a-1 a 31 de julho de a
    # Apenas grupo 'desmat' (DESMATAMENTO_CR, DESMATAMENTO_VEG, MINERACAO)
    desmat_gdf = gdf_total[gdf_total["grupo"] == "desmat"].copy()
    
    def definir_ano_prodes(dt):
        if dt.month >= 8:
            return dt.year + 1
        return dt.year

    desmat_gdf["ano_prodes"] = desmat_gdf["vd_dt"].apply(definir_ano_prodes)

    totais_deter_prodes = desmat_gdf.groupby("ano_prodes")["area_calc_km2"].sum().reset_index()
    totais_deter_prodes["taxa_oficial_prodes_km2"] = totais_deter_prodes["ano_prodes"].map(TAXAS_OFICIAIS_INPE_PA)
    totais_deter_prodes["razao_deter_prodes"] = (
        totais_deter_prodes["area_calc_km2"] / totais_deter_prodes["taxa_oficial_prodes_km2"]
    )

    print("\nTABELA 3 — COMPARAÇÃO ANUAL: DETER (DESMAT) vs PRODES (OFICIAL INPE):")
    for _, r in totais_deter_prodes.iterrows():
        ano_p = int(r["ano_prodes"])
        if ano_p in range(2019, 2026): # Anos com 12 meses completos no período coletado
            print(
                f" Ano PRODES {ano_p}: DETER = {r['area_calc_km2']:>8.1f} km² | "
                f"PRODES Oficial = {r['taxa_oficial_prodes_km2']:>8.1f} km² | "
                f"Razão DETER/PRODES = {r['razao_deter_prodes']:>6.3f}"
            )
            if r["razao_deter_prodes"] < 0.2 or r["razao_deter_prodes"] > 1.5:
                msg = f"FALHA NA VALIDAÇÃO DETER/PRODES: Razão {r['razao_deter_prodes']:.3f} no ano {ano_p} fora da faixa [0.2, 1.5]."
                logger.error(msg)
                raise ValueError(msg)

    logger.info("Validação DETER/PRODES aprovada com sucesso: todas as razões dentro da faixa esperada [0.2, 1.5]!")

    return gdf_total, tabela_classe_ano, totais_deter_prodes

if __name__ == "__main__":
    executar_coleta_deter()
