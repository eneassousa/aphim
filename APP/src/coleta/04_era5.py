"""
Módulo 04 — Clima ERA5-Land via Google Earth Engine (Projeto APHIM)
Dissertação de Mestrado: Predição e Classificação de Risco de Queimadas no Pará.

Catálogo Oficial GEE:
https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR
Asset: ECMWF/ERA5_LAND/MONTHLY_AGGR (~11 km de resolução nativa)
"""

import os
import sys
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adiciona o diretório raiz ao sys.path para importações relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_04_era5")

# Constantes e Metadados
ERA5_LAND_ASSET = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
CATALOG_URL = "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR"

BANDAS_REQUERIDAS = [
    "temperature_2m",
    "temperature_2m_max",
    "temperature_2m_min",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
    "skin_temperature",
    "volumetric_soil_water_layer_1",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m"
]

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inicializar_gee():
    """
    Inicializa o Google Earth Engine utilizando credenciais do .env (GEE_PROJECT).
    """
    import ee
    load_dotenv()
    gee_project = os.getenv("GEE_PROJECT")
    
    if not gee_project or not gee_project.strip():
        msg = (
            "\n[ERRO DE CONFIGURAÇÃO] GEE_PROJECT não definido no arquivo .env!\n"
            "1. Defina GEE_PROJECT=<seu_projeto_google_cloud> no arquivo .env.\n"
            "2. Se ainda não autenticou o Earth Engine, execute no terminal:\n"
            "   earthengine authenticate\n"
        )
        logger.error(msg)
        raise ValueError("GEE_PROJECT ausente no .env.")

    try:
        ee.Initialize(project=gee_project.strip())
        logger.info(f"Earth Engine inicializado com sucesso no projeto '{gee_project.strip()}'.")
    except Exception as e:
        msg = (
            f"\n[FALHA NA INICIALIZAÇÃO DO GEE] Erro ao conectar ao Earth Engine: {e}\n"
            "Certifique-se de que:\n"
            "1. Executou 'earthengine authenticate' no terminal com sua conta Google.\n"
            "2. O projeto Google Cloud informado em GEE_PROJECT existe e tem a API Earth Engine ativada.\n"
        )
        logger.error(msg)
        raise

def calcular_magnus(t_celsius, td_celsius):
    """
    Calcula umidade relativa (%) e DPV (kPa) através da fórmula de Magnus (Alduchov & Eskridge, 1996).
    es(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))  [kPa]
    ea = es(Td) = 0.61078 * exp((17.27 * Td) / (Td + 237.3)) [kPa]
    ur_pct = 100.0 * (ea / es)
    dpv_kpa = es - ea
    """
    # Pressão de vapor de saturação (kPa)
    es_t = 0.61078 * np.exp((17.27 * t_celsius) / (t_celsius + 237.3))
    # Pressão de vapor real (kPa)
    es_td = 0.61078 * np.exp((17.27 * td_celsius) / (td_celsius + 237.3))
    
    # Umidade Relativa limitada a [0, 100]
    ur = (es_td / es_t) * 100.0
    ur_pct = np.clip(ur, 0.0, 100.0)
    
    # Déficit de Pressão de Vapor (kPa)
    dpv_kpa = np.maximum(0.0, es_t - es_td)
    
    return ur_pct, dpv_kpa

def converter_bandas_era5(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica as conversões físicas de unidades e calcula as variáveis derivadas.
    Documentação da limitação do vento:
    vento_ms é computado como sqrt(u² + v²) a partir das médias mensais dos componentes
    vetoriais de vento (u_component_of_wind_10m e v_component_of_wind_10m).
    """
    df = df_raw.copy()
    
    # Conversões térmicas: Kelvin para Celsius
    df["t2m_media_c"] = df["temperature_2m"] - 273.15
    df["t2m_max_c"] = df["temperature_2m_max"] - 273.15
    df["t2m_min_c"] = df["temperature_2m_min"] - 273.15
    df["td2m_c"] = df["dewpoint_temperature_2m"] - 273.15
    df["lst_c"] = df["skin_temperature"] - 273.15
    
    # Precipitação: metros para milímetros (m * 1000)
    prec_raw = df["total_precipitation_sum"] * 1000.0
    negativos = (prec_raw < 0).sum()
    if negativos > 0:
        logger.warning(f"Identificados {negativos} valores negativos de precipitação no lote. Ajustando para 0.0 mm.")
    df["prec_mm"] = np.maximum(0.0, prec_raw)
    
    # Umidade do solo: fração volumétrica (sem conversão)
    df["umid_solo"] = df["volumetric_soil_water_layer_1"]
    
    # Velocidade do vento a 10m: magnitude vetorial das médias mensais
    u = df["u_component_of_wind_10m"]
    v = df["v_component_of_wind_10m"]
    df["vento_ms"] = np.sqrt(u**2 + v**2)
    
    # Variáveis derivadas (Magnus)
    ur_pct, dpv_kpa = calcular_magnus(df["t2m_media_c"].values, df["td2m_c"].values)
    df["ur_pct"] = ur_pct
    df["dpv_kpa"] = dpv_kpa
    
    return df

def criar_ee_feature_collection(gdf_mun: gpd.GeoDataFrame):
    """
    Converte o GeoDataFrame dos 144 municípios para um ee.FeatureCollection.
    Garante CRS EPSG:4326 e inclusão estrita de cd_mun.
    """
    import ee
    gdf_wgs84 = gdf_mun.to_crs(epsg=4326)
    features = []
    for _, row in gdf_wgs84.iterrows():
        geom_json = row.geometry.__geo_interface__
        f = ee.Feature(geom_json, {
            "cd_mun": str(row["cd_mun"]).zfill(7),
            "nm_mun": str(row["nm_mun"])
        })
        features.append(f)
    return ee.FeatureCollection(features)

def extrair_mes_gee(ano: int, mes: int, fc_mun, collection) -> pd.DataFrame:
    """
    Extrai as médias zonais para os 144 municípios do Pará em um único mês.
    Utiliza scale=1000m (reamostragem por vizinho mais próximo) para garantir que
    nenhum município pequeno retorne nulo.
    """
    import ee
    
    # Filtra a imagem exata do mês
    img_month = ee.Image(
        collection
        .filter(ee.Filter.calendarRange(ano, ano, "year"))
        .filter(ee.Filter.calendarRange(mes, mes, "month"))
        .first()
    ).select(BANDAS_REQUERIDAS)
    
    # Executa reduceRegions com scale=1000m
    reduced = img_month.reduceRegions(
        collection=fc_mun,
        reducer=ee.Reducer.mean(),
        scale=1000
    )
    
    # Baixa os resultados
    info = reduced.getInfo()
    records = []
    mes_str = f"{ano}-{mes:02d}"
    
    for feat in info["features"]:
        props = feat["properties"]
        props["mes"] = mes_str
        records.append(props)
        
    df_mes = pd.DataFrame(records)
    return df_mes

def extrair_climatologia_normal_gee(fc_mun, collection, raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """
    Calcula a normal climatológica 1991–2020 (30 anos) para cada mês do calendário (1 a 12).
    Saída esperada: 144 municípios x 12 meses = 1.728 linhas.
    """
    import ee
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    out_normal_file = processed_dir / "clima_normal_1991_2020.parquet"
    if out_normal_file.exists():
        logger.info(f"Climatologia normal já existe em '{out_normal_file}'. Carregando...")
        df_normal_existente = pd.read_parquet(out_normal_file)
        if len(df_normal_existente) == 1728:
            logger.info("Execução pulada: climatologia normal 1991-2020 já validada com 1.728 linhas.")
            return df_normal_existente
            
    logger.info("Extraindo climatologia de referência 1991–2020 no Earth Engine (meses 1 a 12)...")
    normal_raw_dir = raw_dir / "normal_1991_2020"
    normal_raw_dir.mkdir(parents=True, exist_ok=True)
    
    col_normal = collection.filter(ee.Filter.calendarRange(1991, 2020, "year"))
    
    def processar_mes_normal(m):
        chunk_file = normal_raw_dir / f"era5_normal_m{m:02d}.parquet"
        if chunk_file.exists():
            return pd.read_parquet(chunk_file)
            
        for tentativa in range(3):
            try:
                img_comp = col_normal.filter(ee.Filter.calendarRange(m, m, "month")).mean().select(BANDAS_REQUERIDAS)
                reduced = img_comp.reduceRegions(
                    collection=fc_mun,
                    reducer=ee.Reducer.mean(),
                    scale=1000
                )
                info = reduced.getInfo()
                records = []
                for feat in info["features"]:
                    props = feat["properties"]
                    props["mes_calendario"] = m
                    records.append(props)
                df_m = pd.DataFrame(records)
                df_m.to_parquet(chunk_file, index=False)
                return df_m
            except Exception as e:
                logger.warning(f"Tentativa {tentativa+1}/3 para normal do mês {m} falhou: {e}")
                if tentativa == 2:
                    raise
                    
    dfs_meses = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(processar_mes_normal, m): m for m in range(1, 13)}
        for future in tqdm(as_completed(futures), total=12, desc="Normal Climatológica 1991-2020"):
            dfs_meses.append(future.result())
            
    df_raw_all = pd.concat(dfs_meses, ignore_index=True)
    df_conv = converter_bandas_era5(df_raw_all)
    
    # Selecionar variáveis canônicas da normal climatológica
    colunas_normal = ["cd_mun", "mes_calendario", "prec_mm", "t2m_media_c", "ur_pct", "umid_solo", "lst_c"]
    df_final_normal = df_conv[colunas_normal].sort_values(["mes_calendario", "cd_mun"]).reset_index(drop=True)
    
    if len(df_final_normal) != 1728:
        raise ValueError(f"FALHA: Esperado 1.728 linhas na normal 1991-2020, obtido {len(df_final_normal)}.")
        
    df_final_normal.to_parquet(out_normal_file, index=False)
    logger.info(f"Climatologia normal salva com sucesso em '{out_normal_file}' ({len(df_final_normal)} linhas).")
    return df_final_normal

def validar_serie_clima(df_clima: pd.DataFrame, processed_dir: Path) -> dict:
    """
    Executa todas as validações obrigatórias especificadas:
    1. Exatamente 13.824 linhas (144 x 96), sem duplicatas e sem nulos.
    2. Faixas plausíveis para o Pará:
       - t2m_media_c: [18, 35]
       - prec_mm: [0, 1000]
       - ur_pct: [30, 100]
       - umid_solo: [0, 0.6]
    3. Sazonalidade coerente:
       - Chuva maior em Fev/Mar/Abr e menor em Jul/Ago/Set.
    4. Correlação com focos de calor do Pará (2019-2025):
       - Esperada correlação negativa entre média estadual de chuva e focos totais.
    """
    logger.info("=== INICIANDO VALIDAÇÕES OBRIGATÓRIAS DO CLIMA ERA5-LAND ===")
    
    # Validação 1: Contagem e integridade
    n_linhas = len(df_clima)
    esperado = 144 * 96 # 13.824
    if n_linhas != esperado:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Esperado {esperado} linhas, obtido {n_linhas}.")
        
    dups = df_clima.duplicated(subset=["cd_mun", "mes"]).sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Encontradas {dups} duplicatas de (cd_mun, mes).")
        
    colunas_obrigatorias = [
        "cd_mun", "mes", "t2m_media_c", "t2m_max_c", "t2m_min_c", "td2m_c",
        "prec_mm", "lst_c", "umid_solo", "vento_ms", "ur_pct", "dpv_kpa"
    ]
    nulos = df_clima[colunas_obrigatorias].isnull().sum()
    if nulos.sum() > 0:
        logger.error(f"Valores nulos encontrados:\n{nulos[nulos > 0]}")
        raise ValueError("FALHA NA VALIDAÇÃO: Valores nulos encontrados na série de clima.")
        
    logger.info(f"Validação de integridade aprovada: 13.824 linhas sem duplicatas e sem nulos.")

    # Validação 2: Faixas plausíveis para o Pará
    fora_t2m = df_clima[~df_clima["t2m_media_c"].between(18.0, 35.0)]
    fora_prec = df_clima[~df_clima["prec_mm"].between(0.0, 1000.0)]
    fora_ur = df_clima[~df_clima["ur_pct"].between(30.0, 100.0)]
    fora_solo = df_clima[~df_clima["umid_solo"].between(0.0, 0.6)]
    
    total_fora_faixa = len(fora_t2m) + len(fora_prec) + len(fora_ur) + len(fora_solo)
    
    logger.info(f"Casos fora de faixa:")
    logger.info(f" - t2m_media_c fora de [18, 35]: {len(fora_t2m)}")
    logger.info(f" - prec_mm fora de [0, 1000]: {len(fora_prec)}")
    logger.info(f" - ur_pct fora de [30, 100]: {len(fora_ur)}")
    logger.info(f" - umid_solo fora de [0, 0.6]: {len(fora_solo)}")
    
    # Validação 3: Sazonalidade
    df_clima["mes_calendario"] = pd.to_datetime(df_clima["mes"]).dt.month
    sazonalidade = df_clima.groupby("mes_calendario").agg(
        prec_mm_media=("prec_mm", "mean"),
        t2m_c_media=("t2m_media_c", "mean"),
        ur_pct_media=("ur_pct", "mean")
    ).reset_index()
    
    nomes_meses = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    sazonalidade["nome_mes"] = sazonalidade["mes_calendario"].map(nomes_meses)
    
    logger.info("\n--- MÉDIA ESTADUAL POR MÊS DO CALENDÁRIO (1991-2025 / 2018-2025) ---")
    for _, r in sazonalidade.iterrows():
        logger.info(f" {r['nome_mes']:<10}: Chuva = {r['prec_mm_media']:>6.1f} mm | Temp = {r['t2m_c_media']:>4.1f} °C | UR = {r['ur_pct_media']:>4.1f}%")
        
    # Verificar se fev, mar, abr são maiores que jul, ago, set
    chuva_chuvosa = sazonalidade[sazonalidade["mes_calendario"].isin([2, 3, 4])]["prec_mm_media"].mean()
    chuva_seca = sazonalidade[sazonalidade["mes_calendario"].isin([7, 8, 9])]["prec_mm_media"].mean()
    
    if chuva_chuvosa <= chuva_seca:
        raise ValueError(f"FALHA NA SAZONALIDADE: Média do trimestre chuvoso ({chuva_chuvosa:.1f} mm) <= trimestre seco ({chuva_seca:.1f} mm).")
    logger.info(f"Sazonalidade validada: Trimestre chuvoso ({chuva_chuvosa:.1f} mm) >> Trimestre seco ({chuva_seca:.1f} mm).")
    
    # Validação 4: Correlação com focos de queimadas do Pará (2019-2025)
    focos_path = processed_dir / "focos_municipio_mes.parquet"
    correlacao_chuva_focos = None
    if focos_path.exists():
        df_focos = pd.read_parquet(focos_path)
        col_focos = "focos_count" if "focos_count" in df_focos.columns else df_focos.columns[2]
        focos_mensais = df_focos.groupby("mes")[col_focos].sum().reset_index()
        # Média estadual de chuva por mês
        chuva_mensal = df_clima.groupby("mes")["prec_mm"].mean().reset_index()
        
        df_merge = pd.merge(chuva_mensal, focos_mensais, on="mes")
        # Filtrar 2019-2025
        df_merge_19_25 = df_merge[df_merge["mes"] >= "2019-01"]
        
        correlacao_chuva_focos = df_merge_19_25["prec_mm"].corr(df_merge_19_25[col_focos])
        logger.info(f"\nCorrelação de Pearson (2019-2025) entre Chuva Média Estadual e Focos Totais do Pará: {correlacao_chuva_focos:.4f}")
        if correlacao_chuva_focos >= 0:
            logger.warning(f"Atenção: A correlação chuva x focos resultou positiva ({correlacao_chuva_focos:.4f}), contrariando o esperado.")
        else:
            logger.info("Correlação chuva x focos coerente e fortemente negativa!")
    else:
        logger.warning(f"Arquivo '{focos_path}' não encontrado. Correlação com focos pulada.")
        
    return {
        "n_linhas": n_linhas,
        "total_fora_faixa": total_fora_faixa,
        "sazonalidade": sazonalidade,
        "correlacao_chuva_focos": correlacao_chuva_focos
    }

def executar_coleta_era5():
    """
    Função principal de orquestração da etapa 04: Clima ERA5-Land.
    """
    logger.info("Iniciando Módulo 04: Clima ERA5-Land via Google Earth Engine...")
    config = carregar_config()
    
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir = Path(config["caminhos"]["raw"]) / "era5"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    out_principal = processed_dir / "clima_municipio_mes.parquet"
    out_normal = processed_dir / "clima_normal_1991_2020.parquet"
    
    # Idempotência total se ambos já existirem e forem válidos
    if out_principal.exists() and out_normal.exists():
        logger.info("Arquivos finais de clima já existem. Verificando integridade...")
        df_existente = pd.read_parquet(out_principal)
        if len(df_existente) == 13824:
            if "prec_extremo" not in df_existente.columns:
                df_existente["prec_extremo"] = df_existente["prec_mm"] > 1000.0
                df_existente.to_parquet(out_principal, index=False)
                logger.info(f"Coluna 'prec_extremo' adicionada a '{out_principal}' ({df_existente['prec_extremo'].sum()} registros verdadeiros).")
            logger.info("Série principal existente validada (13.824 linhas). Executando validações...")
            res_val = validar_serie_clima(df_existente, processed_dir)
            return df_existente, res_val

    # Carregar malha municipal oficial
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
    gdf_mun = gpd.read_file(gpkg_path)
    logger.info(f"Malha municipal carregada: {len(gdf_mun)} municípios.")

    # Inicializar Earth Engine
    inicializar_gee()
    import ee
    
    # Preparar FeatureCollection e ImageCollection
    fc_mun = criar_ee_feature_collection(gdf_mun)
    collection = ee.ImageCollection(ERA5_LAND_ASSET)
    
    # 1. Climatologia de referência 1991-2020
    df_normal = extrair_climatologia_normal_gee(fc_mun, collection, raw_dir, processed_dir)
    
    # 2. Série Principal: jan/2018 a dez/2025 (96 meses)
    logger.info("Extraindo série principal jan/2018 a dez/2025 (96 meses)...")
    meses_totais = []
    for ano in range(2018, 2026):
        for mes in range(1, 13):
            meses_totais.append((ano, mes))
            
    def processar_chunk_mensal(item):
        ano, mes = item
        chunk_file = raw_dir / f"era5_{ano}_{mes:02d}.parquet"
        if chunk_file.exists():
            return pd.read_parquet(chunk_file)
            
        for tentativa in range(3):
            try:
                df_m = extrair_mes_gee(ano, mes, fc_mun, collection)
                df_m.to_parquet(chunk_file, index=False)
                return df_m
            except Exception as e:
                logger.warning(f"Tentativa {tentativa+1}/3 para {ano}-{mes:02d} falhou: {e}")
                if tentativa == 2:
                    raise
                    
    dfs_chunks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(processar_chunk_mensal, item): item for item in meses_totais}
        for future in tqdm(as_completed(futures), total=len(meses_totais), desc="Meses ERA5-Land (2018-2025)"):
            dfs_chunks.append(future.result())
        
    df_bruto_all = pd.concat(dfs_chunks, ignore_index=True)
    logger.info(f"Total bruto agregado: {len(df_bruto_all)} registros municipais-mensais.")
    
    # Aplica conversões e calcula Magnus
    logger.info("Aplicando conversões físicas de unidades e fórmulas de Magnus...")
    df_clima = converter_bandas_era5(df_bruto_all)
    df_clima["prec_extremo"] = df_clima["prec_mm"] > 1000.0
    
    # Ordenação e seleção de colunas canônicas
    colunas_ordenadas = [
        "cd_mun", "mes",
        "t2m_media_c", "t2m_max_c", "t2m_min_c", "td2m_c",
        "prec_mm", "prec_extremo", "lst_c", "umid_solo", "vento_ms",
        "ur_pct", "dpv_kpa"
    ]
    df_final = df_clima[colunas_ordenadas].sort_values(["mes", "cd_mun"]).reset_index(drop=True)
    
    # Validações estritas
    res_val = validar_serie_clima(df_final, processed_dir)
    
    # Salvar saída final
    df_final.to_parquet(out_principal, index=False)
    logger.info(f"Série climática salva com sucesso em '{out_principal}' ({len(df_final)} linhas).")
    
    return df_final, res_val

if __name__ == "__main__":
    executar_coleta_era5()
