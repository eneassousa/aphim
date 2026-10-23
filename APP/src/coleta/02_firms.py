import os
import sys
import time
import yaml
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_02_firms")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def gerar_janelas_datas(data_inicio_str, data_fim_str, step_days=5):
    inicio = datetime.strptime(data_inicio_str, "%Y-%m-%d")
    fim = datetime.strptime(data_fim_str, "%Y-%m-%d")
    marco_2019 = datetime.strptime("2019-01-01", "%Y-%m-%d")
    
    # Se o período começar antes de 2019 e for até 2019 ou além,
    # alinha as janelas em 2019-01-01 para garantir 100% de reaproveitamento do cache histórico!
    if inicio < marco_2019 <= fim:
        fim_anterior = marco_2019 - timedelta(days=1)
        janelas_ant = gerar_janelas_datas(inicio.strftime("%Y-%m-%d"), fim_anterior.strftime("%Y-%m-%d"), step_days)
        janelas_pos = gerar_janelas_datas("2019-01-01", data_fim_str, step_days)
        return janelas_ant + janelas_pos

    janelas = []
    curr = inicio
    while curr <= fim:
        prox = min(curr + timedelta(days=step_days - 1), fim)
        diff_days = (prox - curr).days + 1
        janelas.append((curr.strftime("%Y-%m-%d"), diff_days))
        curr = prox + timedelta(days=1)
    return janelas

def baixar_janela_firms(map_key, source_main, source_nrt, bbox, day_range, start_date, raw_dir):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    out_file = raw_dir / f"firms_{source_main}_{start_date}_{day_range}d.csv"
    out_file_nrt = raw_dir / f"firms_{source_nrt}_{start_date}_{day_range}d.csv"
    
    if out_file.exists():
        try:
            df = pd.read_csv(out_file)
            return df, "SP", start_date
        except Exception:
            pass

    if out_file_nrt.exists():
        try:
            df = pd.read_csv(out_file_nrt)
            return df, "NRT", start_date
        except Exception:
            pass
            
    # Tenta fonte principal SP
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source_main}/{bbox_str}/{day_range}/{start_date}"
    
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                text = r.text
                if "Invalid MAP_KEY" in text:
                    raise ValueError("FIRMS MAP_KEY inválida.")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                return pd.read_csv(out_file), "SP", start_date
            elif r.status_code == 429:
                time.sleep(5)
            else:
                time.sleep(1)
        except Exception as e:
            if "Invalid MAP_KEY" in str(e):
                raise e
            time.sleep(1)

    # Fallback para NRT se SP falhar
    url_nrt = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source_nrt}/{bbox_str}/{day_range}/{start_date}"
    try:
        r = requests.get(url_nrt, timeout=30)
        if r.status_code == 200 and "Invalid MAP_KEY" not in r.text:
            with open(out_file_nrt, "w", encoding="utf-8") as f:
                f.write(r.text)
            return pd.read_csv(out_file_nrt), "NRT", start_date
    except Exception:
        pass

    return pd.DataFrame(), "NONE", start_date

def processar_firms():
    load_dotenv()
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    
    if not map_key or map_key == "sua_chave_nasa_firms_aqui":
        logger.error("=" * 60)
        logger.error("ERRO: FIRMS_MAP_KEY não definida no arquivo .env!")
        logger.error("=" * 60)
        sys.exit(1)
        
    config = carregar_config()
    source_main = config["firms"]["fonte_principal"]
    source_nrt = config["firms"]["fonte_nrt"]
    bbox = config["firms"]["bbox"]
    data_inicio = config["periodo"]["data_inicio"]
    data_fim = config["periodo"]["data_fim"]
    
    raw_dir = Path(config["caminhos"]["raw"]) / "firms"
    interim_dir = Path(config["caminhos"]["interim"])
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        logger.error(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
        sys.exit(1)
    gdf_mun = gpd.read_file(gpkg_path)

    janelas = gerar_janelas_datas(data_inicio, data_fim, step_days=5)
    logger.info(f"Iniciando download paralelo (10 workers) de {len(janelas)} janelas ({data_inicio[:4]}-{data_fim[:4]})...")
    
    dfs = []
    janelas_faltantes = 0
    janelas_nrt = []
    
    # Download paralelo usando ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(baixar_janela_firms, map_key, source_main, source_nrt, bbox, day_range, start_date, raw_dir): (start_date, day_range)
            for start_date, day_range in janelas
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download FIRMS Multithreaded"):
            try:
                df, fonte, s_date = future.result()
                if fonte == "NRT":
                    janelas_nrt.append(s_date)
                if not df.empty and len(df) > 0:
                    dfs.append(df)
                else:
                    janelas_faltantes += 1
            except Exception as e:
                if "Invalid MAP_KEY" in str(e):
                    logger.error("FIRMS MAP_KEY inválida.")
                    sys.exit(1)
                janelas_faltantes += 1
                
    logger.info(f"Download concluído! Registros obtidos em {len(dfs)} janelas. Janelas vazias/sem focos: {janelas_faltantes}.")
    logger.info(f"Auditoria NRT: {len(janelas_nrt)} janelas caíram no fallback NRT.")
    if janelas_nrt:
        logger.warning(f"Lista de janelas em fallback NRT: {sorted(janelas_nrt)}")
    else:
        logger.info("Confirmado: zero janelas caíram no fallback NRT. 100% dos dados vieram da fonte principal SP.")

    if not dfs:
        logger.error("Nenhum dado baixado.")
        sys.exit(1)

    df_raw = pd.concat(dfs, ignore_index=True)
    total_bruto = len(df_raw)
    logger.info(f"Total de registros brutos baixados da API FIRMS: {total_bruto:,}")

    # Remover duplicatas brutas
    df_raw = df_raw.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    logger.info(f"Registros após remoção de duplicatas brutas: {len(df_raw):,}")

    # 3. Aplicar Filtros
    # Filtro 1: Junção Espacial (manter apenas dentro do Pará)
    gdf_focos = gpd.GeoDataFrame(
        df_raw,
        geometry=gpd.points_from_xy(df_raw["longitude"], df_raw["latitude"]),
        crs="EPSG:4326"
    )
    gdf_mun_4326 = gdf_mun.to_crs(epsg=4326)
    
    gdf_joined = gpd.sjoin(gdf_focos, gdf_mun_4326[["cd_mun", "nm_mun", "geometry"]], how="inner", predicate="within")
    total_spat_filt = len(gdf_joined)
    descarte_espacial = total_bruto - total_spat_filt
    logger.info(f"Filtro 1 (Junção Espacial Pará): Mantidos {total_spat_filt:,} focos (Descartados {descarte_espacial:,} fora do Pará).")

    # Filtro 2: tipo == 0 (vegetação presumida)
    df_filt = gdf_joined.copy()
    total_antes_tipo = len(df_filt)
    if "type" in df_filt.columns:
        df_filt = df_filt[df_filt["type"].isna() | (df_filt["type"] == 0)]
        descarte_tipo = total_antes_tipo - len(df_filt)
        logger.info(f"Filtro 2 (type == 0 ou NRT): Mantidos {len(df_filt):,} focos (Descartados {descarte_tipo:,}).")

    # Filtro 3: descartar confidence == 'l' ou 'low'
    total_antes_conf = len(df_filt)
    if "confidence" in df_filt.columns:
        df_filt["confidence_str"] = df_filt["confidence"].astype(str).str.lower().str.strip()
        df_filt = df_filt[~df_filt["confidence_str"].isin(["l", "low"])]
        df_filt.drop(columns=["confidence_str"], inplace=True)
        descarte_conf = total_antes_conf - len(df_filt)
        logger.info(f"Filtro 3 (confidence != low): Mantidos {len(df_filt):,} focos (Descartados {descarte_conf:,}).")

    # 4. Salvar tabela de pontos interim
    df_interim = df_filt.drop(columns=["geometry", "index_right"], errors="ignore")
    out_interim = interim_dir / "firms_focos_pa.parquet"
    df_interim.to_parquet(out_interim, index=False)
    logger.info(f"Focos pontuais salvos em '{out_interim}' com {len(df_interim):,} linhas.")

    # 5. Agregar por Município x Mês com ZEROS EXPLÍCITOS
    df_interim["acq_date"] = pd.to_datetime(df_interim["acq_date"])
    df_interim["mes"] = df_interim["acq_date"].dt.strftime("%Y-%m")
    
    contagem_obs = df_interim.groupby(["cd_mun", "mes"]).size().reset_index(name="focos_count")
    
    todos_muns = sorted(gdf_mun["cd_mun"].astype(str).unique())
    data_inicio_dt = datetime.strptime(data_inicio, "%Y-%m-%d")
    data_fim_dt = datetime.strptime(data_fim, "%Y-%m-%d")
    todos_meses = pd.date_range(start=data_inicio_dt, end=data_fim_dt, freq="MS").strftime("%Y-%m").tolist()
    
    grade_completa = pd.MultiIndex.from_product(
        [todos_muns, todos_meses],
        names=["cd_mun", "mes"]
    ).to_frame().reset_index(drop=True)
    
    df_mensal = pd.merge(grade_completa, contagem_obs, on=["cd_mun", "mes"], how="left")
    df_mensal["focos_count"] = df_mensal["focos_count"].fillna(0).astype(int)
    
    out_processed = processed_dir / "focos_municipio_mes.parquet"
    df_mensal.to_parquet(out_processed, index=False)
    logger.info(f"Tabela mensal município x mês salva em '{out_processed}' ({len(df_mensal):,} linhas).")

    # Totais Anuais
    df_mensal["ano"] = df_mensal["mes"].str.slice(0, 4)
    totais_anuais = df_mensal.groupby("ano")["focos_count"].sum().reset_index()
    logger.info("\n=== TOTAIS ANUAIS DE FOCOS FIRMS (VIIRS S-NPP) PARÁ ===")
    for _, row in totais_anuais.iterrows():
        logger.info(f"Ano {row['ano']}: {row['focos_count']:,} focos")
    logger.info("========================================================\n")
    
    return df_mensal, df_interim

if __name__ == "__main__":
    processar_firms()
