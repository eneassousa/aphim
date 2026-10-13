import sys
import subprocess
from pathlib import Path

def run_script(module_name, desc):
    print("\n" + "=" * 70)
    print(f" EXECUTANDO: {desc} ({module_name})")
    print("=" * 70)
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        print(f"\n[ERRO] O script {module_name} falhou com código de saída {result.returncode}.")
        return False
    return True

def main():
    print("Iniciando o Pipeline APHIM - Coleta e Validação Inicial\n")
    
    scripts = [
        ("src.coleta.01_ibge", "01 — Malha Municipal IBGE"),
        ("src.coleta.02_firms", "02 — Focos de Calor NASA FIRMS (VIIRS S-NPP)"),
        ("src.coleta.03_auditoria_inpe", "03 — Auditoria INPE BDQueimadas vs FIRMS"),
        ("src.coleta.04_era5", "04 — Clima ERA5-Land via Google Earth Engine"),
        ("src.coleta.05_mapbiomas", "05 — Uso e Cobertura do Solo MapBiomas (Coleção 10)"),
        ("src.coleta.06_prodes", "06 — Desmatamento Anual PRODES Amazônia"),
        ("src.coleta.07_renda", "07 — Renda Municipal per Capita (Censo 2022 SIDRA)"),
        ("src.coleta.08_risco_fogo", "08 — INPE Risco de Fogo Mensal"),
        ("src.coleta.09_tabela_final", "09 — Montagem da Tabela Final Integrada APHIM"),
        ("src.processamento.01_atributos_alvo", "Processamento 01 — Engenharia de Atributos, Alvo e Divisão Temporal"),
        ("src.processamento.02_baselines", "Processamento 02 — Baselines de Priorização e Avaliação Comparativa"),
        ("src.processamento.03_modelo_perigo", "Processamento 03 — Módulo de Perigo e Modelagem Preditiva"),
        ("src.processamento.04_excesso", "Processamento 04 — Análise de Excesso e Rankings de Anomalia")
    ]
    
    for mod, desc in scripts:
        success = run_script(mod, desc)
        if not success:
            print(f"\nPipeline interrompido no módulo {mod}.")
            sys.exit(1)
            
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETO APHIM EXECUTADO COM SUCESSO!")
    print(" BASE DE MODELAGEM: data/processed/aphim_modelagem.parquet")
    print(" RESULTADOS BASELINES: data/processed/resultados/baselines_resumo.csv")
    print(" RESULTADOS MODELOS PERIGO: data/processed/resultados/perigo_resumo.csv")
    print(" RESULTADOS ANÁLISE EXCESSO: data/processed/resultados/excesso_resumo.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
