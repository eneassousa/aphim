# APHIM - Pipeline de Coleta e Montagem da Base de Dados

Este repositório contém a implementação do pipeline automático de coleta, limpeza, agregação e validação da base de dados do projeto **APHIM** (Arquitetura Preditiva Híbrida com Inteligência computacional e Modelagem multicritério), dissertação de mestrado em Ciência da Computação (UFPA) sobre predição e classificação de risco de focos de queimadas nos municípios do Pará (2019-2025).

---

## 📁 Estrutura do Projeto

```
APP/
├── config.yaml               # Configurações globais (período, UF, bbox, filtros)
├── .env.example              # Exemplo de variáveis de ambiente
├── .env                      # Variáveis de ambiente locais (chaves da API)
├── requirements.txt          # Dependências do projeto Python
├── run_all.py                # Executador sequencial de todo o pipeline
├── README.md                 # Documentação e instruções de execução
├── src/
│   ├── utils/
│   │   └── logger.py         # Configuração centralizada de logs
│   ├── coleta/
│   │   ├── 01_ibge.py        # Malha municipal e áreas do Pará (UF 15)
│   │   ├── 02_firms.py       # Focos de calor NASA FIRMS (VIIRS S-NPP)
│   │   └── 03_auditoria_inpe.py # Auditoria com INPE BDQueimadas (Sat. Referência)
│   └── processamento/
├── data/
│   ├── raw/                  # Dados brutos baixados por fonte
│   ├── interim/              # Tabelas intermediárias com dados pontuais filtrados
│   └── processed/            # Tabelas finais agregadas e validadas (GeoPackage/Parquet)
└── logs/                     # Arquivos de log detalhados por script
```

---

## ⚙️ Configuração do Ambiente

### 1. Pré-requisitos
- **Python 3.11+**
- Instale as dependências executando:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Configurar o arquivo `.env`
Crie ou edite o arquivo `.env` na raiz da pasta `APP`:
```env
# NASA FIRMS MAP_KEY (Obtenha gratuitamente em: https://firms.modaps.eosdis.nasa.gov/api/map_key/)
FIRMS_MAP_KEY=sua_chave_aqui

# Google Earth Engine Project ID (Projeto com a API Earth Engine ativada)
GEE_PROJECT=seu_projeto_gee
```

### 3. Autenticação no Google Earth Engine
Antes de executar as etapas climáticas e ambientais via Earth Engine, faça login com sua conta Google executando no terminal:
```bash
earthengine authenticate
```
Isso abrirá uma janela no navegador para autorizar o acesso à API do Earth Engine.

---

## 🚀 Como Executar

Você pode rodar todo o pipeline de uma só vez ou executar os scripts individualmente.

### Opção A: Executar todo o pipeline sequencialmente
```bash
python run_all.py
```

### Opção B: Executar scripts individualmente

1. **Malha Municipal e Áreas IBGE (144 Municípios do PA)**
   ```bash
   python -m src.coleta.01_ibge
   ```
   *Saída:* `data/processed/municipios_pa.gpkg`

2. **Focos de Calor NASA FIRMS (VIIRS S-NPP)**
   ```bash
   python -m src.coleta.02_firms
   ```
   *Saídas:* `data/interim/firms_focos_pa.parquet` e `data/processed/focos_municipio_mes.parquet`

3. **Auditoria com INPE BDQueimadas (Satélite de Referência)**
   ```bash
   python -m src.coleta.03_auditoria_inpe
   ```
   *Saídas:* `data/processed/auditoria_firms_vs_inpe.csv` e relatório no console.

4. **Clima ERA5-Land via Google Earth Engine**
   ```bash
   python -m src.coleta.04_era5
   ```
   *Saídas:*
   - `data/processed/clima_municipio_mes.parquet` (13.824 linhas: 144 municípios × 96 meses [2018–2025])
   - `data/processed/clima_normal_1991_2020.parquet` (1.728 linhas: 144 municípios × 12 meses do calendário)

5. **Uso e Cobertura do Solo MapBiomas (Coleção 10)**
   ```bash
   python -m src.coleta.05_mapbiomas
   ```
   *Saída:* `data/processed/mapbiomas_municipio_ano.parquet` (1.008 linhas: 144 municípios × 7 anos [2018–2024])

6. **Desmatamento Anual PRODES Amazônia (INPE TerraBrasilis)**
   ```bash
   python -m src.coleta.06_prodes
   ```
   *Saída:* `data/processed/prodes_municipio_ano.parquet` (1.152 linhas: 144 municípios × 8 anos [2018–2025])

7. **Vulnerabilidade Social: Renda per Capita Censo 2022 (IBGE SIDRA)**
   ```bash
   python -m src.coleta.07_renda
   ```
   *Saída:* `data/processed/renda_municipio.parquet` (144 linhas: 144 municípios)

8. **Risco de Fogo Meteorológico Mensal (INPE BDQueimadas)**
   ```bash
   python -m src.coleta.08_risco_fogo
   ```
   *Saída:* `data/processed/risco_fogo_inpe_municipio_mes.parquet` (12.096 linhas: 144 municípios × 84 meses [2019–2025])

9. **Montagem da Tabela Final Integrada APHIM**
   ```bash
   python -m src.coleta.09_tabela_final
   ```
   *Saídas:*
   - `data/processed/aphim_base_municipio_mes.parquet` (12.096 linhas × 44 colunas)
   - `data/processed/aphim_base_municipio_mes.csv`

---

## 📊 Validações Automáticas

- **IBGE (01):** Garante exatamente 144 municípios e área total de ~1.247.753 km² (dentro da margem de ±1% da área oficial do Pará de ~1.245.870 km²).
- **FIRMS (02):** Garante cobertura de 100% das janelas de 5 dias entre 2019 e 2025 e constrói a grade completa de 144 municípios × 84 meses (12.096 linhas com zeros explícitos).
- **Auditoria INPE (03):** Baixa o satélite de referência do INPE (249.210 focos no Pará de 2019 a 2025 com junção espacial) e calcula correlação de Pearson (0,9822), Spearman (0,9898), meses de pico e razão FIRMS/INPE.
- **ERA5-Land (04):** Garante 13.824 linhas sem duplicatas e sem nulos; valida limites físicos do Pará (temperatura, chuva, umidade e solo); valida sazonalidade hídrica (Fev/Mar/Abr > Jul/Ago/Set) e correlação negativa entre chuva e focos de calor.
- **MapBiomas (05):** Garante 1.008 linhas sem duplicatas e sem nulos; valida soma das classes de uso entre 98% e 102%; valida coerência histórica da floresta (77% para 74%) e pastagem (15% para 17%).
- **PRODES (06):** Garante 1.152 linhas com zeros explícitos e sem nulos; valida consistência de ordem de grandeza e tendência com a taxa oficial do INPE.
