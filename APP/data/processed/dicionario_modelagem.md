# Dicionário de Dados da Base de Modelagem APHIM (`aphim_modelagem.parquet`)

Este documento descreve detalhadamente cada uma das 43 colunas presentes na tabela final de modelagem do projeto **APHIM**.

## Resumo da Estrutura Temporal
- **Unidade de análise:** Par (município, mês de emissão $t$).
- **Mês de emissão ($t$):** 2019-01 a 2025-11 (83 meses x 144 municípios = **11.952 linhas**).
- **Mês alvo ($t+1$):** 2019-02 a 2025-12.
- **Partições:**
  - `aquecimento`: 2019-01 a 2019-12 (1.728 linhas) - utilizado para cálculo de defasagens; descartado do treino.
  - `treino`: 2020-01 a 2023-11 (6.768 linhas) - alvos de 2020-02 a 2023-12.
  - `teste`: 2023-12 a 2025-11 (3.456 linhas) - alvos de 2024-01 a 2025-12.
- **Regra de ouro temporal:** Nenhuma variável preditiva utiliza dado posterior ao fechamento do mês de emissão $t$.

## Tabela de Metadados

| Coluna | Tipo | Descrição | Fórmula / Origem | Escopo do Modelo |
| :--- | :---: | :--- | :--- | :--- |
| `cd_mun` | **Identificação** | Código IBGE oficial do município (7 dígitos) | `Identificador geográfico` | Todas |
| `nm_mun` | **Identificação** | Nome oficial do município | `IBGE Malha Municipal 2024` | Todas |
| `mes_emissao` | **Identificação** | Mês t de fechamento e emissão da priorização (AAAA-MM) | `Data limite de observação` | Todas |
| `mes_alvo` | **Identificação** | Mês t+1 de previsão e ação preventiva (AAAA-MM) | `Mês subsequente a mes_emissao` | Todas |
| `conjunto` | **Identificação** | Partição cronológica da modelagem ('aquecimento', 'treino', 'teste') | `aquecimento: 2019; treino: 2020-01 a 2023-11; teste: 2023-12 a 2025-11` | Todas |
| `y` | **Alvo** | Contagem real de focos de calor observados no mês alvo t+1 | `focos_count do município em mes_alvo (NASA FIRMS)` | Todas |
| `base_rf_inpe_media_t` | **Baseline** | Média zonal do Risco de Fogo do INPE no mês de emissão t | `rf_inpe_media em t (não usado como feature)` | Baselines |
| `focos_t` | **Atributo** | Contagem de focos de calor observados no mês de emissão t | `focos_count em t` | Todos os Modelos |
| `focos_t1` | **Atributo** | Contagem de focos de calor no mês anterior t-1 | `focos_count em t-1` | Todos os Modelos |
| `focos_t2` | **Atributo** | Contagem de focos de calor em t-2 | `focos_count em t-2` | Todos os Modelos |
| `focos_mm3` | **Atributo** | Média móvel de focos dos últimos 3 meses (t, t-1, t-2) | `(focos_t + focos_t1 + focos_t2) / 3` | Todos os Modelos |
| `focos_mesmo_mes_ano_ant` | **Atributo** | Focos de calor no mesmo mês do ano anterior ao alvo (t-11) | `focos_count em (mes_alvo - 12 meses) = t-11` | Todos os Modelos |
| `focos_media_hist_mes` | **Atributo** | Média histórica de focos no mesmo mês de calendário do alvo | `Média de focos no mês calendário do alvo para anos estritamente < ano(alvo)` | Todos os Modelos |
| `dens_t` | **Atributo** | Densidade de focos em t por km² | `focos_t / area_km2` | Todos os Modelos |
| `dens_t1` | **Atributo** | Densidade de focos em t-1 por km² | `focos_t1 / area_km2` | Todos os Modelos |
| `dens_t2` | **Atributo** | Densidade de focos em t-2 por km² | `focos_t2 / area_km2` | Todos os Modelos |
| `dens_mm3` | **Atributo** | Densidade da média móvel de 3 meses por km² | `focos_mm3 / area_km2` | Todos os Modelos |
| `dens_mesmo_mes_ano_ant` | **Atributo** | Densidade de focos no mesmo mês do ano anterior por km² | `focos_mesmo_mes_ano_ant / area_km2` | Todos os Modelos |
| `dens_media_hist_mes` | **Atributo** | Densidade da média histórica do mês alvo por km² | `focos_media_hist_mes / area_km2` | Todos os Modelos |
| `prec_mm` | **Atributo** | Precipitação mensal acumulada no mês de emissão t (mm) | `ERA5-Land total_precipitation em t` | Todos os Modelos |
| `t2m_max_c` | **Atributo** | Temperatura máxima diária média a 2m no mês t (°C) | `ERA5-Land temperature_2m_max em t` | Todos os Modelos |
| `ur_pct` | **Atributo** | Umidade relativa média do ar a 2m no mês t (%) | `ERA5-Land deduzida por Magnus-Tetens em t` | Todos os Modelos |
| `dpv_kpa` | **Atributo** | Déficit de pressão de vapor saturado no mês t (kPa) | `ERA5-Land es(T) - ea(Td) em t` | Todos os Modelos |
| `umid_solo` | **Atributo** | Umidade volumétrica do solo na camada superficial 0-7 cm (m³/m³) | `ERA5-Land volumetric_soil_water_layer_1 em t` | Todos os Modelos |
| `lst_c` | **Atributo** | Temperatura da superfície terrestre diurna no mês t (°C) | `ERA5-Land skin_temperature em t` | Todos os Modelos |
| `vento_ms` | **Atributo** | Velocidade média do vento a 10m no mês t (m/s) | `ERA5-Land sqrt(u10² + v10²) em t` | Todos os Modelos |
| `anom_prec_mm` | **Atributo** | Anomalia de precipitação em t em relação à normal 1991-2020 | `prec_mm(t) - normal_prec(mês calendário)` | Todos os Modelos |
| `anom_t2m_media_c` | **Atributo** | Anomalia de temperatura média a 2m em t em relação à normal | `t2m_media_c(t) - normal_t2m(mês calendário)` | Todos os Modelos |
| `anom_ur_pct` | **Atributo** | Anomalia de umidade relativa em t em relação à normal | `ur_pct(t) - normal_ur(mês calendário)` | Todos os Modelos |
| `anom_umid_solo` | **Atributo** | Anomalia de umidade do solo em t em relação à normal | `umid_solo(t) - normal_umid(mês calendário)` | Todos os Modelos |
| `anom_lst_c` | **Atributo** | Anomalia de LST em t em relação à normal | `lst_c(t) - normal_lst(mês calendário)` | Todos os Modelos |
| `prec_3m` | **Atributo** | Precipitação acumulada dos últimos 3 meses (t, t-1, t-2) | `prec_mm(t) + prec_mm(t-1) + prec_mm(t-2)` | Todos os Modelos |
| `anom_prec_3m` | **Atributo** | Anomalia de precipitação acumulada de 3 meses | `prec_3m - soma das normais dos 3 meses de calendário` | Todos os Modelos |
| `mes_sin` | **Atributo** | Componente seno da sazonalidade do mês alvo t+1 | `sin(2 * pi * mes_alvo_num / 12)` | Todos os Modelos |
| `mes_cos` | **Atributo** | Componente cosseno da sazonalidade do mês alvo t+1 | `cos(2 * pi * mes_alvo_num / 12)` | Todos os Modelos |
| `area_km2` | **Estrutural** | Área territorial oficial do município em km² | `IBGE Malha Municipal 2024 / SIRGAS 2000` | Só para o híbrido C |
| `pct_floresta` | **Estrutural** | Cobertura florestal (%) com defasagem anual t-1 | `MapBiomas Col. 10 (ano = ano(t) - 1)` | Só para o híbrido C |
| `pct_pastagem` | **Estrutural** | Área de pastagem (%) com defasagem anual t-1 | `MapBiomas Col. 10 (ano = ano(t) - 1)` | Só para o híbrido C |
| `pct_agricultura` | **Estrutural** | Área agrícola consolidada (%) com defasagem anual t-1 | `MapBiomas Col. 10 (ano = ano(t) - 1)` | Só para o híbrido C |
| `pct_solo_exposto` | **Estrutural** | Área de solo exposto/degradado (%) com defasagem anual t-1 | `MapBiomas Col. 10 (ano = ano(t) - 1)` | Só para o híbrido C |
| `desmat_km2_prodes` | **Estrutural** | Desmatamento anual acumulado em km² com defasagem anual t-1 | `PRODES Amazônia (ano = ano(t) - 1)` | Só para o híbrido C |
| `desmat_pct_area_prodes` | **Estrutural** | Desmatamento anual em relação à área total (%) com defasagem t-1 | `PRODES Amazônia / area_km2 * 100` | Só para o híbrido C |
| `renda_pc` | **Estrutural** | Renda domiciliar per capita mediana em R$ | `Censo Demográfico 2022 (IBGE SIDRA)` | Só para o híbrido C |
