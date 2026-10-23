# Registro de Congelamento dos Modelos — Avaliação em 2026

**Data:** 23 de outubro de 2026  
**Commit de Referência:** `aba2b7b36488d86bdcae3be23fb273585784e56d` (`aba2b7b`)  
**Tag:** `modelos-congelados-2026`

---

## 1. Objetivo e Diretrizes de Congelamento

Este documento registra formalmente o congelamento das arquiteturas, conjuntos de atributos, hiperparâmetros, regras de transição hierárquica e procedimentos de treino dos modelos preditivos do projeto **APHIM** antes de qualquer acesso, coleta ou avaliação em dados do ano de 2026.

**Regra de Ouro:** Não há ajuste de hiperparâmetros, seleção de atributos nem alteração de métricas nesta etapa. Os modelos congelados são avaliados diretamente nos meses de 2026 sem qualquer intervenção corretiva ("sem conserto a posteriori").

---

## 2. Nível 1: Modelo Municipal (D0)

- **Algoritmo:** XGBoost Regressor (`objective="count:poisson"`).
- **Atributos (20 features):** Definidos a partir de `data/processed/resultados/features_v2_permutacao.csv`:
  1. `focos_media_hist_mes`
  2. `focos_t`
  3. `focos_t1`
  4. `mes_sin`
  5. `focos_vizinhos_t`
  6. `mes_cos`
  7. `prec_6m`
  8. `anom_prec_6m`
  9. `razao_fogo_t`
  10. `anom_prec_mm`
  11. `anom_t2m_media_c`
  12. `focos_t2`
  13. `variacao_anual_t`
  14. `umid_solo_t1`
  15. `razao_fogo_vizinhos_t`
  16. `anom_umid_solo`
  17. `anom_umid_solo_3m`
  18. `anom_prec_3m`
  19. `meses_secos_seguidos`
  20. `prec_3m`
- **Hiperparâmetros:** Registrados na linha D0 de `deter_validacao_municipio.csv`:
  ```json
  {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1
  }
  ```
- **Métrica Primária:** Captura@15 (fração de focos do estado concentrada nos 15 municípios com maior previsão).

---

## 3. Nível 2: Modelo da Grade Celular de 5 km (G2)

- **Algoritmo:** XGBoost Regressor (`objective="count:poisson"`).
- **Atributos (27 features):**
  - **Base Celular (16 features):**
    - `focos_t`, `focos_t1`, `focos_t2`, `focos_mm3`, `focos_mesmo_mes_ano_ant`, `focos_media_hist_mes`
    - `focos_vizinhos_t`, `focos_vizinhos_media_hist_mes`
    - `mes_sin`, `mes_cos`
    - `focos_mun_t`
    - `prec_mm`, `ur_pct`, `dpv_kpa`, `umid_solo`, `anom_prec_mm`
  - **DETER Celular (5 features):**
    - `deter_desmat_cel_3m`, `deter_desmat_cel_12m`
    - `deter_degrad_cel_3m`, `deter_degrad_cel_12m`
    - `deter_desmat_viz_12m`
  - **Recorrência Celular (6 features):**
    - `meses_desde_ultimo_foco`: meses decorridos desde o último foco ($A=1$) na célula até $t$ (máximo/sem histórico = 120).
    - `anos_com_fogo_3a`: número de janelas anuais (entre as 3 anteriores) com ao menos 1 foco (0 a 3).
    - `frac_meses_fogo_24m`: fração dos últimos 24 meses ($t-23$ a $t$) com foco.
    - `focos_janela_retorno`: total de focos da célula entre $t-12$ e $t-10$.
    - `taxa_retorno_emp`: probabilidade empírica de foco em $t+1$ calculada nas faixas temporais e mês de calendário do alvo.
    - `frac_meses_fogo_24m_viz`: média espacial de `frac_meses_fogo_24m` nas 8 células vizinhas.
- **Regra da Taxa de Retorno Empírica (`taxa_retorno_emp`):**
  - Faixas de `meses_desde_ultimo_foco`: `0`, `1`, `2`, `3-5`, `6-9`, `10-12`, `13-24`, `25+`, `nunca`.
  - Agrupamento por (faixa, mês do calendário do alvo $t+1$).
  - Estimador suavizado com Laplace: $\frac{\text{focos} + 1}{n + 2}$.
  - Strictly sem vazamento: calculada unicamente sobre o conjunto de treino histórico.
- **Hiperparâmetros:** Registrados na linha G2 de `recorrencia_validacao_grade.csv`:
  ```json
  {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_weight": 10
  }
  ```
- **Métricas Primárias:**
  - CapturaÁrea@10% nas células dos 15 municípios priorizados pelo D0 em cada mês.
  - Métricas secundárias: CapturaÁrea@5% e CapturaÁrea@20%.
  - Captura ponta a ponta (fração dos focos totais do Pará capturados nas células priorizadas).

---

## 4. Modelos Comparadores para 2026

1. **Grade Original (G0):**
   - 16 atributos de base.
   - Hiperparâmetros: `max_depth=4, learning_rate=0.1, n_estimators=600, min_child_weight=1`.
2. **Grade com DETER (G1):**
   - 21 atributos (base + 5 de DETER).
   - Hiperparâmetros: `max_depth=4, learning_rate=0.1, n_estimators=300, min_child_weight=1`.
3. **Baselines no Nível 1 (Município):**
   - Climatologia municipal (média histórica por mês de calendário).
   - Persistência municipal (focos observados em $t$).
   - Ranking histórico fixo municipal (soma acumulada histórica de focos).
4. **Baselines no Nível 2 (Célula):**
   - Histórico fixo da célula (focos acumulados na célula até o fim do treino).
   - Climatologia da célula (média histórica da célula naquele mês do calendário).
   - Persistência da célula (focos observados na célula em $t$).
   - Oráculo (limite teórico superior com conhecimento perfeito dos focos em $t+1$).

---

## 5. Regimes de Treino Avaliados em 2026

- **Regime Principal ("T25" — Operacional):**
  - Modelo retreinado com todos os dados disponíveis até **2025-12** (alvos de 2020-02 a 2025-12).
  - Tabela `taxa_retorno_emp` recalculada com todo o histórico de treino até 2025-12.
  - Representa a aplicação real do sistema em produção em 2026.
- **Regime Secundário ("T23" — Estudo Comparativo Estrito):**
  - Modelos exatamente como avaliados no período de teste 2024–2025, com alvos de treino até **2023-12**.
  - Tabela `taxa_retorno_emp` mantida estritamente com histórico até 2023-12 (`taxa_retorno_final.csv`).
  - Permite verificar a robustez temporal sem atualização anual.

---

## 6. Procedimento de Teste Estatístico

- Diferença média pareada de captura.
- Intervalo de confiança de 95% via *bootstrap* com 10.000 reamostragens mensais (semente 42).
- Teste de postos sinalizados de Wilcoxon pareado bicaudal.
- Correção de multiplicidade via Holm-Bonferroni (step-down) dentro de cada nível hierárquico.
- Contagem do número de meses em que a configuração proposta superou cada comparador.
- Relatório desagregado por sazonalidade (estação chuvosa: jan–jun; estação seca: jul–dez).
