# APHIM - Arquitetura Preditiva Híbrida com Inteligência Computacional e Modelagem Multicritério

Repositório da dissertação de mestrado em Ciência da Computação (**UFPA**) focada na predição e classificação de risco de focos de queimadas e incêndios florestais nos municípios do Pará (2019–2025).

---

## 🏛️ Estrutura do Repositório

O projeto é organizado nos seguintes módulos:

```text
├── APP/               # Pipeline de Coleta, Engenharia de Recursos e Modelagem
│   ├── config.yaml    # Configurações globais (datas, limites geográficos, bounding boxes)
│   ├── requirements.txt # Dependências do ecossistema Python
│   ├── run_all.py     # Script mestre para execução sequencial do pipeline
│   ├── src/           # Códigos de coleta (IBGE, FIRMS/NASA, ERA5, MapBiomas, PRODES, INPE)
│   ├── data/          # Dados organizados (raw, interim, processed)
│   └── DOC/           # Documentações técnicas e metodológicas
│
├── dashboard/         # Aplicação Frontend Interativa para visualização e análise de risco
│   ├── package.json   # Dependências Node.js / React / Vite
│   ├── src/           # Componentes, mapas e dashboards analíticos
│   └── public/        # Recursos estáticos
│
├── Artigo_IA_CD_v5.pdf # Artigo científico associado à pesquisa
└── Classificação de Riscos.pdf # Metodologia de classificação de riscos
```

---

## 🚀 Como Começar

### 1. Pipeline de Dados (`APP/`)
Consulte o guia completo em [APP/README.md](APP/README.md).

```bash
cd APP
python -m venv venv
# No Windows:
.\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env # Configure sua FIRMS_MAP_KEY e GEE_PROJECT
python run_all.py
```

### 2. Painel Interativo (`dashboard/`)
Consulte [dashboard/README.md](dashboard/README.md).

```bash
cd dashboard
npm install
npm run dev
```

---

## 📄 Licença e Autoria
Desenvolvido no âmbito do Programa de Pós-Graduação em Ciência da Computação - UFPA.
Autor: Enéas Sousa.
