# TCC S&OP Modular

Pipeline modular e reprodutível para um Trabalho de Conclusão de Curso (TCC) em Sales & Operations Planning (S&OP), cobrindo previsão de demanda, planejamento agregado, RRP e desagregação, com CLI, logging e testes.

## Arquitetura

```
TCC_SOP_Modular/
├─ README.md
├─ Makefile
├─ requirements.txt
├─ pyproject.toml
├─ setup.cfg
├─ configs/
│  ├─ config.yaml
│  └─ logging.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/
├─ reports/
│  ├─ figures/
│  └─ tables/
├─ src/
│  ├─ core/
│  ├─ data_prep/
│  ├─ forecast/
│  ├─ optimization/
│  ├─ rrp/
│  ├─ disaggregation/
│  ├─ viz/
│  └─ cli.py
├─ notebooks/
│  └─ 00_quickstart.ipynb
└─ tests/
```

Cada domínio possui módulos dedicados com docstrings, type hints e logging. As saídas (tabelas, gráficos, modelos) são direcionadas aos diretórios definidos no YAML.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows
make install
```

## Execução via CLI

Todos os comandos aceitam `--config configs/config.yaml`.

```bash
python -m src.cli prepare-data      # pré-processa dados e gera features
python -m src.cli train             # treina modelos e salva campeão por família
python -m src.cli evaluate          # avalia no hold-out e gera gráficos
python -m src.cli forecast          # gera demanda prevista para o horizonte configurado
python -m src.cli optimize          # resolve plano agregado (PuLP)
python -m src.cli rrp               # calcula RRP e utilização de capacidade
python -m src.cli disaggregate      # desagrega plano agregado por produto
python -m src.cli all               # executa o pipeline fim-a-fim
```

## Saídas Geradas

- `reports/tables/metricas_modelos.csv`: ranking dos modelos por família (cross validation).
- `reports/tables/avaliacao_modelos.csv`: métricas no conjunto de teste.
- `reports/tables/demanda_prevista.csv`: horizonte futuro previsto.
- `reports/tables/plano_agregado.csv`: plano de produção otimizado.
- `reports/tables/rrp_resumo.csv`: resumo de capacidade vs horas requeridas.
- `reports/tables/plano_desagregado.csv`: plano por produto.
- `reports/figures/*.png`: gráficos de previsão, resíduos, plano agregado e utilização.
- `models/`: artefatos treinados (modelos e scalers).

## Dados Sintéticos x Dados Reais

O pipeline procura `data/raw/vendas_historicas.csv`. Se não existir, gera automaticamente uma série sintética com tendência, sazonalidade e ruído, registrando aviso no log. Para usar dados reais, substitua o arquivo por um CSV com colunas mínimas:

- `data` no formato `YYYY-MM-DD`
- `familia`
- `produto`
- `quantidade`

O esquema é validado durante a etapa `prepare-data`.

## Desenvolvimento e Testes

- `make lint` executa `flake8` com regras definidas em `setup.cfg`.
- `make test` roda `pytest` (smoke tests de data prep, treino, otimização e RRP).
- Notebook `notebooks/00_quickstart.ipynb` traz uma visão exploratória inicial.

## Configuração

`configs/config.yaml` centraliza caminhos, seed, paralelismo e parâmetros de:

- preparação de dados (lags, janelas, geração sintética);
- previsão (horizonte, folds, grids);
- otimização (custos, capacidades, estoque inicial);
- RRP (produtividade, capacidade);
- desagregação (proporções por produto).

Logging estruturado (console + arquivo) é definido em `configs/logging.yaml`.
# Forecast
