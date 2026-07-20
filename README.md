# Classificação de Gênero Musical em Plataformas de Streaming

Aplicação de técnicas de **Machine Learning** para classificar automaticamente o gênero de músicas a partir de suas características acústicas, comparando três algoritmos (**Árvore de Decisão**, **Random Forest** e **Naive Bayes**) implementados em **duas linguagens**: Python e R.

Trabalho da disciplina de Ciência de Dados do curso de Análise e Desenvolvimento de Sistemas da **Universidade Católica de Brasília (UCB)**, 2026.

**Orientador:** William Roberto Malvezzi

**Autores:**

- Maria Eduarda Marques
- Ramon Miguel Ataides
- Louie Nery Silva
- Gabryella Santos Pinho
- Nathanael Victor Magno

---

## Objetivo

Com o crescimento das plataformas de streaming, a classificação manual de faixas em gêneros tornou-se impraticável. Este projeto investiga o quanto modelos de aprendizado de máquina conseguem categorizar músicas em gêneros (Pop, Rock, Latin, R&B, EDM, ...) usando apenas atributos de áudio, e compara o comportamento dos mesmos modelos em **Python** e **R**.

## Base de dados

**Spotify Music Dataset** — Solomon Ameh, Kaggle (2022).
🔗 https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset/

O dataset reúne metadados e características acústicas extraídas pela [Web API oficial do Spotify](https://developer.spotify.com/documentation/web-api/reference/get-audio-features). São usados dois arquivos, tratados como um único problema de classificação:

| Arquivo | Descrição |
|---|---|
| `high_popularity_spotify_data.csv` | Faixas de alta popularidade |
| `low_popularity_spotify_data.csv`  | Faixas de baixa popularidade |

**Variável alvo:** `playlist_genre` (gênero da faixa).

**Principais atributos (features):**

- **Acústicos:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `key`, `mode`, `time_signature`
- **Derivados:** `duration_minutes` (a partir de `duration_ms`), `release_year` (a partir da data de lançamento)
- **Auxiliares:** `track_popularity`, `playlist_subgenre`

## Metodologia

1. **Carga e união** dos dois CSVs, com marcação da origem (`source_dataset`).
2. **Features derivadas:** duração em minutos e ano de lançamento.
3. **Remoção de duplicatas** por `track_id` — a mesma faixa aparece em várias playlists; sem isso, treino e teste deixam de ser uma comparação confiável.
4. **Pré-processamento** via pipeline: imputação de valores ausentes e *one-hot encoding* das variáveis categóricas.
5. **Split estratificado** treino/teste (80/20, `random_state = 42`).
6. **Treino e avaliação** dos três algoritmos, medindo *accuracy*, *precision*, *recall* e *F1-score*.

## Resultados

### Python

| Algoritmo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree | 25% | 54% | 25% | 26% |
| Random Forest | 76% | 81% | 76% | 77% |
| **Naive Bayes** | **80%** | **87%** | **80%** | **81%** |

### R

| Algoritmo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree | 31% | 33% | 13% | 38% |
| **Random Forest** | **81%** | **82%** | 60% | 70% |
| Naive Bayes | 72% | 65% | 66% | 66% |

**Leitura dos resultados:**

- O **Random Forest** foi o modelo mais robusto e estável entre as duas linguagens — em R alcançou a maior acurácia (≈ 81%), embora com queda de *recall* (≈ 60%).
- O **Naive Bayes** teve o melhor equilíbrio geral em Python (F1 ≈ 81%), mas caiu em R, mostrando sensibilidade à implementação e aos parâmetros padrão das bibliotecas.
- A **Árvore de Decisão** teve o pior desempenho em ambos os ambientes, com sinais de *overfitting* e baixa generalização.

O trabalho reforça que o resultado final depende não só da escolha do algoritmo, mas também do **pré-processamento rigoroso** (deduplicação, tratamento de ausentes) e do ambiente/bibliotecas utilizados.

## Estrutura do repositório

```
.
├── python/                  # Implementação em Python (pandas + scikit-learn)
│   ├── main.py              # Pipeline completa — Random Forest
│   ├── decision_tree.py     # Árvore de Decisão
│   ├── naive_bayes.py       # Naive Bayes
│   ├── tratamentos.py       # Funções de tratamento de dados
│   ├── test_main.py         # Testes
│   └── data/cru/            # CSVs brutos do Spotify Music Dataset
├── r/                       # Implementação em R (tidyverse + caret)
│   ├── dataset/             # CSVs + pipeline de tratamento
│   ├── decision_tree/
│   ├── naive_bayes/
│   └── random_forest/
└── docs/                    # Relatório, apêndice e apresentação (PDF)
```

## Como executar

### Python

```bash
cd python
pip install pandas scikit-learn joblib
python main.py            # treina e avalia o Random Forest (pipeline completa)
python decision_tree.py   # Árvore de Decisão
python naive_bayes.py     # Naive Bayes
```

Os artefatos gerados (modelos `.joblib`, métricas e dados processados) ficam em `python/artifacts/` e `python/data/` e **não são versionados** — são reproduzidos ao rodar os scripts.

### R

Os scripts em R foram desenvolvidos no **RStudio** e usam `rstudioapi` para resolver o diretório de trabalho. Pacotes necessários:

```r
install.packages(c("tidyverse", "caret", "naivebayes", "rpart"))
```

Abra os scripts em `r/` no RStudio e execute. Ajuste os caminhos dos dados conforme a sua estrutura local, se necessário.

## Documentação

O detalhamento completo está em [`docs/`](docs/):

- **[Relatório](docs/Relatorio.pdf)** — artigo completo (fundamentação, metodologia, resultados e discussão)
- **[Apêndice](docs/Apendice.pdf)**
- **[Apresentação](docs/Apresentacao.pdf)**

## Referências

- AMEH, Solomon. *Spotify Music Dataset*. Kaggle, 2022. Disponível em: https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset/
- SPOTIFY. *Web API Reference — Get Audio Features*. Spotify for Developers, 2024. Disponível em: https://developer.spotify.com/documentation/web-api/reference/get-audio-features
- TJOA, Steve. *Introduction to Music Information Retrieval*. Disponível em: https://musicinformationretrieval.com/intro.html
