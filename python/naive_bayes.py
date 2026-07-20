import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Configuracao do projeto

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "cru"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SPLIT_DATA_DIR = BASE_DIR / "data" / "splits"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
MODELS_DIR = BASE_DIR / "artifacts" / "models"

TRACK_ID_COLUMN = "track_id"
TRACK_NAME_COLUMN = "track_name"
POPULARITY_COLUMN = "track_popularity"

# Alvo: gênero da música (classificação multiclasse)
TARGET_COLUMN = "playlist_genre"

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_minutes",
    "release_year",
    "track_popularity",
]

CATEGORICAL_FEATURES = [
    "playlist_subgenre",  # Subgênero permanece como feature auxiliar
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# dict.fromkeys preserva a ordem e elimina duplicatas
# (track_popularity aparece em POPULARITY_COLUMN e em NUMERIC_FEATURES)
DATASET_COLUMNS = list(dict.fromkeys([
    TRACK_ID_COLUMN,
    TRACK_NAME_COLUMN,
    "source_dataset",
    POPULARITY_COLUMN,
    TARGET_COLUMN,
    *MODEL_FEATURES,
]))

# Grade de busca para var_smoothing — valores em escala logaritmica,
# Espelha a filosofia de busca de hiperparametros do Random Forest em main.py.
VAR_SMOOTHING_GRID = [10 ** exp for exp in np.arange(-11, -5, 0.5)]

# Validacao cruzada estratificada: 5 folds, preservando a proporcao das classes.
CV_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def create_output_folders() -> None:
    """Cria as pastas usadas para salvar dados tratados e artefatos do modelo."""
    for folder in (PROCESSED_DATA_DIR, SPLIT_DATA_DIR, METRICS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_raw_spotify_data() -> pd.DataFrame:
    """
    Carrega os dois CSVs brutos e marca de qual arquivo cada linha veio.
    O projeto trata os dois arquivos como um unico problema de classificacao.
    """
    high_popularity = pd.read_csv(RAW_DATA_DIR / "high_popularity_spotify_data.csv")
    high_popularity["source_dataset"] = "high"

    low_popularity = pd.read_csv(RAW_DATA_DIR / "low_popularity_spotify_data.csv")
    low_popularity["source_dataset"] = "low"

    return pd.concat([high_popularity, low_popularity], ignore_index=True)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variaveis derivadas que ajudam o modelo.
    - duration_minutes: converte milissegundos para minutos
    - release_year: extrai o ano da data de lancamento
    """
    featured_df = df.copy()

    featured_df["duration_minutes"] = (
        pd.to_numeric(featured_df["duration_ms"], errors="coerce") / 60000.0
    )
    featured_df["release_year"] = pd.to_datetime(
        featured_df["track_album_release_date"],
        errors="coerce",
    ).dt.year

    featured_df = featured_df[featured_df["duration_minutes"].fillna(0) > 0].copy()
    return featured_df


def remove_duplicate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantem apenas uma linha por musica.
    A mesma faixa pode aparecer em varias playlists. Se ela ficar repetida,
    treino e teste deixam de ser uma comparacao confiavel.
    """
    ordered_df = df.sort_values(
        by=[POPULARITY_COLUMN, TRACK_ID_COLUMN],
        ascending=[False, True],
        na_position="last",
    )
    deduplicated_df = ordered_df.drop_duplicates(subset=[TRACK_ID_COLUMN], keep="first")
    return deduplicated_df.reset_index(drop=True)


def keep_only_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas metadados e variaveis usadas no projeto."""
    modeling_df = df.loc[:, DATASET_COLUMNS].copy()

    # Converte para numerico todas as colunas que deveriam ser numericas.
    # Se uma coluna aparecer duplicada, iloc[:, 0] garante que trabalhamos
    # sempre com uma Series antes de passar ao to_numeric.
    numeric_columns_to_convert = list(dict.fromkeys([POPULARITY_COLUMN, *NUMERIC_FEATURES]))
    for column in numeric_columns_to_convert:
        col_data = modeling_df[column]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        modeling_df[column] = pd.to_numeric(col_data, errors="coerce")

    return modeling_df


def preprocess_dataset() -> tuple[pd.DataFrame, dict[str, object]]:
    """Executa toda a pipeline de preparacao dos dados."""
    raw_df = load_raw_spotify_data()
    rows_in_raw_data = len(raw_df)

    # Remove linhas sem gênero definido (target obrigatório)
    filtered_df = raw_df[raw_df[TARGET_COLUMN].notna()].copy()

    featured_df = create_derived_features(filtered_df)

    rows_before_deduplication = len(featured_df)
    deduplicated_df = remove_duplicate_tracks(featured_df)
    modeling_df = keep_only_model_columns(deduplicated_df)

    pipeline_summary = {
        "raw_rows": int(rows_in_raw_data),
        "rows_after_genre_filter": int(len(filtered_df)),
        "duplicate_tracks_removed": int(rows_before_deduplication - len(modeling_df)),
        "processed_rows": int(len(modeling_df)),
        "class_distribution": modeling_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }

    return modeling_df, pipeline_summary


def save_processed_dataset_and_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Salva a base tratada completa e a divisao treino/teste.
    Guardar o split ajuda na reprodutibilidade e na comparacao com outros modelos.
    """
    processed_dataset_path = PROCESSED_DATA_DIR / "spotify_genre_dataset.csv"
    train_dataset_path = SPLIT_DATA_DIR / "train.csv"
    test_dataset_path = SPLIT_DATA_DIR / "test.csv"
    split_metadata_path = METRICS_DIR / "train_test_split_metadata.json"

    dataset.to_csv(processed_dataset_path, index=False)

    train_df, test_df = train_test_split(
        dataset,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset[TARGET_COLUMN],
    )

    train_df = train_df.sort_values(TRACK_ID_COLUMN).reset_index(drop=True)
    test_df = test_df.sort_values(TRACK_ID_COLUMN).reset_index(drop=True)

    train_df.to_csv(train_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)

    split_summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    split_metadata_path.write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return train_df, test_df, split_summary


def build_feature_preprocessor() -> ColumnTransformer:
    """
    Pre-processamento compativel com GaussianNB.

    GaussianNB exige que todas as colunas sejam numericas. Por isso:
    - Numericas: imputacao pela mediana (mantem a escala original).
    - Categoricas: imputacao pelo valor mais frequente + OneHotEncoder.
      As colunas binarias resultantes (0/1) sao tratadas como gaussianas,
      o que e uma aproximacao aceitavel para features de baixa cardinalidade.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def compute_class_priors(y_train: pd.Series) -> list[float]:
    """
    Calcula as probabilidades a priori a partir da distribuicao real do treino.
    Passar os priors explicitamente garante que o modelo use exatamente a
    proporcao observada no conjunto de treino, sem depender da ordem interna
    das classes no GaussianNB (que ordena alfabeticamente).
    A lista retornada respeita a ordem alfabetica das classes para compatibilidade
    com o sklearn.
    """
    total = len(y_train)
    counts = y_train.value_counts()
    # Ordem alfabetica para coincidir com model.classes_ do GaussianNB
    classes_sorted = sorted(counts.index)
    return [float(counts.get(cls, 0)) / total for cls in classes_sorted]


def tune_var_smoothing(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    priors: list[float],
) -> float:
    """
    Encontra o melhor var_smoothing via busca em grade com validacao cruzada.
    Usa acuracia como criterio de selecao, espelhando o objetivo principal
    de maximizar a proporcao de acertos do modelo.
    """
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("model", GaussianNB(priors=priors)),
        ]
    )

    param_grid = {"model__var_smoothing": VAR_SMOOTHING_GRID}

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        refit=False,  # nao re-treina aqui; o treino final e feito separadamente
    )
    grid_search.fit(x_train, y_train)

    best_var_smoothing = grid_search.best_params_["model__var_smoothing"]
    return float(best_var_smoothing)


def build_naive_bayes_pipeline(priors: list[float], var_smoothing: float) -> Pipeline:
    """
    Cria a pipeline completa: pre-processamento + GaussianNB com parametros otimizados.
    - priors: probabilidades a priori calculadas do conjunto de treino.
    - var_smoothing: melhor valor encontrado pelo GridSearchCV.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("model", GaussianNB(priors=priors, var_smoothing=var_smoothing)),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """
    Calcula as metricas principais para classificacao multiclasse.
    Usa average='weighted' para ponderar as metricas pelo suporte de cada classe.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_model_artifacts(
    test_df: pd.DataFrame,
    predictions: pd.Series,
    metrics_summary: dict[str, object],
) -> None:
    """Salva matriz de confusao, predicoes e metricas do Naive Bayes."""
    confusion_matrix_path = METRICS_DIR / "naive_bayes_confusion_matrix.csv"
    predictions_path = METRICS_DIR / "naive_bayes_test_predictions.csv"
    metrics_path = METRICS_DIR / "naive_bayes_metrics.json"

    genre_labels = sorted(test_df[TARGET_COLUMN].unique())
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(
            test_df[TARGET_COLUMN],
            predictions,
            labels=genre_labels,
        ),
        index=[f"real_{g}" for g in genre_labels],
        columns=[f"previsto_{g}" for g in genre_labels],
    )
    confusion_matrix_df.to_csv(confusion_matrix_path)

    predictions_df = test_df[[TRACK_ID_COLUMN, TRACK_NAME_COLUMN, TARGET_COLUMN]].copy()
    predictions_df["predicao"] = predictions
    predictions_df.to_csv(predictions_path, index=False)

    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Nota: GaussianNB nao possui feature_importances_, portanto nao ha
    # arquivo de importancia de variaveis equivalente ao do Random Forest.


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Treina o GaussianNB otimizado, avalia no conjunto de teste e salva os artefatos.
    Fluxo:
    1. Calcula os priors a partir da distribuicao real do treino (N classes).
    2. Busca o melhor var_smoothing com GridSearchCV (5-fold estratificado).
    3. Treina o modelo final com os melhores parametros.
    4. Avalia no conjunto de teste e salva artefatos.
    """
    x_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    # Passo 1: priors a partir da distribuicao real do treino (multiclasse)
    priors = compute_class_priors(y_train)

    # Passo 2: busca em grade para var_smoothing
    best_var_smoothing = tune_var_smoothing(x_train, y_train, priors)

    # Passo 3: treino final com parametros otimizados
    naive_bayes_params = {
        "priors": priors,
        "var_smoothing": best_var_smoothing,
    }
    model_pipeline = build_naive_bayes_pipeline(
        priors=priors,
        var_smoothing=best_var_smoothing,
    )
    model_pipeline.fit(x_train, y_train)

    # Passo 4: avaliacao no conjunto de teste
    predictions = model_pipeline.predict(x_test)

    metrics_summary = calculate_metrics(y_test, predictions)
    metrics_summary["features"] = MODEL_FEATURES
    metrics_summary["model_params"] = naive_bayes_params

    save_model_artifacts(
        test_df=test_df,
        predictions=predictions,
        metrics_summary=metrics_summary,
    )

    return metrics_summary


def main() -> None:
    """Executa o fluxo completo: tratamento, split, treino e avaliacao."""
    create_output_folders()

    processed_dataset, pipeline_summary = preprocess_dataset()
    train_df, test_df, split_summary = save_processed_dataset_and_split(processed_dataset)
    model_summary = train_and_evaluate_model(train_df, test_df)

    final_summary = {
        "pipeline": pipeline_summary,
        "split": split_summary,
        "naive_bayes": model_summary,
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()