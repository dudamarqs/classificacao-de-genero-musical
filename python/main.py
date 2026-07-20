import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# Configuracao do projeto
# =========================

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "cru"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SPLIT_DATA_DIR = BASE_DIR / "data" / "splits"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
MODELS_DIR = BASE_DIR / "artifacts" / "models"

TRACK_ID_COLUMN = "track_id"
TRACK_NAME_COLUMN = "track_name"
POPULARITY_COLUMN = "track_popularity"

# Agora o alvo é o gênero da música (classificação multiclasse)
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

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


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
    # O squeeze() garante que, mesmo que haja colunas de mesmo nome, sempre
    # trabalhamos com uma Series antes de passar ao to_numeric.
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
    Cria o passo de pre-processamento usado antes do treino.

    - Colunas numericas: preenche faltantes com a mediana
    - Colunas categoricas: preenche faltantes com o valor mais comum e
      depois transforma texto em colunas binarias
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Cria a pipeline completa de treino."""
    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("model", RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
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
    fitted_pipeline: Pipeline,
    test_df: pd.DataFrame,
    predictions: pd.Series,
    metrics_summary: dict[str, object],
) -> None:
    """Salva todos os arquivos gerados pelo modelo treinado."""
    confusion_matrix_path = METRICS_DIR / "random_forest_confusion_matrix.csv"
    feature_importance_path = METRICS_DIR / "random_forest_feature_importance.csv"
    predictions_path = METRICS_DIR / "random_forest_test_predictions.csv"
    metrics_path = METRICS_DIR / "random_forest_metrics.json"
    model_path = MODELS_DIR / "random_forest_model.joblib"

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

    transformed_feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": fitted_pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    predictions_df = test_df[[TRACK_ID_COLUMN, TRACK_NAME_COLUMN, TARGET_COLUMN]].copy()
    predictions_df["predicao"] = predictions
    predictions_df.to_csv(predictions_path, index=False)

    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    joblib.dump(fitted_pipeline, model_path)


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    """Treina o modelo, avalia o resultado e salva os artefatos."""
    model_pipeline = build_random_forest_pipeline()

    x_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    metrics_summary = calculate_metrics(y_test, predictions)
    metrics_summary["features"] = MODEL_FEATURES
    metrics_summary["model_params"] = RANDOM_FOREST_PARAMS

    save_model_artifacts(
        fitted_pipeline=model_pipeline,
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
        "random_forest": model_summary,
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
