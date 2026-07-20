"""
Testes para o pipeline de classificacao de genero musical.

Organizacao:
  - Fixtures compartilhadas
  - Grupo 1: Pre-processamento (create_derived_features, remove_duplicate_tracks,
             keep_only_model_columns, preprocess_dataset)
  - Grupo 2: Pipeline de treino (build_random_forest_pipeline, train_and_evaluate_model)
  - Grupo 3: Metricas (calculate_metrics)
  - Grupo 4: Integracao end-to-end (fluxo completo com dados sinteticos)

Como rodar:
  pip install pytest
  pytest test_main.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from main import (
    CATEGORICAL_FEATURES,
    DATASET_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    POPULARITY_COLUMN,
    TARGET_COLUMN,
    TRACK_ID_COLUMN,
    TRACK_NAME_COLUMN,
    build_random_forest_pipeline,
    calculate_metrics,
    create_derived_features,
    keep_only_model_columns,
    remove_duplicate_tracks,
    train_and_evaluate_model,
)


# ==============================================================================
# Helpers / Fixtures
# ==============================================================================

GENRES = ["pop", "rap", "rock", "latin", "r&b", "edm"]
SUBGENRES = {
    "pop": "dance pop",
    "rap": "hip hop",
    "rock": "classic rock",
    "latin": "latin pop",
    "r&b": "urban contemporary",
    "edm": "electro house",
}


def _make_raw_row(
    track_id: str = "id_001",
    track_name: str = "Song",
    genre: str = "pop",
    popularity: float = 70.0,
    duration_ms: float = 210000.0,
    release_date: str = "2020-06-15",
    source: str = "high",
    **overrides,
) -> dict:
    """Retorna um dicionario representando uma linha bruta do CSV."""
    row = {
        "track_id": track_id,
        "track_name": track_name,
        "playlist_genre": genre,
        "playlist_subgenre": SUBGENRES.get(genre, "unknown"),
        "track_popularity": popularity,
        "duration_ms": duration_ms,
        "track_album_release_date": release_date,
        "source_dataset": source,
        "danceability": 0.75,
        "energy": 0.80,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.10,
        "instrumentalness": 0.00,
        "liveness": 0.12,
        "valence": 0.65,
        "tempo": 120.0,
        "time_signature": 4,
    }
    row.update(overrides)
    return row


def _make_raw_df(n_per_genre: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Cria um DataFrame bruto sintetico com n_per_genre linhas por genero.
    Os valores numericos variam levemente para dar variancia ao modelo.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, genre in enumerate(GENRES):
        for j in range(n_per_genre):
            idx = i * n_per_genre + j
            rows.append(
                _make_raw_row(
                    track_id=f"id_{idx:04d}",
                    track_name=f"Track {idx}",
                    genre=genre,
                    popularity=float(rng.integers(30, 100)),
                    duration_ms=float(rng.integers(150_000, 300_000)),
                    release_date=f"{rng.integers(2000, 2024)}-01-01",
                    danceability=float(rng.uniform(0.1, 1.0)),
                    energy=float(rng.uniform(0.1, 1.0)),
                    key=int(rng.integers(0, 11)),
                    loudness=float(rng.uniform(-20, 0)),
                    mode=int(rng.integers(0, 1)),
                    speechiness=float(rng.uniform(0.02, 0.5)),
                    acousticness=float(rng.uniform(0.0, 1.0)),
                    instrumentalness=float(rng.uniform(0.0, 0.5)),
                    liveness=float(rng.uniform(0.05, 0.5)),
                    valence=float(rng.uniform(0.1, 1.0)),
                    tempo=float(rng.uniform(60, 180)),
                    time_signature=int(rng.choice([3, 4])),
                )
            )
    return pd.DataFrame(rows)


def _make_processed_df(n_per_genre: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Retorna um DataFrame ja com as features derivadas e colunas do modelo,
    simulando a saida de preprocess_dataset().
    """
    raw = _make_raw_df(n_per_genre=n_per_genre, seed=seed)
    df = raw.copy()
    df["duration_minutes"] = df["duration_ms"] / 60000.0
    df["release_year"] = pd.to_datetime(df["track_album_release_date"], errors="coerce").dt.year
    # Garante que apenas colunas de DATASET_COLUMNS existam
    for col in DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DATASET_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ==============================================================================
# Grupo 1 — Pre-processamento
# ==============================================================================

class TestCreateDerivedFeatures:

    def test_duration_minutes_calculada_corretamente(self):
        df = pd.DataFrame([_make_raw_row(duration_ms=240_000.0)])
        result = create_derived_features(df)
        assert pytest.approx(result["duration_minutes"].iloc[0], abs=1e-6) == 4.0

    def test_release_year_extraido_corretamente(self):
        df = pd.DataFrame([_make_raw_row(release_date="2019-03-22")])
        result = create_derived_features(df)
        assert result["release_year"].iloc[0] == 2019

    def test_release_year_apenas_ano(self):
        """Datas no formato 'YYYY' (sem mes/dia) devem ser aceitas."""
        df = pd.DataFrame([_make_raw_row(release_date="2015")])
        result = create_derived_features(df)
        assert result["release_year"].iloc[0] == 2015

    def test_linhas_com_duration_zero_sao_removidas(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_ok", duration_ms=200_000.0),
            _make_raw_row(track_id="id_zero", duration_ms=0.0),
        ])
        result = create_derived_features(df)
        assert "id_zero" not in result["track_id"].values

    def test_linhas_com_duration_nula_sao_removidas(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_ok", duration_ms=200_000.0),
            _make_raw_row(track_id="id_null", duration_ms=None),
        ])
        result = create_derived_features(df)
        assert "id_null" not in result["track_id"].values

    def test_data_invalida_vira_nan_no_release_year(self):
        df = pd.DataFrame([_make_raw_row(release_date="nao-e-data")])
        result = create_derived_features(df)
        assert pd.isna(result["release_year"].iloc[0])

    def test_duration_ms_nao_numerica_vira_nan(self):
        df = pd.DataFrame([_make_raw_row(duration_ms="abc")])
        result = create_derived_features(df)
        # Linha deve ser removida pois duration_minutes sera NaN (tratado como 0)
        assert len(result) == 0


class TestRemoveDuplicateTracks:

    def test_duplicata_removida_mantem_maior_popularidade(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_001", popularity=90.0, genre="pop"),
            _make_raw_row(track_id="id_001", popularity=40.0, genre="rock"),
        ])
        result = remove_duplicate_tracks(df)
        assert len(result) == 1
        assert result.iloc[0]["track_popularity"] == 90.0

    def test_faixas_distintas_sao_mantidas(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_001"),
            _make_raw_row(track_id="id_002"),
            _make_raw_row(track_id="id_003"),
        ])
        result = remove_duplicate_tracks(df)
        assert len(result) == 3

    def test_multiplas_duplicatas_mesmo_id(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_001", popularity=50.0),
            _make_raw_row(track_id="id_001", popularity=70.0),
            _make_raw_row(track_id="id_001", popularity=60.0),
        ])
        result = remove_duplicate_tracks(df)
        assert len(result) == 1
        assert result.iloc[0]["track_popularity"] == 70.0

    def test_resultado_tem_index_resetado(self):
        df = pd.DataFrame([
            _make_raw_row(track_id="id_001"),
            _make_raw_row(track_id="id_001"),
        ])
        result = remove_duplicate_tracks(df)
        assert list(result.index) == list(range(len(result)))


class TestKeepOnlyModelColumns:

    def test_colunas_esperadas_presentes(self):
        raw = _make_raw_df(n_per_genre=2)
        raw["duration_minutes"] = raw["duration_ms"] / 60000.0
        raw["release_year"] = 2020
        result = keep_only_model_columns(raw)
        for col in DATASET_COLUMNS:
            assert col in result.columns, f"Coluna ausente: {col}"

    def test_colunas_numericas_convertidas(self):
        raw = _make_raw_df(n_per_genre=2)
        raw["duration_minutes"] = raw["duration_ms"] / 60000.0
        raw["release_year"] = 2020
        # Injeta strings nos numericos para forcar conversao
        raw["tempo"] = raw["tempo"].astype(str)
        result = keep_only_model_columns(raw)
        assert pd.api.types.is_float_dtype(result["tempo"]) or pd.api.types.is_numeric_dtype(result["tempo"])

    def test_sem_colunas_duplicadas(self):
        raw = _make_raw_df(n_per_genre=2)
        raw["duration_minutes"] = raw["duration_ms"] / 60000.0
        raw["release_year"] = 2020
        result = keep_only_model_columns(raw)
        assert result.columns.duplicated().sum() == 0

    def test_nao_retorna_colunas_extras(self):
        raw = _make_raw_df(n_per_genre=2)
        raw["duration_minutes"] = raw["duration_ms"] / 60000.0
        raw["release_year"] = 2020
        raw["coluna_extra_inutil"] = 99
        result = keep_only_model_columns(raw)
        assert "coluna_extra_inutil" not in result.columns


class TestPreprocessDataset:

    def _make_csv_pair(self, tmp_path: Path):
        """Gera os dois CSVs brutos esperados por load_raw_spotify_data."""
        raw = _make_raw_df(n_per_genre=15)
        half = len(raw) // 2
        high = raw.iloc[:half].drop(columns=["source_dataset"])
        low = raw.iloc[half:].drop(columns=["source_dataset"])
        cru = tmp_path / "data" / "cru"
        cru.mkdir(parents=True)
        high.to_csv(cru / "high_popularity_spotify_data.csv", index=False)
        low.to_csv(cru / "low_popularity_spotify_data.csv", index=False)

    def test_dataset_processado_nao_tem_genero_nulo(self, tmp_path):
        self._make_csv_pair(tmp_path)
        with patch("main.BASE_DIR", tmp_path):
            import main as m
            import importlib
            importlib.reload(m)
            result, _ = m.preprocess_dataset()
        assert result[TARGET_COLUMN].notna().all()

    def test_summary_contem_chaves_esperadas(self, tmp_path):
        self._make_csv_pair(tmp_path)
        with patch("main.BASE_DIR", tmp_path):
            import main as m
            import importlib
            importlib.reload(m)
            _, summary = m.preprocess_dataset()
        for key in ("raw_rows", "rows_after_genre_filter", "processed_rows", "class_distribution"):
            assert key in summary, f"Chave ausente no summary: {key}"

    def test_todos_generos_presentes_na_distribuicao(self, tmp_path):
        self._make_csv_pair(tmp_path)
        with patch("main.BASE_DIR", tmp_path):
            import main as m
            import importlib
            importlib.reload(m)
            result, summary = m.preprocess_dataset()
        genres_in_data = set(result[TARGET_COLUMN].unique())
        genres_in_summary = set(summary["class_distribution"].keys())
        assert genres_in_data == genres_in_summary


# ==============================================================================
# Grupo 2 — Pipeline de Treino
# ==============================================================================

class TestBuildRandomForestPipeline:

    def test_pipeline_tem_dois_steps(self):
        pipeline = build_random_forest_pipeline()
        assert len(pipeline.steps) == 2

    def test_pipeline_steps_corretos(self):
        pipeline = build_random_forest_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["preprocessor", "model"]

    def test_pipeline_e_instancia_de_pipeline(self):
        assert isinstance(build_random_forest_pipeline(), Pipeline)

    def test_pipeline_treina_sem_erros(self):
        pipeline = build_random_forest_pipeline()
        df = _make_processed_df(n_per_genre=10)
        pipeline.fit(df[MODEL_FEATURES], df[TARGET_COLUMN])

    def test_pipeline_prediz_generos_validos(self):
        pipeline = build_random_forest_pipeline()
        df = _make_processed_df(n_per_genre=12)
        pipeline.fit(df[MODEL_FEATURES], df[TARGET_COLUMN])
        preds = pipeline.predict(df[MODEL_FEATURES])
        assert set(preds).issubset(set(GENRES))

    def test_pipeline_aceita_features_com_nulos(self):
        """O SimpleImputer deve tratar NaN sem explodir."""
        pipeline = build_random_forest_pipeline()
        df = _make_processed_df(n_per_genre=12)
        df.loc[df.index[:5], "danceability"] = np.nan
        df.loc[df.index[5:10], "acousticness"] = np.nan
        pipeline.fit(df[MODEL_FEATURES], df[TARGET_COLUMN])
        pipeline.predict(df[MODEL_FEATURES])  # nao deve levantar excecao

    def test_pipeline_aceita_subgenero_desconhecido_no_predict(self):
        """OneHotEncoder com handle_unknown='ignore' nao deve falhar."""
        pipeline = build_random_forest_pipeline()
        train_df = _make_processed_df(n_per_genre=12)
        pipeline.fit(train_df[MODEL_FEATURES], train_df[TARGET_COLUMN])
        test_row = _make_processed_df(n_per_genre=2).iloc[:1].copy()
        test_row["playlist_subgenre"] = "genero_nunca_visto_xpto"
        pipeline.predict(test_row[MODEL_FEATURES])  # nao deve levantar excecao


class TestTrainAndEvaluateModel:

    def test_retorna_metricas_com_chaves_corretas(self, tmp_path):
        _create_artifact_dirs(tmp_path)
        train_df = _make_processed_df(n_per_genre=20)
        test_df = _make_processed_df(n_per_genre=5, seed=99)
        with _patch_artifact_dirs(tmp_path):
            result = train_and_evaluate_model(train_df, test_df)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert key in result

    def test_accuracy_entre_0_e_1(self, tmp_path):
        _create_artifact_dirs(tmp_path)
        train_df = _make_processed_df(n_per_genre=20)
        test_df = _make_processed_df(n_per_genre=5, seed=99)
        with _patch_artifact_dirs(tmp_path):
            result = train_and_evaluate_model(train_df, test_df)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_artefatos_salvos_em_disco(self, tmp_path):
        _create_artifact_dirs(tmp_path)
        train_df = _make_processed_df(n_per_genre=20)
        test_df = _make_processed_df(n_per_genre=5, seed=99)
        with _patch_artifact_dirs(tmp_path):
            train_and_evaluate_model(train_df, test_df)
        metrics_dir = tmp_path / "artifacts" / "metrics"
        models_dir = tmp_path / "artifacts" / "models"
        assert (metrics_dir / "random_forest_metrics.json").exists()
        assert (metrics_dir / "random_forest_confusion_matrix.csv").exists()
        assert (metrics_dir / "random_forest_feature_importance.csv").exists()
        assert (metrics_dir / "random_forest_test_predictions.csv").exists()
        assert (models_dir / "random_forest_model.joblib").exists()

    def test_modelo_treinado_em_dados_claros_tem_accuracy_acima_de_chance(self, tmp_path):
        """
        Com dados sinteticos que tem variancia real, o modelo deve superar
        o baseline de acerto aleatorio (1/num_generos = ~17% para 6 classes).
        """
        _create_artifact_dirs(tmp_path)
        train_df = _make_processed_df(n_per_genre=30)
        test_df = _make_processed_df(n_per_genre=8, seed=77)
        with _patch_artifact_dirs(tmp_path):
            result = train_and_evaluate_model(train_df, test_df)
        chance_baseline = 1.0 / len(GENRES)
        assert result["accuracy"] > chance_baseline


# ==============================================================================
# Grupo 3 — Metricas
# ==============================================================================

class TestCalculateMetrics:

    def test_predicao_perfeita_retorna_1_em_todas_metricas(self):
        y = pd.Series(["pop", "rap", "rock", "pop", "rock"])
        metrics = calculate_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1_score"] == pytest.approx(1.0)

    def test_predicao_errada_retorna_0_em_accuracy(self):
        y_true = pd.Series(["pop", "pop", "pop"])
        y_pred = pd.Series(["rap", "rock", "edm"])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_metricas_entre_0_e_1(self):
        y_true = pd.Series(["pop", "rap", "rock", "pop", "rock", "rap"])
        y_pred = pd.Series(["pop", "pop",  "rock", "rap", "rock", "rap"])
        metrics = calculate_metrics(y_true, y_pred)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} fora do intervalo [0,1]: {value}"

    def test_chaves_retornadas_corretas(self):
        y = pd.Series(["pop", "rap"])
        metrics = calculate_metrics(y, y)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}

    def test_multiclasse_com_todas_classes_corretas(self):
        y = pd.Series(GENRES)
        metrics = calculate_metrics(y, y)
        assert metrics["f1_score"] == pytest.approx(1.0)

    def test_zero_division_nao_explode(self):
        """Classe sem predicoes nao deve levantar ZeroDivisionError."""
        y_true = pd.Series(["pop", "pop", "pop"])
        y_pred = pd.Series(["rap", "rap", "rap"])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["precision"] == pytest.approx(0.0)


# ==============================================================================
# Grupo 4 — Integracao end-to-end
# ==============================================================================

class TestEndToEnd:

    def test_fluxo_completo_sem_erros(self, tmp_path):
        """
        Simula o fluxo completo: preprocessamento -> split -> treino -> metricas.
        Usa dados sinteticos sem tocar no disco real.
        """
        _create_all_dirs(tmp_path)
        raw = _make_raw_df(n_per_genre=20)
        half = len(raw) // 2
        cru = tmp_path / "data" / "cru"
        raw.iloc[:half].drop(columns=["source_dataset"]).to_csv(
            cru / "high_popularity_spotify_data.csv", index=False
        )
        raw.iloc[half:].drop(columns=["source_dataset"]).to_csv(
            cru / "low_popularity_spotify_data.csv", index=False
        )

        with _patch_artifact_dirs(tmp_path):
            import main as m
            import importlib
            importlib.reload(m)

            m.create_output_folders()
            processed, pipeline_summary = m.preprocess_dataset()
            train_df, test_df, split_summary = m.save_processed_dataset_and_split(processed)
            model_summary = m.train_and_evaluate_model(train_df, test_df)

        assert "accuracy" in model_summary
        assert 0.0 <= model_summary["accuracy"] <= 1.0
        assert split_summary["train_rows"] + split_summary["test_rows"] == pipeline_summary["processed_rows"]

    def test_split_estratificado_mantem_proporcao(self, tmp_path):
        """Cada genero deve aparecer em treino e teste."""
        _create_all_dirs(tmp_path)
        raw = _make_raw_df(n_per_genre=20)
        half = len(raw) // 2
        cru = tmp_path / "data" / "cru"
        raw.iloc[:half].drop(columns=["source_dataset"]).to_csv(
            cru / "high_popularity_spotify_data.csv", index=False
        )
        raw.iloc[half:].drop(columns=["source_dataset"]).to_csv(
            cru / "low_popularity_spotify_data.csv", index=False
        )

        with _patch_artifact_dirs(tmp_path):
            import main as m
            import importlib
            importlib.reload(m)
            m.create_output_folders()
            processed, _ = m.preprocess_dataset()
            train_df, test_df, _ = m.save_processed_dataset_and_split(processed)

        genres_in_train = set(train_df[TARGET_COLUMN].unique())
        genres_in_test = set(test_df[TARGET_COLUMN].unique())
        assert genres_in_train == genres_in_test

    def test_predicoes_salvas_contem_todas_colunas_esperadas(self, tmp_path):
        _create_all_dirs(tmp_path)
        raw = _make_raw_df(n_per_genre=20)
        half = len(raw) // 2
        cru = tmp_path / "data" / "cru"
        raw.iloc[:half].drop(columns=["source_dataset"]).to_csv(
            cru / "high_popularity_spotify_data.csv", index=False
        )
        raw.iloc[half:].drop(columns=["source_dataset"]).to_csv(
            cru / "low_popularity_spotify_data.csv", index=False
        )

        with _patch_artifact_dirs(tmp_path):
            import main as m
            import importlib
            importlib.reload(m)
            m.create_output_folders()
            processed, _ = m.preprocess_dataset()
            train_df, test_df, _ = m.save_processed_dataset_and_split(processed)
            m.train_and_evaluate_model(train_df, test_df)

        preds_path = tmp_path / "artifacts" / "metrics" / "random_forest_test_predictions.csv"
        preds_df = pd.read_csv(preds_path)
        for col in (TRACK_ID_COLUMN, TRACK_NAME_COLUMN, TARGET_COLUMN, "predicao"):
            assert col in preds_df.columns, f"Coluna ausente nas predicoes: {col}"


# ==============================================================================
# Utilitarios internos dos testes
# ==============================================================================

def _create_artifact_dirs(tmp_path: Path) -> None:
    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "models").mkdir(parents=True, exist_ok=True)


def _create_all_dirs(tmp_path: Path) -> None:
    _create_artifact_dirs(tmp_path)
    (tmp_path / "data" / "cru").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "splits").mkdir(parents=True, exist_ok=True)


def _patch_artifact_dirs(tmp_path: Path):
    """Redireciona todos os caminhos de saida para tmp_path."""
    return patch.multiple(
        "main",
        BASE_DIR=tmp_path,
        RAW_DATA_DIR=tmp_path / "data" / "cru",
        PROCESSED_DATA_DIR=tmp_path / "data" / "processed",
        SPLIT_DATA_DIR=tmp_path / "data" / "splits",
        METRICS_DIR=tmp_path / "artifacts" / "metrics",
        MODELS_DIR=tmp_path / "artifacts" / "models",
    )
