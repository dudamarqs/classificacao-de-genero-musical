# =============================================================================
# Pipeline de tratamento de dados — Spotify Genre Classification
# Equivalente R de preprocess_dataset() + save_processed_dataset_and_split()
# =============================================================================
# install.packages(c("dplyr", "jsonlite", "lubridate"))
# install.packages("randomForest")
# install.packages("caret")
library(lubridate)
library(dplyr)
library(jsonlite)
library(randomForest)
library(caret)

# =============================================================================
# Configuracao do projeto
# =============================================================================

BASE_DIR            <- dirname(rstudioapi::getSourceEditorContext()$path)
RAW_DATA_DIR        <- file.path(BASE_DIR, "data", "cru")
PROCESSED_DATA_DIR  <- file.path(BASE_DIR, "data", "processed")
SPLIT_DATA_DIR      <- file.path(BASE_DIR, "data", "splits")
METRICS_DIR         <- file.path(BASE_DIR, "artifacts", "metrics")

TRACK_ID_COLUMN     <- "track_id"
TRACK_NAME_COLUMN   <- "track_name"
POPULARITY_COLUMN   <- "track_popularity"
TARGET_COLUMN       <- "playlist_genre"

RANDOM_STATE        <- 42
TEST_SIZE           <- 0.20

NUMERIC_FEATURES <- c(
  "danceability", "energy", "key", "loudness", "mode",
  "speechiness", "acousticness", "instrumentalness", "liveness",
  "valence", "tempo", "time_signature", "duration_minutes",
  "release_year", "track_popularity"
)

CATEGORICAL_FEATURES <- c("playlist_subgenre")

MODEL_FEATURES <- c(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

DATASET_COLUMNS <- unique(c(
  TRACK_ID_COLUMN,
  TRACK_NAME_COLUMN,
  "source_dataset",
  POPULARITY_COLUMN,
  TARGET_COLUMN,
  MODEL_FEATURES
))


# =============================================================================
# Criacao das pastas de saida
# =============================================================================

create_output_folders <- function() {
  dirs <- c(PROCESSED_DATA_DIR, SPLIT_DATA_DIR, METRICS_DIR)
  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))
}


# =============================================================================
# Carregamento dos dados brutos
# =============================================================================

load_raw_spotify_data <- function() {
  high <- high_popularity_spotify_data %>%
    mutate(source_dataset = "high")
  
  low <- low_popularity_spotify_data %>%
    mutate(source_dataset = "low")
  
  bind_rows(high, low)
}


# =============================================================================
# Criacao das features derivadas
# =============================================================================

create_derived_features <- function(df) {
  df |>
    mutate(
      duration_minutes = suppressWarnings(as.numeric(duration_ms)) / 60000,
      release_year     = year(ymd(track_album_release_date, quiet = TRUE))
    ) |>
    filter(!is.na(duration_minutes) & duration_minutes > 0)
}


# =============================================================================
# Remocao de faixas duplicadas
# Mantem a linha de maior popularidade por track_id.
# Em caso de empate na popularidade, prioriza o menor track_id (ordem alfabetica).
# =============================================================================

remove_duplicate_tracks <- function(df) {
  df |>
    arrange(desc(.data[[POPULARITY_COLUMN]]), .data[[TRACK_ID_COLUMN]]) |>
    distinct(.data[[TRACK_ID_COLUMN]], .keep_all = TRUE) |>
    arrange(.data[[TRACK_ID_COLUMN]])
}


# =============================================================================
# Selecao e coercao das colunas do modelo
# =============================================================================

keep_only_model_columns <- function(df) {
  # Seleciona apenas as colunas necessarias (sem duplicatas, preservando ordem)
  cols_presentes <- intersect(DATASET_COLUMNS, colnames(df))
  df <- df |> select(all_of(cols_presentes))
  
  # Converte colunas numericas (coerce silenciosamente valores invalidos para NA)
  numeric_to_convert <- unique(c(POPULARITY_COLUMN, NUMERIC_FEATURES))
  df <- df |>
    mutate(across(
      all_of(intersect(numeric_to_convert, colnames(df))),
      ~ suppressWarnings(as.numeric(.x))
    ))
  
  df
}


# =============================================================================
# Pipeline completa de pre-processamento
# =============================================================================

preprocess_dataset <- function() {
  raw_df <- load_raw_spotify_data()
  rows_in_raw_data <- nrow(raw_df)
  
  # Remove linhas sem genero definido (target obrigatorio)
  filtered_df <- raw_df |> filter(!is.na(.data[[TARGET_COLUMN]]))
  rows_after_genre_filter <- nrow(filtered_df)
  
  featured_df <- create_derived_features(filtered_df)
  rows_before_dedup <- nrow(featured_df)
  
  dedup_df    <- remove_duplicate_tracks(featured_df)
  modeling_df <- keep_only_model_columns(dedup_df)
  
  class_dist <- modeling_df |>
    count(.data[[TARGET_COLUMN]], name = "n") |>
    arrange(.data[[TARGET_COLUMN]]) |>
    tibble::deframe()
  
  pipeline_summary <- list(
    raw_rows               = rows_in_raw_data,
    rows_after_genre_filter = rows_after_genre_filter,
    duplicate_tracks_removed = rows_before_dedup - nrow(modeling_df),
    processed_rows         = nrow(modeling_df),
    class_distribution     = class_dist
  )
  
  list(data = modeling_df, summary = pipeline_summary)
}


# =============================================================================
# Divisao treino / teste e salvamento
# =============================================================================

save_processed_dataset_and_split <- function(dataset) {
  write_csv(dataset, file.path(PROCESSED_DATA_DIR, "spotify_genre_dataset.csv"))
  
  set.seed(RANDOM_STATE)
  
  # Split estratificado: amostra proporcional por genero
  train_indices <- dataset |>
    mutate(.row = row_number()) |>
    group_by(.data[[TARGET_COLUMN]]) |>
    slice_sample(prop = 1 - TEST_SIZE) |>
    pull(.row)
  
  train_df <- dataset[train_indices, ]  |> arrange(.data[[TRACK_ID_COLUMN]])
  test_df  <- dataset[-train_indices, ] |> arrange(.data[[TRACK_ID_COLUMN]])
  
  write_csv(train_df, file.path(SPLIT_DATA_DIR, "train.csv"))
  write_csv(test_df,  file.path(SPLIT_DATA_DIR, "test.csv"))
  
  split_summary <- list(
    random_state = RANDOM_STATE,
    test_size    = TEST_SIZE,
    train_rows   = nrow(train_df),
    test_rows    = nrow(test_df)
  )
  
  split_metadata_path <- file.path(METRICS_DIR, "train_test_split_metadata.json")
  jsonlite::write_json(split_summary, split_metadata_path, pretty = TRUE, auto_unbox = TRUE)
  
  list(train = train_df, test = test_df, summary = split_summary)
}


# =============================================================================
# Ponto de entrada
# =============================================================================

main <- function() {
  create_output_folders()
  
  result       <- preprocess_dataset()
  split_result <- save_processed_dataset_and_split(result$data)
  
  final_summary <- list(
    pipeline = result$summary,
    split    = split_result$summary
  )
  
  cat(jsonlite::toJSON(final_summary, pretty = TRUE, auto_unbox = TRUE), "\n")
  
  invisible(list(
    train = split_result$train,
    test  = split_result$test
  ))
}

main()

resultado <- main()

train <- resultado$train
test  <- resultado$test

treinar_random_forest <- function(train, test) {
  
  # converte target para fator
  train$playlist_genre <- as.factor(train$playlist_genre)
  test$playlist_genre  <- as.factor(test$playlist_genre)
  
  # remove colunas que não ajudam
  colunas_remover <- c("track_id", "track_name")
  
  train_modelo <- train %>% select(-any_of(colunas_remover))
  test_modelo  <- test  %>% select(-any_of(colunas_remover))
  
  # emove valores ausentes
  train_modelo <- na.omit(train_modelo)
  test_modelo  <- na.omit(test_modelo)
  
  # treinamento do modelo
  set.seed(42)
  modelo <- randomForest(
    playlist_genre ~ .,
    data = train_modelo,
    ntree = 200,
    importance = TRUE
  )
  
  # previsões
  predicoes <- predict(modelo, test_modelo)
  
  # Avaliação
  matriz <- confusionMatrix(predicoes, test_modelo$playlist_genre)
  
  print(matriz)
  
  metricas <- matriz$byClass
  
  precision_media <- mean(metricas[, "Precision"], na.rm = TRUE)
  recall_media    <- mean(metricas[, "Recall"], na.rm = TRUE)
  f1_media        <- mean(metricas[, "F1"], na.rm = TRUE)
  
  cat("Acurácia:", matriz$overall["Accuracy"], "\n")
  cat("Precisão média:", precision_media, "\n")
  cat("Recall médio:", recall_media, "\n")
  cat("F1-score médio:", f1_media, "\n")
  
  return(list(
    modelo = modelo,
    predicoes = predicoes,
    metricas = matriz
  ))
}

resultado_modelo <- treinar_random_forest(train, test)
