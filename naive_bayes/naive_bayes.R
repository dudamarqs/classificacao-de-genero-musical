library(tidyverse)
library(naivebayes)
library(caret)
library(lubridate)

set.seed(42)

# Carregamento dos dados brutos
high_pop <- read_csv("data/cru/high_popularity_spotify_data.csv", show_col_types = FALSE)
low_pop  <- read_csv("data/cru/low_popularity_spotify_data.csv", show_col_types = FALSE)

# União e limpeza (Lógica da Pipeline)
df_final <- bind_rows(high_pop, low_pop) %>%
  filter(!is.na(playlist_genre)) %>%
  mutate(
    target = as.factor(playlist_genre),
    playlist_subgenre = as.factor(playlist_subgenre),
    duration_minutes = as.numeric(duration_ms) / 60000,
    release_year = year(ymd(track_album_release_date, quiet = TRUE))
  ) %>%
  filter(duration_minutes > 0) %>%
  distinct(track_id, .keep_all = TRUE) %>%
  select(speechiness, acousticness, valence, danceability, energy, loudness,
         tempo, track_popularity, duration_minutes, release_year, 
         playlist_subgenre, target) %>%
  drop_na()

# Divisão Treino/Teste (80/20)
index <- createDataPartition(df_final$target, p = 0.8, list = FALSE)
train_df <- df_final[index, ]
test_df  <- df_final[-index, ]

modelo_nb <- naive_bayes(
  target ~ ., 
  data = train_df, 
  laplace = 1, 
  usekernel = TRUE
)

predicoes <- predict(modelo_nb, test_df)
resultado <- confusionMatrix(predicoes, test_df$target)

print("=== RESULTADOS NAIVE BAYES ===")
print(resultado$overall['Accuracy'])
print(as.data.frame(resultado$byClass)[, c("Precision", "Recall", "F1")])

# 1. Acurácia Geral
acuracia <- resultado$overall['Accuracy']

# 2. Médias de Precisão, Recall e F1
metricas_por_classe <- as.data.frame(resultado$byClass)

precisao_media <- mean(metricas_por_classe$`Pos Pred Value`, na.rm = TRUE)
recall_medio   <- mean(metricas_por_classe$Sensitivity, na.rm = TRUE)
f1_medio       <- mean(metricas_por_classe$F1, na.rm = TRUE)

cat("\n--- VALORES PARA A TABELA FINAL ---\n")
cat("Acurácia:", acuracia, "\n")
cat("Precisão Média:", precisao_media, "\n")
cat("Recall Médio:", recall_medio, "\n")
cat("F1-Score Médio:", f1_medio, "\n")