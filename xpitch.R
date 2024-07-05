library(abdwr3edata)
library(baseballr)
library(dplyr)
library(fs)
library(lubridate)
library(caret)
library(xgboost)
library(gt)
library(gtExtras)
library(mlbplotR)

dir.create("dir2021")
dir.create("dir2022")
dir.create("dir2023")
dir.create("dir2024")

statcast_season(2021, "dir2021")
pitches_2021 <- statcast_read_csv(dir = "./dir2021")

statcast_season(2022, "dir2022")
pitches_2022 <- statcast_read_csv(dir = "./dir2022")

statcast_season(2023, "dir2023")
pitches_2023 <- statcast_read_csv(dir = "./dir2023")

statcast_season(2024, "dir2024")
pitches_2024 <- statcast_read_csv(dir = "./dir2024")

# until july 4th

pitches_2021 <- pitches_2021 %>% mutate(season = 2021)
pitches_2022 <- pitches_2022 %>% mutate(season = 2022)
pitches_2023 <- pitches_2023 %>% mutate(season = 2023)
pitches_2024 <- pitches_2024 %>% mutate(season = 2024)

data <- rbind(pitches_2021, pitches_2022, pitches_2023, pitches_2024)

fastballs <- c("SI", "FF", "FC", "FA")
breaking <- c("SL", "ST", "CU", "KC", "SV", "CS")
offspeed <- c("CH", "FS", "EP", "SC", "KN", "FO")

data <- data %>%
  filter(!(pitch_type %in% c(NA, "PO")), !is.na(delta_run_exp)) %>%
  mutate(pitch_type = ifelse(pitch_type %in% fastballs, "Fastball", ifelse(pitch_type %in% breaking, "Breaking", "Offspeed"))) %>%
  group_by(game_pk, at_bat_number) %>%
  arrange(game_pk, at_bat_number, pitch_number) %>%
  mutate(prev_pitch_type = lag(pitch_type), count = as.factor(paste0(balls, "-", strikes)), inning_half = as.factor(paste0(inning, " ", inning_topbot)), stand = as.factor(stand), p_throws = as.factor(p_throws), score_diff = fld_score - bat_score, pitch_number = as.factor(pitch_number), month = as.factor(month(game_date)), outs_when_up = as.factor(outs_when_up)) %>%
  ungroup() %>%
  group_by(season, batter) %>%
  mutate(hit_quality = cumsum(delta_run_exp) - delta_run_exp) %>%
  ungroup() %>%
  select(pitcher, season, pitch_type, prev_pitch_type, count, inning_half, stand, p_throws, outs_when_up, score_diff, pitch_number, month, hit_quality)

data$prev_pitch_type[which(is.na(data$prev_pitch_type))] <- "FIRST"

dummy <- dummyVars(" ~ . -pitch_type", data = data)
final_data <- data.frame(predict(dummy, newdata = data))

final_data$pitch_type <- data$pitch_type

xgboost_train <- final_data %>%
  filter(season != 2024) 

xgboost_test <- final_data %>%
  filter(season == 2024) 

labels_train <- as.matrix(xgboost_train[, 84]) 
xgboost_trainfinal <- as.matrix(xgboost_train[, c(3:83)]) 
xgboost_testfinal <- as.matrix(xgboost_test[, c(3:83)])

labels_train <- as.numeric(factor(labels_train, levels = c("Fastball", "Breaking", "Offspeed"))) - 1

pitch_pred_model <- xgboost(data = xgboost_trainfinal, label = labels_train, nrounds = 100, objective = "multi:softprob", num_class = length(unique(final_data$pitch_type)), early_stopping_rounds = 10, max_depth = 6, eta = 0.3) 

pitch_predictions <- predict(pitch_pred_model, xgboost_testfinal, type = "prob")

pitch_pred_df <- data.frame(
  pred_fastball = pitch_predictions[seq(1, length(pitch_predictions), by = 3)],
  pred_breaking = pitch_predictions[seq(2, length(pitch_predictions), by = 3)],
  pred_offspeed = pitch_predictions[seq(3, length(pitch_predictions), by = 3)]
)

test_data <- xgboost_test %>% select(pitcher, pitch_type) %>% cbind(pitch_pred_df) %>% group_by(pitcher) %>% summarize(actual_fastball = sum(pitch_type == "Fastball")/n() * 100, actual_breaking = sum(pitch_type == "Breaking")/n() * 100, actual_offspeed = sum(pitch_type == "Offspeed")/n() * 100, pred_fastball = mean(pred_fastball) * 100, pred_breaking = mean(pred_breaking) * 100, pred_offspeed = mean(pred_offspeed) * 100, dfep = abs(actual_fastball - pred_fastball) + abs(actual_breaking - pred_breaking) + abs(actual_offspeed - pred_offspeed))

qual_pitchers <- mlb_stats_leaders(leader_categories = 'era', sport_id = 1, season = 2024) %>% select(pitcher = person_id, name = person_full_name, team = team_name) %>% filter(row_number() <= n() - 30)
# for some reason catchers are included? but just ignore that

test_data <- inner_join(test_data, qual_pitchers, by = "pitcher")

teams <- mlb_teams() %>% select(team = team_full_name, abbr = team_abbreviation)

test_data <- test_data %>%
  left_join(teams, by = "team") %>%
  select(-team) %>%
  arrange(-dfep) %>%
  mutate(across(2:8, ~ round(.x, 1))) %>%
  select(pitcher, name, abbr, everything())

headshots <- load_headshots() %>% select(pitcher = savant_id, headshot = espn_headshot)

test_data <- left_join(test_data, headshots, by = "pitcher")
test_data$headshot[which(test_data$pitcher == 663362)] <- "https://a.espncdn.com/combiner/i?img=/i/headshots/mlb/players/full/4020158.png&w=350&h=254"
test_data$headshot[which(test_data$pitcher == 663623)] <- "https://a.espncdn.com/combiner/i?img=/i/headshots/mlb/players/full/41290.png&w=350&h=254"
test_data$headshot[which(test_data$pitcher == 663372)] <- "https://a.espncdn.com/combiner/i?img=/i/headshots/mlb/players/full/4019484.png&w=350&h=254"
test_data <- test_data %>% select(-pitcher) %>% select(headshot, everything())

t10 <- test_data %>% head(10)
b10 <- test_data %>% tail(10) %>% arrange(dfep)

gt_align_caption <- function(left, right) {
  caption <- paste0(
    '<span style="float: left;">', left, '</span>',
    '<span style="float: right;">', right, '</span>'
  )
  return(caption)
}

caption = gt_align_caption("Data from <b>Statcast</b>", "Amrit Vignesh | <b>@avsportsanalyst</b>")

gt_func <- function(df, str) {
  save_table <- df %>% gt() %>% 
    gt_fmt_mlb_logo(columns = "abbr") %>%
    gt_img_rows(columns = "headshot") %>%
    gt_theme_538() %>%
    cols_align(
      align = "center",
      columns = c(headshot, name, abbr, actual_fastball, actual_breaking, actual_offspeed, pred_fastball, pred_breaking, pred_offspeed, dfep)
    ) %>%
    gt_hulk_col_numeric(c(actual_fastball, actual_breaking, actual_offspeed, pred_fastball, pred_breaking, pred_offspeed, dfep)) %>%
    cols_label(
      headshot = md(""),
      name = md("**Pitcher**"),
      abbr = md("**Team**"),
      actual_fastball = md("**Fastball %**"),
      actual_breaking = md("**Breaking %**"),
      actual_offspeed = md("**Offspeed %**"),
      pred_fastball = md("**xFastball %**"),
      pred_breaking = md("**xBreaking %**"),
      pred_offspeed = md("**xOffspeed %**"),
      dfep = md("**DFEP**")
    ) %>%
    tab_header(
      title = paste0("2024 MLB Pitcher Deviation From Expected Pitch ", str, " 10"),
      subtitle = md("*Updated Until **07/04/24***")
    ) %>%
    opt_align_table_header(align = "center") %>%
    tab_style(
      style = list(
        cell_text(weight = "bold")
      ),
      locations = cells_body(
        columns = c(name, actual_fastball, actual_breaking, actual_offspeed, pred_fastball, pred_breaking, pred_offspeed, dfep)
      )
    ) %>%
    tab_source_note(html(caption)) 
  
  return(save_table)
}

t10_table <- gt_func(t10, "Top")
b10_table <- gt_func(b10, "Bottom")
gtsave(t10_table, "t10_table.png")
gtsave(b10_table, "b10_table.png")