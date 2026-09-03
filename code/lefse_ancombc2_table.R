library(dplyr)
library(tidyr)

# -----------------------------
# LEfSe genera
# -----------------------------

lefse_genus <- lefse_plot %>%
  select(enrich_group,Genus) %>%
  filter(!is.na(Genus)) %>%
  distinct() %>%
  mutate(LEfSe = "*")


# -----------------------------
# ANCOM-BC2 genera
# -----------------------------

ancom_EoCRC <- sig_EoCRC %>%
  select(Genus) %>%
  filter(!is.na(Genus)) %>%
  distinct() %>%
  mutate(EoCRC_vs_Control = "#")

ancom_LoCRC <- sig_LoCRC %>%
  select(Genus) %>%
  filter(!is.na(Genus)) %>%
  distinct() %>%
  mutate(LoCRC_vs_Control = "#")

ancom_LoCRC_EoCRC <- sig_LoCRC_vs_EoCRC %>%
  select(Genus) %>%
  filter(!is.na(Genus)) %>%
  distinct() %>%
  mutate(LoCRC_vs_EoCRC = "#")


# -----------------------------
# Combine everything
# -----------------------------

library(dplyr)

combined_table_ordered <- combined_table %>%
  mutate(
    enrich_group = case_when(
      is.na(enrich_group) ~ "ANCOM-BC2 only",
      TRUE ~ as.character(enrich_group)
    ),
    enrich_group = factor(
      enrich_group,
      levels = c(
        "Control",
        "EoCRC",
        "LoCRC",
        "ANCOM-BC2 only"
      )
    ),
    
    # Identify taxa supported by both approaches
    Concordant = case_when(
      LEfSe == "*" &
        (
          EoCRC_vs_Control == "#" |
          LoCRC_vs_Control == "#" |
          LoCRC_vs_EoCRC == "#"
        ) ~ "* + #",
      TRUE ~ ""
    )
  ) %>%
  arrange(enrich_group, Genus)

# Write to CSV
write.csv(
  combined_table_ordered,
  "LEfSe_ANCOMBC2_combined_table.csv",
  row.names = FALSE
)