
  ########################################
library(dplyr)
library(ggrepel)

Control <- gut %>%
  filter(Patient_Status == "Control")

EoCRC <- gut %>%
  filter(Patient_Status == "EoCRC")

LoCRC <- gut %>%
  filter(Patient_Status == "LoCRC")

Control_genus <- names(Control)[
  colSums(Control[, -(1:2)] > 0) > 0
]

EoCRC_genus <- names(EoCRC)[
  colSums(EoCRC[, -(1:2)] > 0) > 0
]

LoCRC_genus <- names(LoCRC)[
  colSums(LoCRC[, -(1:2)] > 0) > 0
]

library(ggVennDiagram)
library(ggplot2)
ggVennDiagram(
  list(
    Control = Control_genus,
    EoCRC = EoCRC_genus,
    LoCRC = LoCRC_genus
  )
) +
  scale_fill_gradient(low = "white", high = "pink")


################################################
### Venn diagram for significant genera
################################################

library(ggVennDiagram)

venn_list <- list(
  "EoCRC vs Control" = sig_EoCRC$Genus,
  "LoCRC vs Control" = sig_LoCRC$Genus,
  "EoCRC vs LoCRC"   = sig_LoCRC_vs_EoCRC$Genus
)

# ggVennDiagram(venn_list)
ggVennDiagram(
  venn_list,
  label_alpha = 0
) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_void()
