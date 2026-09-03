library(phyloseq)
library(ANCOMBC)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(tidyr)
library(pheatmap)
library(ComplexHeatmap)
library(circlize)
library(grid)
library(ggpubr)
library(microbiomeMarker)
library(patchwork)

# Load data
# abund <- read.csv("data/differential_abundant_genera.csv", row.names = 1)

abund <- read.csv("data/Chemo_Gut_ML_clean.csv", row.names = 1)
abund_clean <- abund[, !(names(abund) %in% "Patient_Status")]
meta  <- read.csv("data/metadata.csv", row.names = 1)

abund_clean[abund_clean == 0] <- 1e-6

rownames(meta) <- rownames(abund_clean)

OTU <- otu_table(as.matrix(abund_clean), taxa_are_rows = FALSE)
colnames(OTU) <- colnames(abund_clean)  # restore correct taxa names

META  <- sample_data(meta)

physeq  <- phyloseq(OTU, META)

# res  <- ancombc(
#   phyloseq = physeq,
#   formula = "Patient_Status",
#   p_adj_method = "fdr",
#   group = "Patient_Status"
# )

sample_data(physeq)$Patient_Status <- relevel(
  factor(sample_data(physeq)$Patient_Status),
  ref = "Control"
)

levels(sample_data(physeq)$Patient_Status)


res2 <- ancombc2(
  data = physeq,
  assay_name = "counts",
  fix_formula = "Patient_Status",
  group = "Patient_Status",
  p_adj_method = "BH",
  prv_cut = 0.10,
  lib_cut = 0,
  alpha = 0.05,
  global = TRUE,
  pairwise = TRUE,
  dunnet = FALSE,
  trend = FALSE,
  verbose = TRUE
)

res_out <- res2$res_pair

#######################
## Table
#######################

results <- data.frame(
  Genus = res_out$taxon,

  LogFC_EoCRC_vs_Control = res_out$lfc_Patient_StatusEoCRC,
  LogFC_LoCRC_vs_Control = res_out$lfc_Patient_StatusLoCRC,
  LogFC_LoCRC_vs_EoCRC = res_out$lfc_Patient_StatusLoCRC_Patient_StatusEoCRC,

  P_EoCRC_vs_Control = res_out$p_Patient_StatusEoCRC,
  P_LoCRC_vs_Control = res_out$p_Patient_StatusLoCRC,
  P_LoCRC_vs_EoCRC = res_out$p_Patient_StatusLoCRC_Patient_StatusEoCRC,

  FDR_EoCRC_vs_Control = res_out$q_Patient_StatusEoCRC,
  FDR_LoCRC_vs_Control = res_out$q_Patient_StatusLoCRC,
  FDR_LoCRC_vs_EoCRC = res_out$q_Patient_StatusLoCRC_Patient_StatusEoCRC
)

head(results)

sig_LoCRC <- subset(
  results,
  FDR_LoCRC_vs_Control < 0.05
)

sig_EoCRC <- subset(
  results,
  FDR_EoCRC_vs_Control < 0.05
)


sig_LoCRC_vs_EoCRC <- subset(
  results,
  FDR_LoCRC_vs_EoCRC < 0.05
)
# res_out <- res$res
# qval <- res_out$q_val

# #######################
# ## Table
# ######################
# results <- data.frame(
#   Genus = res$res$lfc$taxon,
#   LogFC_EoCRC_vs_Control = res$res$lfc$Patient_StatusEoCRC,
#   LogFC_LoCRC_vs_Control = res$res$lfc$Patient_StatusLoCRC,
#   P_EoCRC_vs_Control = res$res$p_val$Patient_StatusEoCRC,
#   P_LoCRC_vs_Control = res$res$p_val$Patient_StatusLoCRC,
#   FDR_EoCRC_vs_Control = res$res$q_val$Patient_StatusEoCRC,
#   FDR_LoCRC_vs_Control = res$res$q_val$Patient_StatusLoCRC
# )

# head(results)

# sig_LoCRC <- subset(
#   results,
#   FDR_LoCRC_vs_Control < 0.05
# )

# sig_LoCRC

# sig_EoCRC <- subset(
#   results,
#   FDR_EoCRC_vs_Control < 0.05
# )

# sig_EoCRC


# library(dplyr)

# results_long <- results %>%
#   tidyr::pivot_longer(
#     cols = c(
#       LogFC_EoCRC_vs_Control,
#       LogFC_LoCRC_vs_Control,
#       P_EoCRC_vs_Control,
#       P_LoCRC_vs_Control,
#       FDR_EoCRC_vs_Control,
#       FDR_LoCRC_vs_Control
#     ),
#     names_to = c(".value", "Comparison"),
#     names_pattern = "(.*)_(EoCRC_vs_Control|LoCRC_vs_Control)"
#   )

# head(results_long)

# EoCRC_vs_Control <- results_long %>%
#   filter(Comparison == "EoCRC_vs_Control")

# LoCRC_vs_Control <- results_long %>%
#   filter(Comparison == "LoCRC_vs_Control")

############################
## Table EoCRC vs LoCRC
#############################

# META$Patient_Status <- factor(
#   META$Patient_Status,
#   levels = c("LoCRC", "Control", "EoCRC")
# )

# sample_data(physeq)$Patient_Status <- META$Patient_Status

# res_LoCRC_ref <- ancombc(
#   phyloseq = physeq,
#   formula = "Patient_Status",
#   p_adj_method = "fdr",
#   group = "Patient_Status"
# )
# colnames(res_LoCRC_ref$res$lfc)

# EoCRC_vs_LoCRC <- data.frame(
#   Genus = res_LoCRC_ref$res$lfc$taxon,
#   LogFC = res_LoCRC_ref$res$lfc$Patient_StatusEoCRC,
#   Pvalue = res_LoCRC_ref$res$p_val$Patient_StatusEoCRC,
#   FDR = res_LoCRC_ref$res$q_val$Patient_StatusEoCRC,
#   Significant = res_LoCRC_ref$res$diff_abn$Patient_StatusEoCRC
# )

# head(EoCRC_vs_LoCRC)

# Eo_vs_Lo_sig <- subset(EoCRC_vs_LoCRC, FDR < 0.05)

library(openxlsx)

# Create a workbook
wb <- createWorkbook()

# Add worksheets
addWorksheet(wb, "LoCRC_vs_Control")
writeData(wb, "LoCRC_vs_Control", sig_LoCRC)

addWorksheet(wb, "EoCRC_vs_Control")
writeData(wb, "EoCRC_vs_Control", sig_EoCRC)

addWorksheet(wb, "LoCRC_vs_EoCRC")
writeData(wb, "LoCRC_vs_EoCRC", sig_LoCRC_vs_EoCRC)

# Save workbook
saveWorkbook(
  wb,
  file = "Significant_Genera.xlsx",
  overwrite = TRUE
)
# ##############################################################
## Volcano plot for EoCRC vs Control and LoCRC vs Control
# ##############################################################

volcano <- results %>%
  mutate(
    Direction = case_when(
      FDR_LoCRC_vs_Control < 0.05 &
        LogFC_LoCRC_vs_Control > 0 ~ "Higher in LoCRC",

      FDR_LoCRC_vs_Control < 0.05 &
        LogFC_LoCRC_vs_Control < 0 ~ "Higher in EoCRC",

      TRUE ~ "Not significant"
    ),
    negLogFDR = -log10(FDR_LoCRC_vs_Control)
  )

ggplot(volcano,
       aes(LogFC_LoCRC_vs_Control,
           negLogFDR,
           color = Direction)) +

  geom_point(size = 2.8, alpha = 0.8) +

  geom_text_repel(
  data = volcano %>%
    filter(Direction != "Not significant") %>%
    arrange(FDR_LoCRC_vs_Control) %>%   # ascending: smallest FDR first
    slice_head(n = 5),
  aes(label = Genus),
  size = 5,
  fontface = "italic",
  max.overlaps = Inf,
  box.padding = 0.5,
  point.padding = 0.3,
  segment.size = 0.4
) +

  geom_vline(xintercept = 0,
             linetype = "dashed") +

  geom_hline(yintercept = -log10(0.05),
             linetype = "dashed") +

  scale_color_manual(values = c(
    "Higher in LoCRC" = "#D73027",
    "Higher in EoCRC" = "#4575B4",
    "Not significant" = "grey80"
  )) +

  labs(
    x = "Log2 Fold Change (LoCRC vs EoCRC)",
    y = expression(-log[10](FDR))
  ) +

  theme_classic(base_size = 14)

####################################################################
## Re-usable function to create volcano plots for different comparisons
####################################################################

make_volcano <- function(data,
                         logfc_col,
                         fdr_col,
                         comparison_title,
                         group1,
                         group2,
                         label_n = 10) {

  volcano <- data %>%
    mutate(
      LogFC = .data[[logfc_col]],
      FDR = .data[[fdr_col]],
      negLogFDR = -log10(FDR),
      Direction = case_when(
        FDR < 0.05 & LogFC > 0 ~ "Higher group 1",
        FDR < 0.05 & LogFC < 0 ~ "Higher group 2",
        TRUE ~ "Not significant"
      )
    )

  ggplot(
    volcano,
    aes(x = LogFC,
        y = negLogFDR,
        color = Direction)
  ) +

    geom_point(
      size = 6.8,
      alpha = 0.8
    ) +
    geom_text_repel(
      data = volcano %>%
      filter(FDR < 0.05) %>%
      arrange(desc(abs(LogFC))) %>%
      slice_head(n = label_n),
      aes(label = Genus),
      size = 12,
      fontface = "italic",
      max.overlaps = Inf,
      box.padding = 0.5,
      point.padding = 0.3,
      segment.size = 0.4
    ) +

    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      colour = "grey40"
    ) +

    geom_hline(
      yintercept = -log10(0.05),
      linetype = "dashed",
      colour = "grey40"
    ) +

    scale_color_manual(
      name = "Direction",
      values = c(
        "Higher group 1" = "#D73027",
        "Higher group 2" = "#4575B4",
        "Not significant" = "grey80"
      ),
      labels = c(
        "Higher group 1" = paste("Higher in", group1),
        "Higher group 2" = paste("Higher in", group2),
        "Not significant" = "Not significant"
      )
    ) +

    labs(
      title = comparison_title,
      x = "Log2 Fold Change",
      y = expression(-log[10](FDR)),
      color = NULL
    ) +

    theme_classic(base_size = 14) +

    theme(
      legend.title = element_text(size = 21),
      legend.text = element_text(size = 18),
      axis.text = element_text(size = 25),
      legend.position = "right",
      plot.title = element_text(
        face = "bold",
        hjust = 0.5
      )
    )
}


### LoCRC_vs_EoCRC
vol_Eo_vs_Control <- make_volcano(
  data = results,
  logfc_col = "LogFC_EoCRC_vs_Control",
  fdr_col = "FDR_EoCRC_vs_Control",
  comparison_title = "EoCRC vs Control",
  group1 = "EoCRC",
  group2 = "Control",
  label_n = 11
)

vol_Eo_vs_Control

vol_Lo_vs_Control <- make_volcano(
  data = results,
  logfc_col = "LogFC_LoCRC_vs_Control",
  fdr_col = "FDR_LoCRC_vs_Control",
  comparison_title = "LoCRC vs Control",
  group1 = "LoCRC",
  group2 = "Control",
  label_n = 12
) + theme(
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_blank()
  )
  # + coord_flip()

vol_Lo_vs_Control

vol_Lo_vs_Eo <- make_volcano(
  data = results,
  logfc_col = "LogFC_LoCRC_vs_EoCRC",
  fdr_col = "FDR_LoCRC_vs_EoCRC",
  comparison_title = "LoCRC vs EoCRC",
  group1 = "LoCRC",
  group2 = "EoCRC",
  label_n = 11
)
# + coord_flip()

vol_Lo_vs_Eo 

#############################
## plot and save
##################################
volcano_combined <- vol_Eo_vs_Control +
  vol_Lo_vs_Control +
  vol_Lo_vs_Eo +
  plot_layout(
    nrow = 3,
    guides = "collect"
  ) +
  plot_annotation(
    tag_levels = "A"
  ) &
  theme(
    legend.position = "bottom",
    plot.tag = element_text(
      face = "bold",
      size = 25
    )
  )

volcano_combined

ggsave(
  filename = "CRC_volcano_three_comparisons1.png",
  plot = volcano_combined,
  width = 23,
  height = 30,
  units = "in",
  dpi = 600,
  bg = "white"
)


library(ComplexHeatmap)
library(grid)

ht_grob <- grid.grabExpr(
  draw(nicercomplexheatmap)
)

library(cowplot)

plot_grid(
  ggdraw(ht_grob),
  volcano_combined,
  ncol = 2,
  rel_widths = c(1, 1)
)


png(
  filename = "Heatmap_Volcano_Combined.png",
  width = 30,
  height = 20,
  units = "in",
  res = 600,
  bg = "white"
)

plot_grid(
  ggdraw(ht_grob),
  volcano_combined,
  ncol = 2,
  rel_widths = c(1, 1)
)

dev.off()
