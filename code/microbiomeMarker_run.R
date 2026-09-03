library(phyloseq)
library(dplyr)
library(forcats)

## Matrix of taxa abundances (rows: taxa, columns: samples)
otu <- abund %>%
  select(-Patient_Status) %>%
  as.matrix()

rownames(otu) <- rownames(abund)

# phyloseq requires taxa as rows
otu <- t(otu)

OTU <- otu_table(otu, taxa_are_rows = TRUE)

meta <- abund %>%
  select(Patient_Status)

rownames(meta) <- rownames(abund)

SAM <- sample_data(meta)

######################################
## Taxa table
###################################
tax <- data.frame(
  Kingdom = "Bacteria",
  Phylum  = NA,
  Class   = NA,
  Order   = NA,
  Family  = NA,
  Genus   = rownames(otu),
  row.names = rownames(otu)
)

TAX <- tax_table(as.matrix(tax))

########################################
## Phyloseq object
########################################
ps <- phyloseq(
  OTU,
  SAM,
  TAX
)
ps


lefse_res <- microbiomeMarker::run_lefse(
  ps,
  group = "Patient_Status",
  kw_cutoff = 0.05,
  lda_cutoff = 2
)

microbiomeMarker::plot_ef_bar(lefse_res)
microbiomeMarker::plot_ef_dot(lefse_res)

png(
  "LEfSe_LDA_Barplot.png",
  width = 3200,
  height = 2200,
  res = 300
)

plot_ef_bar(lefse_res)

dev.off()

#############################################
## plot
#########################

library(dplyr)
library(forcats)

# lefse_results <- marker_table(lefse_res)
# lefse_results <- as.data.frame(marker_table(lefse_res))
lefse_results <- tibble::as_tibble(marker_table(lefse_res))
lefse_results <- lefse_results %>%
  mutate(
    Genus = gsub(".*\\|g__", "", feature)
  )

lefse_results <- lefse_results %>%
  mutate(
    FDR = p.adjust(pvalue, method = "BH")
  )  

lefse_plot <- lefse_results %>%

  # keep significant LEfSe biomarkers
  filter(
    ef_lda >= 2,
    pvalue < 0.05
  ) %>%

  mutate(

    Genus = Genus,
    Class = enrich_group,
    LDA = ef_lda,
    Pvalue = pvalue,

    # significance annotation
    sig = case_when(
      Pvalue < 0.001 ~ "***",
      Pvalue < 0.01  ~ "**",
      Pvalue < 0.05  ~ "*",
      TRUE ~ ""
    )
  ) %>%

  # put one group on negative side
  mutate(
    LDA = ifelse(
      Class == "LoCRC",
      -LDA,
      LDA
    )
  ) %>%

  arrange(LDA) %>%

  mutate(
    Genus = factor(
      Genus,
      levels = Genus
    )
  )
library(ggplot2)

p <- ggplot(
  lefse_plot,
  aes(
    y = Genus,
    x = LDA,
    fill = Class
  )
) +

geom_col(width = .75) +

## zero line
geom_vline(
  xintercept = 0,
  linewidth = .6,
  colour = "black"
) +

## significance stars
geom_text(
  aes(
    x = ifelse(LDA > 0,
               LDA + .15,
               LDA - .15),
    label = sig
  ),
  size = 4.5,
  fontface = "bold"
) +

scale_fill_manual(
  values = c(
    Control = "#4DBBD5",
    EoCRC = "#E64B35",
    LoCRC = "#00A087"
  )
) +

scale_x_continuous(

  limits = c(
    floor(min(lefse_plot$LDA))-0.5,
    ceiling(max(lefse_plot$LDA))+0.5
  )

) +

labs(
  x = "LDA score (log10)",
  y = NULL,
  fill = NULL
) +

theme_classic(base_size = 14) +

theme(

  legend.position = "top",

  legend.text = element_text(
    size = 12,
    face = "bold"
  ),

  axis.text.y.left = element_text(
    face = "italic",
    size = 10,
    colour = "black"
  ),

  axis.text.x = element_text(
    size = 11,
    colour = "black"
  ),

  axis.title.x = element_text(
    face = "bold",
    size = 13
  ),

  panel.border = element_rect(
    colour = "black",
    fill = NA,
    linewidth = .8
  ),

  plot.margin =
    margin(10,15,10,15)
)

p  


###############################################
## EoCRC vs Control LDA plot
##################################################

library(dplyr)
library(ggplot2)
library(forcats)

lefse_plot <- lefse_results %>%

  filter(enrich_group %in% c("Control","EoCRC")) %>%

  mutate(

    LDA = if_else(
      enrich_group == "EoCRC",
      -ef_lda,
      ef_lda
    ),

    sig = case_when(
      FDR < 0.001 ~ "***",
      FDR < 0.01  ~ "**",
      FDR < 0.05  ~ "*",
      TRUE ~ ""
    )

  ) %>%

  arrange(LDA) %>%

  mutate(
    Genus = factor(
      Genus,
      levels = Genus
    )
  )

lefse_fig <-

ggplot(
  lefse_plot,
  aes(
    x = LDA,
    y = Genus,
    fill = enrich_group
  )
) +

geom_col(
  width = 0.8,
  colour = "black",
  linewidth = 0.25
) +

geom_text(
  aes(
    label = sig,
    x = ifelse(LDA > 0,
               LDA - 0.15,
               LDA + 0.15)
  ),
  hjust = ifelse(lefse_plot$LDA > 0, 1.2, -0.2),
  fontface = "bold",
  size = 4
) +

geom_vline(
  xintercept = 0,
  linewidth = 0.5
) +

scale_fill_manual(

  values = c(

    Control = "#F1C232",
    EoCRC   = "#F39C12"

  )

) +

labs(

  x = "LDA SCORE (log10)",
  y = NULL

) +

theme_classic(base_size = 14) +

theme(
   
  legend.title = element_blank(),

  legend.position = "top",

  axis.text.y = element_text(
    face = "italic",
    size = 10
  ),

  axis.title.x = element_text(
    face = "bold"
  )

)

lefse_fig

ggsave(
  "LEfSe_Control_vs_EoCRC.png",
  lefse_fig,
  width = 8,
  height = 8,
  dpi = 600,
  bg = "white"
)

##################################################
## EoCRC vs LoCRC LDA plot
##################################################

library(dplyr)
library(ggplot2)
library(forcats)

lefse_plot <- lefse_results %>%

  # filter(enrich_group %in% c("Control","EoCRC")) %>%

  mutate(

    LDA = if_else(
      enrich_group == "LoCRC",
      -ef_lda,
      ef_lda
    ),

    sig = case_when(
      FDR < 0.001 ~ "***",
      FDR < 0.01  ~ "**",
      FDR < 0.05  ~ "*",
      TRUE ~ ""
    )

  ) %>%

  arrange(LDA) %>%

  mutate(
    Genus = factor(
      Genus,
      levels = Genus
    )
  )

lefse_fig <-

ggplot(
  lefse_plot,
  aes(
    x = LDA,
    y = Genus,
    fill = enrich_group
  )
) +

geom_col(
  width = 0.8,
  colour = "black",
  linewidth = 0.25
) +

geom_text(
  aes(
    label = sig,
    x = ifelse(LDA > 0,
               4.7,
               4.7)
  ),
  hjust = "left",
  vjust = 0.5,
  fontface = "bold",
  size = 4
) +

geom_vline(
  xintercept = 0,
  linewidth = 0.5
) +

scale_fill_manual(

  values = c(
LoCRC = "#00A087",
    Control = "gray80",
    EoCRC   = "#F39C12"

  )

) +

labs(

  x = "LDA SCORE (log10)",
  y = NULL

) +

theme_classic(base_size = 14) +

theme(
axis.line = element_blank(),
  legend.title = element_blank(),

  legend.position = "top",

  axis.text.y = element_text(
    face = "italic",
    size = 10
  ),

  axis.title.x = element_text(
    face = "bold"
  )

)

lefse_fig + theme(
  axis.text.y = element_blank(),
  axis.ticks.y = element_blank()
) +

geom_text(
  aes(
    x = max(abs(LDA)) + 0.2,
    label = Genus
  ),
  hjust = 0,
  fontface = "italic"
) 

lefse_final <- lefse_fig +

  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +

  geom_text(
    aes(
      x = max(abs(lefse_plot$LDA)) + 0.5,
      label = Genus
    ),
    hjust = 0,
    fontface = "italic",
    size = 3.5
  ) +

  coord_cartesian(clip = "off") +

  scale_x_continuous(
    expand = expansion(mult = c(0.02, 0.35))
  )

ggsave(
  "LEfSe_Control_vs_EoCRC.png",
  lefse_final,
  width = 9,
  height = 10,
  units = "in",
  dpi = 600,
  bg = "white",
  # compression = "lzw"
)
