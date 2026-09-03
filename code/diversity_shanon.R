library(dplyr)

gut <- read.csv("data/Chemo_Gut_ML_clean.csv",
                check.names = FALSE)

metadataTable <- gut %>%
  dplyr::select(Patient_ID, Patient_Status)

rownames(metadataTable) <- metadataTable$Patient_ID

metadataTable <- metadataTable %>%
  dplyr::select(-Patient_ID)

featureTable <- gut %>%
  dplyr::select(-(Patient_ID:Patient_Status))

featureTable <- as.matrix(featureTable)

rownames(featureTable) <- gut$Patient_ID

#########################################

# Load phyloseq package
library(phyloseq)

# Component 1: Genus abundance table
# Convert abundance table into a phyloseq OTU table
# taxa_are_rows = FALSE because rows are samples and columns are genera
features <- otu_table(featureTable, taxa_are_rows = FALSE)


# Component 2: Metadata table
# Contains sample information (e.g., Patient_Status)
metadata <- sample_data(metadataTable)


# Combine abundance data and metadata into a phyloseq object
ps <- phyloseq(features, metadata)


# Check that the object is a phyloseq object
class(ps)


# Display summary of the phyloseq object
ps


# Save the phyloseq object for future analyses
save(ps, file = "phyloseq_object.RData")


# To reload the object later:
# load("phyloseq_object.RData")




################################################ 
## Create a summary table with total abundance per bacterial genus
#################################################
dataTable <- data.frame(
  TotalAbundance = taxa_sums(ps),
  Genus = taxa_names(ps)
)

# View the first rows
head(dataTable)


library(ggplot2)

abund_per_gene <- ggplot(dataTable, aes(x = TotalAbundance)) +
  geom_histogram(bins = 30) +
  ggtitle("Distribution of Total Abundance per Bacterial Genus") +
  labs(
    x = "Total Abundance per Genus",
    y = "Number of Genera"
  ) +
  theme_classic()


################################
## Alpha Diversity
##################################

# Estimate alpha diversity metrics using the phyloseq object
alpha_diversity <- estimate_richness(
  ps,
  measures = c("Shannon", "Simpson")
)

# Make sample IDs easier to work with
alpha_diversity$sample <- rownames(alpha_diversity)

# View alpha diversity results
head(alpha_diversity)


# Convert phyloseq sample_data into a data frame
metadata <- data.frame(sample_data(ps))

# Add sample IDs as a column
metadata$sample <- rownames(metadata)

# Combine metadata with alpha diversity metrics
alpha_div_meta <- merge(
  metadata,
  alpha_diversity,
  by = "sample"
)

# Check the combined table
head(alpha_div_meta)


library(ggplot2)

alpha_diversity_plot <- ggplot(alpha_div_meta,
       aes(x = Patient_Status,
           y = Shannon,
           fill = Patient_Status)) +
  geom_boxplot() +
  theme_classic() +
  labs(
    x = "",
    y = "Shannon diversity"
  )

kruskal.test(
  Shannon ~ Patient_Status,
  data = alpha_div_meta
)

pairwise.wilcox.test(
  alpha_div_meta$Shannon,
  alpha_div_meta$Patient_Status,
  p.adjust.method = "BH"
)



comparisons <- list(
  c("Control", "EoCRC"),
  c("Control", "LoCRC"),
  c("EoCRC", "LoCRC")
)

library(ggplot2)
library(ggpubr)


shannon_plot_sig <- ggplot(alpha_div_meta,
       aes(x = Patient_Status,
           y = Shannon,
           fill = Patient_Status)) +
  geom_boxplot(width = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.6, size = 1.5) +
  stat_compare_means(
    comparisons = comparisons,
    method = "wilcox.test",
    label = "p.signif"
  ) +
  stat_compare_means(
    method = "kruskal.test",
    label.y = max(alpha_div_meta$Shannon) + 0.8
  ) +
  theme_classic() +
  labs(
    x = "",
    y = "Shannon diversity"
  )


#######################################
## Beta Diversity
#######################################
library(phyloseq)

OTU <- otu_table(as.matrix(features), taxa_are_rows = FALSE)

SAM <- sample_data(metadata)

ps <- phyloseq(OTU, SAM)

sample_variables(ps)

head(data.frame(sample_data(ps)))


library(phyloseq)

dist_bc <- phyloseq::distance(ps, method = "bray")

ord <- ordinate(ps, method = "PCoA", distance = dist_bc)

plot_ordination(ps, ord, color = "Patient_Status")

library(vegan)

adonis2(
  dist_bc ~ Patient_Status,
  data = metadata
)
library(pairwiseAdonis)

pairwise.adonis2(
  dist_bc ~ Patient_Status,
  data = metadata
)

library(ggplot2)
library(dplyr)

# Extract PCoA coordinates
pcoa_df <- as.data.frame(ord$vectors[, 1:2])

# Add sample IDs
pcoa_df$Sample <- rownames(pcoa_df)

# Add metadata
meta_df <- data.frame(sample_data(ps))
meta_df$Sample <- rownames(meta_df)

# Merge
plot_df <- left_join(pcoa_df, meta_df, by = "Sample")

# Calculate variance explained
eig <- ord$values$Relative_eig * 100

# PCoA plot
p <- ggplot(
  plot_df,
  aes(
    x = Axis.1,
    y = Axis.2,
    color = Patient_Status
  )
) +
  geom_point(
    size = 3,
    alpha = 0.85
  ) +
  stat_ellipse(
    linewidth = 1
  ) +
  scale_color_manual(
    values = c(
      "Control" = "#1b9e77",
      "EoCRC" = "#d95f02",
      "LoCRC" = "#7570b3"
    )
  ) +
  labs(
    x = paste0(
      "PCoA1 (",
      round(eig[1], 1),
      "%)"
    ),
    y = paste0(
      "PCoA2 (",
      round(eig[2], 1),
      "%)"
    ),
    color = "Group"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "right",
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold")
  )
p

p +
annotate(
  "text",
  x = Inf,
  y = Inf,
  label = "PERMANOVA\nR² = 0.014, p = 0.001",
  hjust = 1.1,
  vjust = 1.5,
  size = 4
)

###############################################
## 
###############################################
library(ggplot2)

p2 <- ggplot(plot_df,
       aes(Axis.1, Axis.2, color = Patient_Status)) +
  stat_ellipse(aes(fill = Patient_Status),
               geom = "polygon",
               alpha = 0.15,
               colour = NA) +
  geom_point(size = 3, alpha = 0.9) +
  scale_color_manual(values = c(
    Control = "#1B9E77",
    EoCRC   = "#D95F02",
    LoCRC   = "#7570B3"
  )) +
  scale_fill_manual(values = c(
    Control = "#1B9E77",
    EoCRC   = "#D95F02",
    LoCRC   = "#7570B3"
  )) +
  labs(
    x = sprintf("PCoA1 (%.1f%%)", eig[1]),
    y = sprintf("PCoA2 (%.1f%%)", eig[2]),
    color = "Group",
    fill = ""
  ) +
  annotate(
    "text",
    x = Inf,
    y = Inf,
    hjust = 1.05,
    vjust = 1.2,
    size = 4.5,
    fontface = "italic",
    label = "PERMANOVA\nR² = 0.014\np = 0.001"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "right",
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    panel.border = element_rect(colour = "black", fill = NA)
  )

p + annotate(
  "text",
  x = Inf,
  y = Inf,
  hjust = 1.05,
  vjust = 1.2,
  size = 3.8,
  label = "
PERMANOVA
Overall: p = 0.001 (R² = 0.014)

Control vs EoCRC: p = 0.015
Control vs LoCRC: p = 0.001
EoCRC vs LoCRC: p = 0.001
"
)
