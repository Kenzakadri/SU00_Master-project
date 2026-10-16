################################################################################
# Gene regulatory network & master regulator analysis — osteosarcoma metastasis
#
# Master 2 Systems Biology (Sorbonne Université), 2019
#
# Data   : GEO series GSE21257 (osteosarcoma biopsies, Illumina platform GPL10295)
# Goal   : identify transcription factors (master regulators) whose regulons are
#          associated with the metastasis phenotype.
#
# Pipeline
#   1. Download data from GEO and check sample distributions
#   2. Infer a transcriptional regulatory network (RTN: mutual information,
#      permutation test, bootstrap, DPI filter)
#   3. Differential expression between groups (limma, GEO2R-style)
#   4. Master Regulator Analysis (MRA) + GSEA-1 / GSEA-2 on regulons
#   5. Network visualisation (RedeR)
#
# Packages: Biobase, GEOquery, limma, RTN, snow, RedeR (Bioconductor)
# Original environment: R 3.2.3, Biobase 2.30.0, GEOquery 2.40.0, limma 3.26.8
#
# Note: permutation / bootstrap / GSEA steps are slow. Intermediate results were
# saved as .RData checkpoints (not included in this repo). Set
# use_checkpoints <- TRUE to reload them instead of recomputing.
################################################################################

library(Biobase)
library(GEOquery)
library(limma)
library(RTN)
library(snow)
library(RedeR)

use_checkpoints <- FALSE

################################################################################
# 1. Load data from GEO
################################################################################

if (file.exists("gset.RDS")) {
  gset <- readRDS("gset.RDS")
} else {
  # getGPL = TRUE also retrieves the platform annotation (probe ID -> gene symbol)
  # Offline alternative: download
  #   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE21nnn/GSE21257/matrix/GSE21257_series_matrix.txt.gz
  # and use getGEO(filename = "GSE21257_series_matrix.txt.gz", GSEMatrix = TRUE, getGPL = TRUE)
  gset <- getGEO("GSE21257", GSEMatrix = TRUE, getGPL = TRUE)
  if (length(gset) > 1) idx <- grep("GPL10295", attr(gset, "names")) else idx <- 1
  gset <- gset[[idx]]
  saveRDS(gset, file = "gset.RDS")
}

# Boxplot of expression values per sample (quality check)
dev.new(width = 4 + dim(gset)[[2]] / 5, height = 6)
par(mar = c(2 + round(max(nchar(sampleNames(gset))) / 2), 4, 2, 1))
title <- paste("GSE21257", "/", annotation(gset), " selected samples", sep = "")
boxplot(exprs(gset), boxwex = 0.7, notch = TRUE, main = title, outline = FALSE, las = 2)

################################################################################
# 2. Infer the regulatory network (RTN)
################################################################################

data("tfsData")                   # reference list of transcription factors
expData <- exprs(gset)            # expression matrix (genes x samples)

# Map probe IDs to gene symbols so RTN can match TFs by symbol
annotations <- subset(pData(featureData(gset)), select = c("ID", "Symbol"))

rtni <- tni.constructor(expData            = expData,
                        regulatoryElements = tfsData$all$SYMBOL,
                        rowAnnotation      = annotations,
                        colAnnotation      = pData(phenoData(gset)),
                        cvfilter           = TRUE)

if (use_checkpoints) {
  load("rtni_dpi_filter_3.RData")   # contains 'rtni' after permutation, bootstrap and DPI
} else {
  # Parallelise the permutation step over 2 cores (otherwise too slow)
  options(cluster = snow::makeCluster(spec = 2, "SOCK"))
  rtni <- tni.permutation(rtni, nPermutations = 1000)
  stopCluster(getOption("cluster"))

  rtni <- tni.bootstrap(rtni, nBootstraps = 100)

  # Data Processing Inequality filter: removes the weakest edge in each
  # triangle to keep mostly direct TF-target interactions
  rtni <- tni.dpi.filter(rtni)
}

################################################################################
# 3. Differential expression between the two groups (limma, GEO2R-style)
################################################################################

fvarLabels(gset) <- make.names(fvarLabels(gset))

# Group assignment for the 53 samples (0 / 1 = metastasis status)
gsms <- "11111111111111111111100001111011010001010011110000000"
sml  <- strsplit(gsms, "")[[1]]

# log2-transform if the data are not already on a log scale
ex <- exprs(gset)
qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = TRUE))
LogC <- (qx[5] > 100) ||
        (qx[6] - qx[1] > 50 && qx[2] > 0) ||
        (qx[2] > 0 && qx[2] < 1 && qx[4] > 1 && qx[4] < 2)
if (LogC) {
  ex[which(ex <= 0)] <- NaN
  exprs(gset) <- log2(ex)
}

fl <- as.factor(paste("G", sml, sep = ""))
gset$description <- fl
design <- model.matrix(~ description + 0, gset)
colnames(design) <- levels(fl)

fit  <- lmFit(gset, design)
cont <- makeContrasts(G1 - G0, levels = design)
fit2 <- eBayes(contrasts.fit(fit, cont), 0.01)

cols <- c("ID", "adj.P.Val", "P.Value", "t", "B", "logFC", "Symbol")

# Top 250 differentially expressed genes (FDR-adjusted) -> "hits"
tT <- subset(topTable(fit2, adjust = "fdr", sort.by = "B", number = 250), select = cols)

# All genes -> phenotype vector (logFC) used for GSEA
allGenes <- subset(topTable(fit2, adjust = "fdr", sort.by = "B", number = Inf), select = cols)

hits      <- tT[tT$Symbol != "", ]$Symbol
phenotype <- allGenes$logFC
names(phenotype) <- allGenes$Symbol

################################################################################
# 4. Master Regulator Analysis and GSEA
################################################################################

rtna <- tni2tna.preprocess(object = rtni, phenotype = phenotype, hits = hits)

if (use_checkpoints) {
  load("rtna_mra_3.RData")
  load("rtna_gsea1_3.RData")
  load("rtna_gsea2_3.RData")
} else {
  # MRA: hypergeometric test of overlap between each regulon and the hits
  rtna <- tna.mra(rtna)
  # GSEA-1: is each regulon enriched at the extremes of the ranked phenotype?
  rtna <- tna.gsea1(rtna, stepFilter = TRUE, nPermutations = 1000)
  # GSEA-2: two-tailed test, separating positive and negative targets
  rtna <- tna.gsea2(rtna, stepFilter = TRUE, nPermutations = 1000)
}

mra   <- tna.get(rtna, what = "mra",   ntop = -1)   # ntop = -1 returns all results
gsea1 <- tna.get(rtna, what = "gsea1", ntop = -1)
gsea2 <- tna.get(rtna, what = "gsea2", ntop = -1)
head(mra)

tna.plot.gsea1(rtna, labPheno = "Metastasis phenotype", file = "gsea1_results",
               ntop = 5, regulon.order = "adj.pvalue")
tna.plot.gsea2(rtna, labPheno = "Metastasis phenotype",
               tfs = c("IRF8", "SP140", "VAV1", "VENTX", "HOXC4"))

################################################################################
# 5. Network visualisation with RedeR
################################################################################

# Significant master regulators (MRA adj. p <= 0.05), coloured by GSEA-1 enrichment
g <- tna.graph(rtna, gtype = "amap", tfs = mra[mra$Adjusted.Pvalue <= 0.05, ]$Regulon)
g <- att.mapv(g, dat = gsea1)
g <- att.setv(g = g, from = "Adjusted.Pvalue", to = "nodeColor",
              breaks = seq(0, 0.1, 0.01), pal = 2)

rdp <- RedPort()
calld(rdp)                          # opens the RedeR interface
addGraph(rdp, g)
addLegend.color(rdp, colvec = g$legNodeColor$scale, labvec = g$legNodeColor$legend,
                title = "enrichment (adj. p-value)")
addLegend.size(rdp, g, type = "node", position = "bottomright")
addLegend.size(rdp, g, type = "edge", position = "topright", dxborder = 40,
               title = "Jaccard coefficient", vertical = FALSE)
relax(rdp, ps = TRUE)

# Candidate regulators highlighted in the analysis: IRF8, VAV1, VENTX, MNDA, SP140, HOXC4, HOXA10
