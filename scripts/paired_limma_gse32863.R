args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: Rscript paired_limma_gse32863.R expression.tsv samples.tsv output_dir")
if (!requireNamespace("limma", quietly = TRUE)) stop("Bioconductor limma is required")

expression_path <- args[[1]]
samples_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expression <- read.delim(expression_path, row.names = 1, check.names = FALSE)
samples <- read.delim(samples_path, stringsAsFactors = FALSE)
if (!setequal(unique(samples$included), c("True", "False"))) stop("Unexpected inclusion flags")
samples <- samples[samples$included == "True", , drop = FALSE]
if (nrow(expression) != 19404 || ncol(expression) != 114 || nrow(samples) != 114) {
  stop("Unexpected GSE32863 paired-cohort dimensions")
}
if (!setequal(colnames(expression), samples$sample_id)) stop("Sample IDs differ")
samples <- samples[match(colnames(expression), samples$sample_id), , drop = FALSE]
if (anyNA(samples$sample_id) || anyNA(expression)) stop("Missing sample or expression value")
counts <- table(samples$patient_id, samples$condition)
if (!identical(colnames(counts), c("Normal", "Tumor")) ||
    nrow(counts) != 57 || any(counts != 1L)) stop("Pairing is invalid")

samples$patient_id <- factor(samples$patient_id)
samples$condition <- factor(samples$condition, levels = c("Normal", "Tumor"))
design <- model.matrix(~ patient_id + condition, data = samples)
if (qr(design)$rank != ncol(design)) stop("Paired design is rank deficient")
fit <- limma::lmFit(as.matrix(expression), design)
fit <- limma::eBayes(fit)
table <- limma::topTable(fit, coef = "conditionTumor", number = Inf,
                         sort.by = "none", adjust.method = "BH")
output <- data.frame(gene_symbol = rownames(table), log2FC = table$logFC,
                     p_value = table$P.Value, fdr = table$adj.P.Val,
                     stringsAsFactors = FALSE)
write.table(output, file.path(output_dir, "luad_limma_all.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "limma_session_info.txt"))
cat(sprintf("limma paired design: %d genes, %d pairs, residual df %d\n",
            nrow(expression), nrow(counts), fit$df.residual[[1]]))
