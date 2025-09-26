param(
  [string]$DocsDir = "data/docs",
  [string]$IndexDir = "data/index"
)

conda activate rag-env | Out-Null
python ingest.py --docs_dir $DocsDir --index_dir $IndexDir

