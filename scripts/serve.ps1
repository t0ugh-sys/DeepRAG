param(
  [string]$Host = "0.0.0.0",
  [int]$Port = 8000
)

conda activate rag-env | Out-Null
uvicorn server:app --host $Host --port $Port --reload

