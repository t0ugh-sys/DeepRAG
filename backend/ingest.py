import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np

from sentence_transformers import SentenceTransformer

import faiss

from backend.config import Settings, ensure_dirs

# Milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_markdown_file(path: str) -> str:
    return read_text_file(path)


def read_pdf_file(path: str) -> str:
    # 尽量少依赖：用 pdfminer.six 简单抽取文本
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("请安装 pdfminer.six 以解析 PDF：pip install pdfminer.six") from exc
    return extract_text(path)


def load_documents(docs_dir: str) -> List[Dict[str, Any]]:
    supported_ext = {".txt", ".md", ".pdf"}
    documents: List[Dict[str, Any]] = []
    ensure_dirs(docs_dir)
    for root, _, files in os.walk(docs_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in supported_ext:
                continue
            full_path = os.path.join(root, name)
            if ext == ".txt":
                text = read_text_file(full_path)
            elif ext == ".md":
                text = read_markdown_file(full_path)
            else:
                text = read_pdf_file(full_path)
            documents.append({
                "path": full_path,
                "text": text,
            })
    return documents


def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[str]:
    # Markdown 优化：按标题、列表、代码块分段，尽量保持语义边界
    # 关键改进：确保标题和内容在同一块中
    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    in_code = False
    fence = None
    min_chunk_lines = 3  # 最小保留行数，避免只有标题的块

    def flush():
        if buf and len(buf) >= min_chunk_lines:
            blocks.append("\n".join(buf).strip())
            buf.clear()
        elif buf:
            # 如果内容太少（比如只有标题），保留在 buf 中继续累积
            pass

    for raw in lines:
        line = raw.rstrip("\n\r")
        # 代码块围栏
        if line.strip().startswith("```"):
            if not in_code:
                flush()
                in_code = True
                fence = line.strip()
                buf.append(line)
                continue
            else:
                buf.append(line)
                in_code = False
                fence = None
                flush()
                continue
        if in_code:
            buf.append(line)
            continue
        # 标题：作为新段落的开始
        if line.startswith(('#', '##', '###')):
            flush()  # 先清空之前的内容（如果足够长）
            buf.append(line)  # 标题作为新段落的开始
            continue
        # 分隔线作为段落边界
        if line.strip() in {'---', '***'}:
            flush()
            continue
        # 空行：只在累积内容足够多时才 flush
        if not line.strip():
            if len(buf) > min_chunk_lines:
                flush()
            continue
        buf.append(line)
    # 最后清空剩余内容（忽略最小行数限制）
    if buf:
        blocks.append("\n".join(buf).strip())

    # 长块二次切分
    chunks: List[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
            continue
        start = 0
        while start < len(block):
            end = min(start + chunk_size, len(block))
            piece = block[start:end]
            chunks.append(piece)
            if end >= len(block):
                break
            start = max(0, end - chunk_overlap)
    return [c for c in chunks if c.strip()]


def build_index(docs: List[Dict[str, Any]], settings: Settings, index_dir: str) -> None:
    # 仍保留本地 meta.jsonl 以便调试与溯源；优先写 Milvus，失败则写 FAISS 本地索引
    ensure_dirs(index_dir)
    model = SentenceTransformer(settings.embedding_model_name)

    all_chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc in docs:
        path = doc["path"]
        text = doc["text"]
        chunks = split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "path": path,
                "chunk_id": idx,
                "chunk_size": len(chunk),
            })

    if not all_chunks:
        raise RuntimeError("未在文档中解析到内容，请检查 docs 目录与文件格式。")

    embeddings = model.encode(all_chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    wrote_milvus = False
    if settings.vector_backend in ("auto", "milvus"):
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password,
                secure=settings.milvus_secure,
            )
            dim = embeddings.shape[1]
            # 与服务端保持一致：集合名加命名空间后缀，默认使用 Settings.default_namespace
            collection_name = f"{settings.milvus_collection}_{settings.default_namespace}"
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="RAG 文档分片")
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.drop()
            collection = Collection(collection_name, schema=schema)
            index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
            collection.create_index(field_name="embedding", index_params=index_params)
            paths = [m["path"] for m in metadatas]
            chunk_ids = [int(m["chunk_id"]) for m in metadatas]
            texts = list(all_chunks)
            collection.insert([paths, chunk_ids, texts, embeddings])
            collection.flush()
            wrote_milvus = True
            print("Milvus 写入完成")
        except Exception as e:
            print(f"Milvus 写入失败，回退到 FAISS: {e}")

    if not wrote_milvus:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("需要 FAISS 以使用本地回退索引，请使用 conda 安装 faiss-cpu: conda install -n rag-env -c conda-forge faiss-cpu") from exc
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
        print("FAISS 索引已写入")

    with open(os.path.join(index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for meta, chunk in zip(metadatas, all_chunks):
            rec = {"meta": meta, "text": chunk}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地 RAG 索引")
    parser.add_argument("--docs_dir", type=str, default=Settings().docs_dir)
    parser.add_argument("--index_dir", type=str, default=Settings().index_dir)
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--chunk_overlap", type=int, default=80)
    args = parser.parse_args()

    settings = Settings()
    ensure_dirs(args.docs_dir)
    ensure_dirs(args.index_dir)

    docs = load_documents(args.docs_dir)
    # 使用参数化切分
    global split_text  # type: ignore
    orig_split = split_text

    def split_text_param(t: str) -> List[str]:  # noqa: ANN001
        return orig_split(t, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    split_text = split_text_param  # type: ignore
    build_index(docs, settings, args.index_dir)
    print(f"索引已构建完成，保存至: {args.index_dir}")


if __name__ == "__main__":
    main()


