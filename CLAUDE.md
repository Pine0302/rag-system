# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) system** for querying Obsidian notes. It uses:
- **FastAPI** - Web server with query endpoint
- **Qdrant** - Vector database for storing embeddings
- **LlamaIndex** - Document indexing and retrieval framework
- **HuggingFace BAAI/bge-base-zh-v1.5** - Chinese text embedding model

The system runs in offline mode (`HF_HUB_OFFLINE=1`) with models cached at `/models`.

## Architecture

```
              ┌─────────────┐
              │   Qdrant    │ ← Vector database (port 6333)
              │  (storage)  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  rag-server │ ← FastAPI (port 8000)
              │ (Python 3.10)│
              └──────┬──────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐      ┌────▼────┐     ┌────▼────┐
│ingest │      │ query   │     │  main   │
│.py    │      │_engine  │     │.py      │
└───────┘      │.py      │     └─────────┘
               └─────────┘
```

- **ingest.py** - Loads markdown files from `/vault/obsidian`, creates embeddings, stores in Qdrant
- **query_engine.py** - Connects to Qdrant and performs similarity search
- **main.py** - FastAPI app with `/query` endpoint

## Commands

### Start services
```bash
docker-compose up -d
```

### Rebuild and start
```bash
docker-compose up -d --build
```

### View logs
```bash
docker-compose logs -f rag-server
docker-compose logs -f qdrant
```

### Stop services
```bash
docker-compose down
```

## API Endpoints

- `GET /` - Health check
- `POST /query?q=<question>` - Query the RAG system

Example:
```bash
curl -X POST "http://localhost:8000/query?q=你的问题"
```

## Key Configuration

- **Vault path**: `/vault/obsidian` (mounted from host)
- **Models path**: `/models` (mounted from host, contains cached embeddings)
- **Collection name**: `obsidian_notes`
- **Embedding dimension**: 768 (bge-base-zh-v1.5)
- **Distance metric**: Cosine similarity

## Docker Build Optimization

To prevent "No space left on device" errors:

1. Use `python:3.10-slim` base image in Dockerfile
2. Pin dependency versions in requirements.txt (especially `torch==2.5.0`)
3. Keep Docker image cache after first build - don't run `docker system prune`

If disk space is full:
```bash
sudo docker system prune -a --volumes -f
```
