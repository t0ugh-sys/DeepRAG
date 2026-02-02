# AGENTS.md
Project-level guidelines for AI coding agents working on DeepRAG.

## Setup commands
- Backend install: `pip install -r backend/requirements.txt`
- Frontend install: `cd frontend` then `npm install`
- Backend dev server: `python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload`
- Frontend dev server: `cd frontend` then `npm run dev`

## Code style
### Python
- Use type hints for public functions and complex data structures.
- Prefer small, single-purpose functions; keep functions under ~100 lines.
- Avoid side-effectful imports; keep startup logic in `if __name__ == "__main__":` or app lifecycle hooks.
- Use f-strings; avoid string concatenation for readability.
- Example:
```python
def build_prompt(question: str, context: str) -> str:
    return f"Q: {question}\n\nContext:\n{context}"
```

### Vue (JavaScript)
- Use `const` by default; `let` only when reassigning.
- Prefer `async/await` over chained `.then()`.
- Keep component state minimal; derive values with computed properties.
- Example:
```js
const isEmpty = computed(() => items.value.length === 0)
```

## Testing instructions
- Backend tests: `cd backend` then `pytest`
- Frontend lint/tests (if added): `cd frontend` then `npm run lint` / `npm test`
- After changes: run the relevant tests for the area you modified.

## Project overview
- Backend: FastAPI app in `backend/`
- Frontend: Vue 3 + Vite in `frontend/`
- Data assets: `data/`
- Deployment: `deploy/`

## Dev environment tips
- Env vars: copy `.env.example` to `.env` and adjust for local use.
- API base URL: frontend reads `VITE_API_BASE_URL`.
- Namespace/API key: set `RAG_NAMESPACE` and `RAG_API_KEY` if needed.

## Rules
- Do not commit secrets or API keys.
- Avoid modifying generated directories (`data/index`, `node_modules`, build outputs).
- Keep changes small and focused; ask before large refactors.
- If behavior is uncertain, add a test or ask for confirmation.

## PR instructions
- Commit messages: Conventional Commits (feat/fix/refactor/test/chore).
- Describe impact and risk in PR/summary notes.
