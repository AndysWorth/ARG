---
globs: ["arg/**"]
---

ARG makes zero outbound network calls during operation. All Ollama calls must go to
`config.ollama_base_url` (localhost:11434). No `requests.*`, `httpx.*`, or external
`http(s)://` calls anywhere in `arg/`.

When staging git changes: explicit paths only. Never `git add -A` or `git add .`
