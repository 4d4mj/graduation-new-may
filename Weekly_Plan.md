# Weekly Plan & Milestones

This plan groups the Definition of Done items into four-week sprints. It aligns with the high‑level DoD tasks.

| Week        | Milestones                        | Key tasks & tips                                                                                                                                                                      |
|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0 (Prep)    | Project scaffold & compliance     | • Init monorepo (`/frontend` + `/backend` or TurboRepo)
• Enable Git hooks for black, ruff, ESLint
• Draft a 1‑page Data‑Protection Impact Assessment (DPIA)                                                               |
| 1 – Data & MCP | DB schema solid + MCP up       | 1. Model core tables (patients, visits, labs, notes)
2. Enable Postgres RLS & create roles (doctor, patient)
3. Install pgvector; add embedding column to notes
4. Spin up `mcp-server-postgresql` in Docker; test SELECT & INSERT via reference client                                     |
| 2 – RAG & Reports | Prototype report generator   | 1. LangChain “retrieve‑then‑read” chain:
   - Embed incoming findings (openai-embedding-3)
   - Similarity search in pgvector
2. Define JSON schema `PatientReport` & enable function‑calling
3. Persist JSON back via MCP (INSERT INTO reports …)                                                                       |
| 3 – Agents & Ask‑Your‑Data | Multi‑agent orchestration | 1. LangGraph agents:
   - DataFetcher (read‑only)
   - ReportGenerator (RAG + write)
   - ComplianceGuard (regex/OpenAI Moderation)
2. Add Intent Classifier to route messages
3. Wire role‑aware JWT → Postgres RLS credentials                                                                       |
| 4 – UX polish & demo | Ship!                       | 1. Build chat UI (react‑chat‑elements) + tabs for “Reports” & “Documents”
2. PDF gen with `react-pdf` or `pdf-make`; branded discharge summary
3. Load‑test with Locust (50 users, <250 ms median)
4. Record 3‑min demo + slides on MCP, RAG & agents                                                                    |
