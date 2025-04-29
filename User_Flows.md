# User Flows

## Doctor Flow

| Step | Actor / UI                         | Behind‑the‑scenes components                                                                            |
|------|------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1    | Dictate findings                   | Whisper‑large → text → ReportGenerator Agent                                                             |
|      | *Dr Ghosn taps the mic on the Next.js mobile app and says:* “Patient MRN 8041, 8 a.m.: K⁺ still 5.6 mmol/L after two doses of insulin‑glucose; maintaining sinus rhythm, urine output 35 ml/h.” |                                                                                                         |
| 2    | Auto‑rapport                       | LLM uses function‑calling to emit JSON; FastAPI validates schema, writes to Postgres via MCP; note is embedded into pgvector. |
|      | *Within 3 s, a structured “Progress Note” appears: narrative + JSON fields (lab_trend, assessment, plan).* |                                                                                                         |
| 3    | Alert query                        | A GraphRAG query template executes in Neo4j/AGE: MATCH (p:Patient)-[:HAS_LAB]->(l:Lab {name:'K'}) WHERE l.value > 5.5 AND l.delta_hours > 24 RETURN p, l; IDs returned to DataFetcher Agent. |
|      | *In the ward dashboard the doctor clicks “Persistent Hyper‑K?”* |                                                                                                         |
| 4    | AI insight                         | Agent formats answer, cites graph path + latest note chunks; ComplianceGuard checks PHI exposure.         |
|      | *Chat panel replies: “3 patients meet the criteria: MRN 8041, 7903, 8117. 8041 has rising creatinine; consider renal consult.”* |                                                                                                         |
| 5    | One‑click document                  | Existing notes + meds feed create_discharge_pdf function; react‑pdf renders, stores in documents table; doctor downloads PDF. |
|      | *Dr Ghosn presses “Generate Discharge Summary” for MRN 8041.* |                                                                                                         |

## Patient Flow

| Step | Actor / UI                                         | Behind‑the‑scenes components                                                                         |
|------|----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 1    | Secure login                                       | Auth0 issues a JWT mapped to Postgres role patient_8041 (row‑level security).                        |
|      | *Patient Layal logs into the React‑native portal with two‑factor SMS.* |                                                                                                      |
| 2    | Natural‑language question                           | Intent → “explanation”; orchestrator picks GraphRAG retriever.                                         |
|      | *She types: “Why am I taking Enoxaparin and when can I stop?”* |                                                                                                      |
| 3    | Graph lookup                                        | Query pattern: MATCH (p:Patient {id:8041})-[:HAS_DIAGNOSIS]->(d)<-[:TREATS]-(m:Medication {name:'Enoxaparin'}) RETURN d, m, p. |
|      |                                                    | Returns edge linking DVT diagnosis to Enoxaparin order.                                               |
| 4    | Friendly answer                                     | LLM cites the diagnosis note & order entry; Guard agent rewrites any jargon.                          |
|      | *Chatbot replies: “You’re on Enoxaparin to prevent clots after your leg surgery (diagnosed DVT on 14 Apr). Your doctor plans to stop it on 28 Apr, once ultrasound confirms no new clots.”* |                                                                                                      |
| 5    | Take‑home sheet                                     | Backend merges structured meds JSON into a PDF (react‑pdf), stores in S3/minio, returns a signed URL. |
|      | *Layal taps “Download medication schedule”.*        |                                                                                                      |
