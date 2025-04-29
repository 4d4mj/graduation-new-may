erDiagram
    %% --- core tables ---
    USERS {
        BIGSERIAL    id            PK
        TEXT         email         "UNIQUE"
        TEXT         password_hash
        TEXT         role          "doctor,patient,admin"
        BIGINT       person_id     FK
        TIMESTAMPTZ  created_at
    }
    PATIENTS {
        BIGSERIAL    id            PK
        TEXT         mrn           "UNIQUE"
        TEXT         first_name
        TEXT         last_name
        DATE         dob
        CHAR         sex           "M/F"
        TEXT         phone
        TEXT         address
        TEXT         language_code "ISO-639-1"
        TIMESTAMPTZ  created_at
    }
    DOCTORS {
        BIGSERIAL    id            PK
        TEXT         npi           "UNIQUE"
        TEXT         first_name
        TEXT         last_name
        TEXT         specialty
        TIMESTAMPTZ  created_at
    }
    VISITS {
        BIGSERIAL    id            PK
        BIGINT       patient_id    FK
        BIGINT       doctor_id     FK
        DATE         admit_date
        DATE         discharge_date
        TEXT         reason
        TEXT         ward
    }

    NOTES {
        BIGSERIAL    id               PK
        BIGINT       visit_id         FK
        BIGINT       author_id        FK
        TEXT         note_type
        TEXT         body
        TEXT         sensitivity_level "restricted,confidential"
        TIMESTAMPTZ  created_at
    }
    NOTE_EMBEDDINGS {
        BIGINT       note_id          PK,FK
        VECTOR       embedding
    }

    %% --- NEW: Note version history ---
    NOTE_VERSIONS {
        BIGSERIAL    id            PK
        BIGINT       note_id       FK
        INT          version_no
        TEXT         body
        BIGINT       edited_by     FK
        TIMESTAMPTZ  edited_at
    }

    %% --- NEW: Vital signs ---
    VITALS {
        BIGSERIAL    id            PK
        BIGINT       visit_id      FK
        TIMESTAMPTZ  taken_at
        TEXT         vital_type    "BP_sys,HR,SpO2,Temp"
        NUMERIC      value
        TEXT         unit
    }

    %% --- NEW: Allergies ---
    ALLERGIES {
        BIGSERIAL    id            PK
        BIGINT       patient_id    FK
        TEXT         substance
        TEXT         reaction
        TEXT         severity      "mild,moderate,severe"
        BOOLEAN      active        DEFAULT false
        TIMESTAMPTZ  recorded_at
    }

    %% --- NEW: Insurance / coverage ---
    INSURANCE_POLICIES {
        BIGSERIAL    id            PK
        BIGINT       patient_id    FK
        TEXT         provider_name
        TEXT         policy_number
        DATE         coverage_start
        DATE         coverage_end
        TEXT         plan_type     "basic,gold,etc"
    }

    %% --- NEW: Appointments & recurring visits ---
    APPOINTMENTS {
        BIGSERIAL    id            PK
        BIGINT       patient_id    FK
        BIGINT       doctor_id     FK
        TIMESTAMPTZ  scheduled_at
        TEXT         status        "scheduled,completed,cancelled"
        TEXT         reason
        TEXT         recurrence_rule "iCal RRULE"
        BIGINT       visit_id      FK
    }

    %% --- NEW: Audit trail ---
    AUDIT_LOGS {
        BIGSERIAL    id            PK
        BIGINT       user_id       FK
        TEXT         action        "SELECT,INSERT,UPDATE,DELETE"
        TEXT         table_name
        BIGINT       record_id
        JSONB        diff          "before/after"
        TIMESTAMPTZ  logged_at
    }

        LABS {
        BIGSERIAL    id            PK
        BIGINT       visit_id      FK
        TEXT         test_name
        NUMERIC      value
        TEXT         unit
        TEXT         ref_range     "e.g. 3.5‑5.0"
        TIMESTAMPTZ  collected_at
        TEXT         status        "pending,final,cancelled"
    }

    %% --- NEW: Medication orders / prescriptions ---
    MEDICATION_ORDERS {
        BIGSERIAL    id            PK
        BIGINT       visit_id      FK
        TEXT         drug_name
        NUMERIC      dose
        TEXT         unit          "mg, mL"
        TEXT         route         "oral,IV,SC"
        NUMERIC      frequency
        TEXT         frequency_unit "per_day,per_hour"
        DATE         start_date
        DATE         end_date
        TEXT         status        "active,stopped,held"
    }

    %% --- NEW: Generated or uploaded documents ---
    DOCUMENTS {
        BIGSERIAL    id            PK
        BIGINT       visit_id      FK
        TEXT         doc_type      "discharge_pdf,med_schedule,consent"
        TEXT         file_url      "S3 / Minio presigned"
        TEXT         sensitivity_level "restricted,confidential"
        TIMESTAMPTZ  created_at
    }

    DIAGNOSES {
        BIGSERIAL id            PK
        BIGINT    visit_id      FK
        TEXT      icd_code
        TEXT      description
        DATE      diagnosed_on
        TEXT      status        "active,resolved"
        TEXT      certainty     "confirmed,probable"
    }

    %% --- relationships ---
    USERS     ||--o{ AUDIT_LOGS         : records
    USERS     ||--o{ PATIENTS           : owns
    USERS     ||--o{ DOCTORS            : owns

    PATIENTS  ||--o{ VISITS             : "has"
    DOCTORS   ||--o{ VISITS             : "conducts"

    VISITS    ||--o{ NOTES              : contains
    NOTES     ||--|| NOTE_EMBEDDINGS    : "has embedding"
    NOTES     ||--o{ NOTE_VERSIONS      : "version history"

    VISITS    ||--o{ DIAGNOSES          : "identifies"
    VISITS    ||--o{ VITALS             : measures
    VISITS    ||--o{ LABS               : orders
    VISITS    ||--o{ MEDICATION_ORDERS  : prescribes
    VISITS    ||--o{ DOCUMENTS          : stores

    PATIENTS  ||--o{ ALLERGIES          : records
    PATIENTS  ||--o{ INSURANCE_POLICIES : covers
    PATIENTS  ||--o{ APPOINTMENTS       : schedules

    DOCTORS   ||--o{ APPOINTMENTS       : sets
