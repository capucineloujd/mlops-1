-- Table alimentee par Fluentd
CREATE TABLE IF NOT EXISTS application_logs (
    id SERIAL PRIMARY KEY,
    log_time TIMESTAMPTZ,
    event TEXT,
    raw_json TEXT
);
