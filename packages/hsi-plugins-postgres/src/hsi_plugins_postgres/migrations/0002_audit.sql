-- Append-only audit log with a hash chain (specs/additions/A3 S8, A5 §5).
create table hsi.audit (
    seq        bigserial primary key,
    at         timestamptz not null,
    index_id   text not null,
    principals jsonb not null,
    question   text not null,
    passages   jsonb not null,
    prev_hash  char(64) not null,
    hash       char(64) not null
);
