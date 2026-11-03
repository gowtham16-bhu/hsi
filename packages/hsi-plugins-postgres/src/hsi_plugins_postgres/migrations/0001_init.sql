create table hsi.passages (
    corpus_id  text not null,
    id         text not null,
    doc_id     text not null,
    title      text,
    text       text not null,
    hash       char(64) not null,
    acl        jsonb not null,
    created_at timestamptz not null default now(),
    primary key (corpus_id, id)
);

create table hsi.edges (
    corpus_id text not null,
    src       text not null,
    dst       text not null,
    type      text not null,
    weight    real not null check (weight > 0 and weight <= 1),
    bridge    text not null,
    salience  real not null check (salience >= 0 and salience <= 1),
    primary key (corpus_id, src, dst, bridge)
);
create index edges_dst_idx on hsi.edges (corpus_id, dst);

create table hsi.shortcuts (
    corpus_id text not null,
    id        text not null,
    passages  text[] not null check (cardinality(passages) between 2 and 4),
    bridges   text[] not null,
    log_prob  double precision not null check (log_prob <= 0),
    level     smallint not null check (level between 1 and 3),
    key_mode  text not null,
    benefit   double precision not null default 0,
    version   int not null default 1,
    primary key (corpus_id, id)
);

create table hsi.runs (
    id           uuid primary key default gen_random_uuid(),
    created_at   timestamptz not null default now(),
    kind         text not null,
    config_hash  text,
    code_version text,
    data_hash    text,
    payload      jsonb not null
);
