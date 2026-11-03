# Rules for working in this repository

Read `specs/00-thesis.md` and `specs/01-constitution.md` before changing anything.

1. **Spec first.** Change the spec (or add a decision record in `specs/decisions/`) before changing code.
2. **No hard-coded knowledge (constitution P13).** No word lists, stop words, language rules,
   ID-prefix tables, model-name checks, prompt strings or magic thresholds in code. Learn them from
   the data, or read them from config or the model's own files. Every value must be visible in the
   run config.
3. **Core stays independent.** `hsi-core` imports only numpy, pydantic, yaml (checked by tests and
   import-linter).
4. **Real services in development.** PostgreSQL in every environment; no SQLite or in-memory stand-ins.
5. **No fake results.** Never report numbers from the hashing encoder or test fixtures as evidence.
   Report blocked steps as blocked.
6. **Before finishing:** `ruff check .`, `lint-imports`, `pytest` (with `HSI_PG_DSN` set).
