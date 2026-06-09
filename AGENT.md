# AGENT.md

Read [docs/LOGGING_GUIDE.md](./docs/LOGGING_GUIDE.md) before changing logging behavior.

## Working rules

- Preserve the separation between event logs and scalar metric logs.
- Keep console output stable and readable.
- Keep progress logging on a fixed cadence.
- Avoid per-batch noise unless debugging is enabled.
- Keep experiment payloads structured and scalar-only.
- When adding new logs, match the existing prefix and field style.
