# Topic Atlas Auto Update Engine

This directory stores generated status files for the atlas update workflow.

The automation has two modes:

- `check`: runs on GitHub-hosted runners. It compares the committed atlas coverage with `configs/auto_update.yaml` and updates `status.md` / `status.json`.
- `refresh-atlas`: runs on a self-hosted runner labeled `ai-paper-trends`. It expects the ignored local `results/` cache to exist, reruns fine-grained clustering, rebuilds `docs/topic-atlas`, and pushes changed atlas files.

Conference entries are treated as proceedings snapshots: a venue-year becomes due after its expected proceedings month. Journal entries are treated as rolling streams: missing journal-years stay in the queue, and the current plus previous publication year are marked as `rolling_refresh` even when they are already indexed.

By default, the scheduled workflow only runs `check`. To make scheduled full refreshes automatic, add a repository variable named `AUTO_REFRESH_ATLAS` with value `true` and attach a self-hosted runner labeled `ai-paper-trends`.

Manual commands:

```bash
python scripts/auto_update_atlas.py check --write-report
python scripts/auto_update_atlas.py refresh
```

The status files are intentionally lightweight. The full embedding cache is not committed to GitHub.
