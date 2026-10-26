# Export tooling

Rebuild the curated repository from the original research workspace into a new directory:

```powershell
python packaging/prepare_export.py --source <research-checkout> --output <new-empty-path>
```

The exporter follows selected dependency closures, copies allowlisted artifacts, applies portability changes, installs the documentation in `documentation.json`, and records checks and hashes. It refuses an existing destination.

`finalize_export.py` handles finalization; `apply_documentation.py` maintains reader-facing guides and local-link/inventory records. `templates/` contains the small submission-specific CLI tools. Rebuilding requires the original workspace and dependency environment; the public repository alone does not contain all source data. No training or emulator gameplay is launched by the exporter.
