# Reinforcement Learning for Prince Valiant ? case study

Read [report.pdf](report.pdf): **10 main pages plus 9 appendix pages**.
The case study explains the RAM/emulator interface, demonstration recording,
PPO and DQfD work, reward/curriculum revisions, later-level imitation, measured
failures, and remaining acceptance/reproducibility gaps. It incorporates HR and
engineering review feedback. The earlier [draft](../case_study_draft/REPORT_DRAFT.pdf)
is retained for history.

## Build the PDF

Editable source: [report.tex](report.tex) and [references.bib](references.bib).
All required rendered figure assets are included in [assets/](assets/).
Using Tectonic 0.17.0 from this directory:

```powershell
tectonic --keep-logs --keep-intermediates report.tex
```

Tectonic obtains TeX resources on first use and runs bibliography/reference
passes automatically. Compiling this document does not require the ROM, native
DLLs, checkpoints, private recordings, or training datasets. No game execution
is part of the document build.

## Data and verification

[evidence_ledger.csv](evidence_ledger.csv) and its [JSON](evidence_ledger.json)
record claims, source identities, units and limitations. Source paths refer to
the original research workspace; not all cited private/experimental inputs are
included in this curated repository. [assets/source_manifest.json](assets/source_manifest.json)
records plotting provenance, and CSV files expose plotted values.

[pdf_validation.json](pdf_validation.json) records the 19-page rendering, boundary,
font and reference checks. No overfull boxes, unresolved references or missing
glyphs were detected. Three nonfatal underfull paragraph warnings remain.
[reward_component_verification.json](reward_component_verification.json) preserves
the distinction between raw reward arithmetic and semantic availability.

The selected route clears do not establish a 7/10 campaign, Level 1 normal-start
compliance, or independent-route robustness. See [results](../RESULTS.md) and
[setup](../SETUP.md) for executable scope and missing game/native assets.
The proposed CV fusion and equal-capacity teacher/student models have not been
trained. The public repository exists; the brief's private-access requirement
remains unresolved.
