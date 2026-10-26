# Reviewer execution checklist

Run from the repository root, after installing the pinned
dependencies described in its `docs/SETUP.md`. These are documented commands,
not commands executed for the case-study revision. Use fresh output directories.

CPU model inspection does not require the ROM or reset states:

```powershell
python run.py inspect_models
python run.py submission_assets audit
```

Gameplay remains blocked until the original compatible assets are supplied:

```powershell
python run.py evaluate_selected_level1 --output analysis/reviewer_l1
python run.py evaluate_level2_confirmed_completion --output analysis/reviewer_l2
python run.py evaluate_level3_imitation --checkpoint training/level3_imitation_refine_v2_20261026/best.pt --output analysis/reviewer_l3
python run.py review_level3_imitation --output analysis/reviewer_l3
```

| Workflow | Required private/runtime inputs |
|---|---|
| All gameplay | Matching ROM, recorder DLL and state/build identity |
| Level 1 | Verified route/reset-389 tree, exact observer DLL/provenance and pinned runtime dependencies |
| Level 2 | Saved title entry/reset and preparation metadata |
| Level 3 inference | Frame-44700 reset and preparation metadata |
| Level 3 independent review | Saved evaluation actions/RAM, reference dataset and actively rendered terminal references |
| Imitation fitting | Level-specific source data; Level 2 also requires the v56 parent and original labels |

Exact filenames, destinations, hashes and restoration conditions are in the
package's `manifests/artifacts.json`. There are no public download URLs. Owner
access can address missing files if redistribution is permitted; it cannot by
itself resolve licensing or establish native rebuild/state compatibility.

Fresh training also needs its applicable protocol/resource authority. Historical
DQfD receipts cannot authorize a new training launch. None of these commands
constitutes the missing official ten-episode acceptance campaign.
