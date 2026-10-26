# Level3 completed by a human-imitation neural model

**Verified ending: LEVEL3 COMPLETE, followed by Level4 – The Sea Battle.**
The model runs from the recorded Level3 intro checkpoint, before gameplay,
through the level using only current RAM observations and ordinary controller
inputs. It is a deliberately simple clone of the selected human route.

|Evidence|Native frame from start|Artifact|
|---|---:|---|
|Candidate flag (not sufficient alone)|47675|frame_047675.png|
|Visible LEVEL3 COMPLETE|47975|[Completion card](frame_047975.png)|
|Visible Level4 – The Sea Battle|49175|[Next-level title](frame_049175.png)|

[Full run and transition video](episode.mp4). The video records every other
native frame at30FPS; native timing is60FPS. It retains49775
frames through the transition(829.58 simulated seconds).
No Level4 gameplay competence is claimed.

## Observation space and model

The separate `level3_raw_ram_imitation_v1` contract exposes a copied2048-byte
`uint8` RAM observation from the current native frame. All bytes are opaque
numeric inputs; no unverified Level2 enemy, collision, health, score or resource
semantics were applied to Level3. Training-only per-byte mean/std normalization
is checkpointed, with0.02 standard-deviation floor after division by255.
Internal game counters are observable RAM; no external frame/record/reset ID,
future state or playback cursor is injected.

The feed-forward MLP is2048→512→256→22, with1186070 neural
parameters and4096 normalization-buffer values. Its22 logits map directly to
the exact raw controller masks observed in the human recording. There is no
nearest-neighbour/action table, saved-action substitution or human data loading
during inference. Every live mask was independently recomputed from the actor.

## Training and provenance

The source is the hash-pinned read-only human SQLite recording. Extraction
follows its selected ancestry from path44700 through93800:49100 contiguous
pre/post RAM transitions and human action labels. The original recording used
rewinds; selected ancestry is training provenance only. The executed run starts
from one initial full state and has **zero mid-episode resets or rewinds**.
All49100 source rows are training data; current RAM states have no conflicting
action labels in this extraction.

The initial20000-update fit reached49036/49100 action agreement. A separate
lower-rate hard-example continuation reached49100/49100 after15500 updates,
verified using the same single-row CUDA execution as live inference. The model
then played through the level; training fit was not substituted for a completion
test. Total supervised updates:35500; no emulator frames were used by
the fitting scripts. No Level1/2 weights or altered schemas were used.

The standalone runner loads only model/reset/native assets and small contract
metadata, not demonstration RAM/actions. It provides all controller inputs,
including the observed post-clear menu inputs, from the neural output. No RAM
intervention, source-frame action playback, action retrieval or optimizer update
occurs during the run. The first49100 live source-range frames matched the
reference RAM exactly. The remaining675 frames are outside the training extract
and are retained as transition tail, not evidence of meaningful generalization.

## Independent checks and limitations

-360 source transitions at early and late checkpoints reproduce all2048 post-RAM
 bytes exactly. Total native preparation/reference work:1512 frames.
-All49775 live neural actions were independently recomputed;
 source/hash lineage, observation dtype/shape, training-only normalization,
 masks, resource receipts and video decoding passed.
-The completion card matched an independently rendered Level3 reference. The
 Level4 title matched exactly after a2-native-pixel vertical scroll correction.
 The first reviewer required a fixed title position and failed that title check;
 its report is preserved as verification_before_title_scroll_fix.json. The
 corrected checker permits only exact RGB matching within±4 vertical pixels,
 not fuzzy text similarity. Two tests cover scroll, wrong text and blank images.
-The title begins with a stale incoming flag127. Fresh gameplay is observed
 before accepting a later candidate flag. A byte flag alone is never accepted.
-Three observation/model tests pass. Resources remain within the project's
 existing limits; live uncertainty is0 and no original data was modified.

**Scope:** completion on this human-trained recording-derived start/route.
This does not establish robustness to new starts, altered actions, seeds or
independent recordings, nor RL-only learning. That matches the requested simple
human-cloning objective. It is not whole-game completion.

## Model and reproduction

-Checkpoint:`training/level3_imitation_refine_v2_20261026/best.pt`
-SHA-256:`1481dd27a35a04e97bcb9dd25717171d16d25fa9063e18cda69e404f9f5afbcb`
-Reset:`analysis/level3_imitation_v1_20261026_preparation/frame_44700.state`
-Selection/proof:`selected_policy.json`, `verification.json`
-Full RAM/masks:`episode.npz`; run provenance:`authority.json`
-Guide:`docs/LEVEL3_IMITATION.md`

Run from the project root with a **fresh unused output directory**:

```powershell
.venv-gpu/Scripts/python.exe run.py evaluate_level3_imitation --checkpoint training/level3_imitation_refine_v2_20261026/best.pt --output analysis/level3_fresh_run
.venv-gpu/Scripts/python.exe run.py review_level3_imitation --output analysis/level3_fresh_run
```

Do not reuse occupied study paths. Across preparation and live evaluation:
new native frames51287,uncertainty0,supervised updates35500.
No job remains active and no old policy was overwritten or globally promoted.
