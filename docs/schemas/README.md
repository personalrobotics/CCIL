# Data schemas

These are the versioned data contracts of the `ccil` core (#16). They are
implemented in [`src/ccil/schema/`](../../src/ccil/schema/).

| Schema | Holds |
|---|---|
| `ccil.trajectories` 1.0 | Episodes of named observation modalities (`state`, `image.<camera>`), actions, termination, and optional timestamps, task, and action chunks. Transitions are identified by `(episode_id, frame)`. |
| `ccil.candidates` 1.0 | One row per attempted candidate label. Each row has its source identity, generator, observation, action and target, an accepted flag, independent rejection reasons, measurements and an optional weight. |

[`reference.md`](reference.md) is the normative field reference, generated from
the specs. Edit the specs, not the file, and regenerate it:

```bash
PYTHONPATH=src python -m ccil schema docs           # write docs/schemas/reference.md
PYTHONPATH=src python -m ccil schema docs --check   # fail if it is out of date
PYTHONPATH=src python -m ccil schema validate TRAJ.npz [--candidates CANDS.npz]
```

## Rules

- **Validation.** It reports every violation with its episode, modality and
  frame. It never coerces, broadcasts, truncates, clamps or mutates. Readers
  validate on load.
- **No executable formats.** Files are `.npz`, written atomically and read with
  `allow_pickle=False`.
- **Compatibility.**
  - A minor version only adds optional fields; a major version may break
    compatibility.
  - Readers reject unknown major versions, missing required fields, and unknown
    fields, except `ext/<name>` extensions and fields that a newer-minor file
    declares optional.
- **Legacy data.** `ccil.schema.legacy.from_m1_npz` converts the M1
  `ccil-trajectories/1` files into `ccil.trajectories` 1.0, one way. It records
  the source hash. Legacy episodes are state-only, identified by index, and
  marked truncated.

## Evidence

```bash
PYTHONPATH=src uv run --no-project --python 3.11 --with-requirements tools/m2_schemas/requirements.txt \
  python tools/m2_schemas/check.py --out <new-dir>
```

[`../fixtures/evidence/m2-schemas/report.json`](../fixtures/evidence/m2-schemas/report.json)
has 26 cases:

- **Valid examples round-trip exactly:** vector, multimodal image, action
  chunks, and multi-reason candidates.
- **One invalid example per rule,** each with the expected diagnostic and the
  input left unmodified.
- **File-level compatibility rules.**
- **The analytic fixture, end to end** from trajectories to accepted labels,
  with source identity and measurements intact.
- **All 15 shipped datasets** converted from the M1 format exactly.
