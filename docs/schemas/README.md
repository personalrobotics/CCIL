# Data schemas

These are the versioned data contracts of the `ccil` core (#16). They are
implemented in [`src/ccil/schema/`](../../src/ccil/schema/).

| Schema | Holds |
|---|---|
| `ccil.trajectories` 1.0 | Episodes of named observation modalities (`state`, `image.<camera>`), actions, termination, and optional timestamps, task, and action chunks. Timestamps and action chunks are all-or-none across episodes. Transitions are identified by `(episode_id, frame)`. |
| `ccil.candidates` 2.3 | One row per attempted candidate label. Each row has its source identity, observation, action and target, an accepted flag, independent rejection reasons, measurements and a weight (0 on rejected rows). The file records the run ID, seed, generator, weighting, and the SHA-256 of each input, always including the transition model. An accepted label can therefore be traced from the candidate file alone. 2.0 made the weight and this provenance required. 2.1 added namespaced extension reasons (`<namespace>.<reason>`). A file declares 2.1 only when it uses one, so built-in-only files remain 2.0. 2.2 added optional `measurement_specs` (the definition, norm and units of every measurement) and the generator's full `params`. A file declares 2.2 only when it carries specs, which pipeline runs always do. 2.3 added optional action chunks (`action_chunks`, `action_chunk_mask`, and the meta `action_chunk_spec`). A file declares 2.3 only when it carries chunks, and lists the arrays as optional fields, so 2.2 readers ignore them. |

[`reference.md`](reference.md) is the normative field reference, generated from
the specs. Edit the specs, not the file, and regenerate it:

```bash
uv run ccil schema docs           # write docs/schemas/reference.md
uv run ccil schema docs --check   # fail if it is out of date
uv run ccil schema validate TRAJ.npz [--candidates CANDS.npz]
```

## Rules

- **Validation.** It reports every violation with its episode, modality and
  frame. It never coerces, broadcasts, truncates, clamps or mutates. Readers
  validate on load.
- **No executable formats.** Files are `.npz`, written atomically and read with
  `allow_pickle=False`.
- **Malformed files** raise `SchemaError` with a `format` diagnostic, never a
  raw decoding error. This covers files that aren't archives, truncated or
  pickled arrays, metadata that is non-JSON or mistyped, and inconsistent array
  layout.
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
uv run --group evidence python tools/m2_schemas/check.py --out <new-dir>
```

[`../fixtures/evidence/m2-schemas/report.json`](../fixtures/evidence/m2-schemas/report.json)
has 38 cases:

- **Valid examples round-trip exactly:** vector, multimodal image, action
  chunks, and multi-reason candidates.
- **One invalid example per rule,** each with the expected diagnostic and the
  input left unmodified.
- **File-level compatibility rules.**
- **The analytic fixture, end to end** from trajectories to accepted labels,
  with source identity and measurements intact.
- **All 15 shipped datasets** converted from the M1 format exactly.
- **Provenance:**
  - the seed, input hashes, weighting and weights survive a write–read round
    trip and `accepted_labels`;
  - missing or malformed provenance fails;
  - a 1.0 candidates file is rejected by its major version.
- **Mixed optional fields:** timestamps or action chunks on only some episodes,
  or differing chunk lengths, fail validation. The writer refuses them with a
  `SchemaError`.
- **Malformed files:** 13 trajectory, 5 candidate and 1 legacy file, each
  rejected with exit 1 and a `SchemaError` diagnostic, and no traceback.
