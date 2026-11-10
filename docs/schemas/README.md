# Data schemas

These are the versioned data contracts of the `ccil` core (#16). They are
implemented in [`src/ccil/schema/`](../../src/ccil/schema/).

The schemas are `ccil.trajectories`, `ccil.candidates`, `ccil.certificates`, `ccil.selection`,
`ccil.diagnosis`, `ccil.calibration`, `ccil.reference-audit`, and the `ccil.generate-result` and
`ccil.curate-result` records of what a command wrote; each version and field is in the reference below.

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

Reports: [`m2-schemas/report.json`](../fixtures/evidence/m2-schemas/report.json).
Each case, and the contract it protects, is listed in the docstring of [`tools/m2_schemas`](../../tools/m2_schemas/).
