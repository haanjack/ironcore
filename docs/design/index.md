# Design Docs

Design docs capture the **architecture and rationale** behind a subsystem — the *why* and *how it
fits together*, not the line-by-line *how* (that's the code) or the *how to use it* (that's the
[guides](../index.md)). They are the durable reference an engineer reads before changing a
subsystem.

The worked exemplar is the [Offload system design](offload.md).

## Standard

Each design doc lives at `docs/design/<subsystem>.md` and follows this skeleton (omit sections that
don't apply):

| Section | Purpose |
| --- | --- |
| **Overview** | One paragraph: what the subsystem does and the problem it solves. |
| **Target / constraints** | Hardware, scale, and assumptions the design optimizes for. |
| **Architecture** | The components and how they relate — lead with a diagram. |
| **Data flow** | What moves where, per operation (forward/backward, save/load, …). |
| **Component interactions** | How the pieces combine in the main scenarios. |
| **Configuration** | The config surface (fields, defaults, validation rules). |
| **Trade-offs / known bottlenecks** | Honest limits, measured where possible. |
| **File index** | Table mapping each module/file to its responsibility. |

Keep design docs verifiable: name real classes and file paths, and cite the tests or benchmarks
that back any performance claim.

## Diagram rule

Diagrams are expected wherever structure is non-trivial. Two tools, each for what it's best at:

| Use **Mermaid** (inline) for… | Use **Excalidraw** (asset) for… |
| --- | --- |
| Sequences, flows, lifecycles | Memory / spatial layouts |
| Decision trees, state machines | "Hero" architecture diagrams |
| Dependency / relationship graphs | Anything needing precise positioning or annotation |

- **Mermaid** is the default. Author it inline in a ` ```mermaid ` fenced block — MkDocs Material
  renders it at build time, and it diffs cleanly in git.
- **Excalidraw** is for the few diagrams Mermaid can't express well. Commit **both** the
  `.excalidraw` JSON source **and** the exported `.png` to `docs/design/assets/`, then embed the
  PNG. The JSON is the editable source of truth; the PNG is what the site shows.

```text
docs/design/
├── index.md            # this standard
├── <subsystem>.md      # one design doc per subsystem
└── assets/
    ├── <name>.excalidraw   # editable source
    └── <name>.png          # rendered, embedded in the .md
```

### Rendering Excalidraw

The `excalidraw-diagram` skill generates the JSON and renders it to PNG (Playwright + headless
Chromium). One-time setup:

```bash
cd ~/.claude/skills/excalidraw-diagram/references
uv sync && uv run playwright install chromium
```

Then render after editing a diagram:

```bash
uv run python ~/.claude/skills/excalidraw-diagram/references/render_excalidraw.py \
    docs/design/assets/<name>.excalidraw -o docs/design/assets/<name>.png
```
