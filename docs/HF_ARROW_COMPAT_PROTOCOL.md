# Hugging Face Arrow Compatibility Protocol

This document defines the CAIDBench-compatible convention for Hugging Face
dataset repositories that store image samples as Arrow shards plus lightweight
metadata sidecars. It is intentionally compatible with existing repositories
such as `nebula/DF-arrow`.

The goal is simple:

- keep Arrow rows compact and stable;
- store labels, splits, and subset membership in sidecar JSON files;
- let CAIDBench normalize all sources into one metadata table;
- keep continual-learning task order outside the dataset package, in protocol
  YAML files.

## Repository Layout

A Hugging Face dataset repository may contain one or more source directories.
Each source directory is one independently loadable Hugging Face
`save_to_disk()` dataset.

```text
<hf_dataset_repo>/
  README.md
  <source_name>/
    data-00000-of-000NN.arrow
    data-00001-of-000NN.arrow
    ...
    dataset_info.json
    state.json
    mapping.json
    train.json        # optional, if train split exists
    val.json          # optional, if val split exists
    test.json         # optional, if test split exists
```

Examples:

```text
nebula/DF-arrow/
  CDDB/
  DIF/
  DiffusionForensics/
  ForenSynths/
  GANGen-Detection/
  GenImage_test/
  Ojha/
  synthbuster/
  synthwildx/
```

The same convention also works if each source directory is uploaded as a
separate Hugging Face dataset repository.

## Arrow Row Fields

The Arrow dataset should contain sample-level storage fields. These are the
minimum fields expected by the compatibility loader:

| field | type | required | meaning |
|---|---:|---:|---|
| `image_path` | string | yes | Stable sample key, relative POSIX path. |
| `image` | binary | yes | Encoded image bytes. |
| `md5` | string | recommended | Digest of the encoded image or source file. |
| `width` | int64 | recommended | Image width in pixels. |
| `height` | int64 | recommended | Image height in pixels. |
| `label` | int64 | optional | `0=real`, `1=fake`; sidecars may provide it. |
| `split` | string | optional | `train`, `val`, or `test`; sidecars may provide it. |
| `source_dataset` | string | optional | Original dataset/source name. |

The current compatibility contract does not require `generator`, `domain`, or
`task_id` inside the Arrow rows. These can be recovered from split sidecars,
source directory names, or protocol YAML filters.

## Path Convention

`image_path` is the primary sample key.

Rules:

- use relative POSIX paths, even on Windows;
- do not use absolute filesystem paths;
- keep paths unique inside a source directory;
- use the same paths in `image_path`, `mapping.json`, and split JSON files.

Recommended path shape:

```text
<subset_or_generator>/<split>/<class_dir>/<filename>
```

Examples:

```text
biggan/train/0_real/0--n01440764_5043.png
biggan/train/1_fake/100001_output.png
cyclegan/0_real/apple_n07740461_2550_real.png
mj/1_fake/img_img_993074131635544074.png
```

The loader may use path components as a fallback, but the sidecar JSON files
are the source of truth for labels, splits, and subset membership.

## `mapping.json`

`mapping.json` maps sample path to Arrow row index.

```json
{
  "biggan/train/0_real/0--n01440764_5043.png": 0,
  "biggan/train/0_real/101--n01871265_2897.png": 1
}
```

Contract:

- keys must match `image_path`;
- values are zero-based Arrow row indexes;
- every path referenced by a split JSON file must exist in `mapping.json`;
- row indexes must be in range for the Arrow dataset.

## Split Sidecars

Split sidecars are AID-style JSON files named by split:

```text
train.json
val.json
test.json
```

Each file stores:

```text
subset_name -> {image_path -> label}
```

Example:

```json
{
  "biggan": {
    "biggan/train/0_real/0--n01440764_5043.png": 0,
    "biggan/train/1_fake/100001_output.png": 1
  },
  "crn": {
    "crn/train/0_real/00103793.png": 0,
    "crn/train/1_fake/100001_output.png": 1
  }
}
```

The filename defines the split. In the example above, all paths are interpreted
as `split=train` because they are in `train.json`.

Labels:

```text
0 = real
1 = fake
```

Subset names are semantic membership labels. In CAIDBench they can be used as
`subset`, `aid_subset`, `task_hint`, and often as `generator`.

Reserved subset names:

| subset | meaning |
|---|---|
| `all` | All samples in the split. |
| `real` | Real samples. |
| `fake` | Fake samples. |

Non-reserved subset names such as `biggan`, `crn`, `cyclegan`, `mj`, or
`stargan` are treated as generator/subset identifiers by convention.

## Packaging Flow

The packer should build one source directory at a time.

Input manifest fields:

| field | required | meaning |
|---|---:|---|
| `image_path` or `path` | yes | Relative sample key. |
| `file_path` | yes | Local image file to read. |
| `label` | yes | `0=real`, `1=fake`. |
| `split` | yes | `train`, `val`, or `test`. |
| `subset` or `generator` | yes | Group key for split sidecars. |
| `source_dataset` | recommended | Original dataset/source name. |
| `md5` | optional | Can be computed during packing. |
| `width`, `height` | optional | Can be computed during packing. |

Output steps:

1. Read image bytes and build Arrow rows with `image_path`, `md5`, `width`,
   `height`, and `image`.
2. Optionally include `label`, `split`, and `source_dataset` in Arrow rows for
   easier inspection.
3. Save the Hugging Face dataset directory with `Dataset.save_to_disk()`.
4. Write `mapping.json` from row order:

   ```text
   image_path -> row_index
   ```

5. Write one split JSON file per split:

   ```text
   subset_name -> {image_path -> label}
   ```

6. Upload the source directory or full repository with
   `huggingface_hub.HfApi.upload_folder()`.

Reference Python skeleton:

```python
from datasets import Dataset

rows = {
    "image_path": [],
    "md5": [],
    "width": [],
    "height": [],
    "image": [],
    "label": [],
    "split": [],
    "source_dataset": [],
}

# Fill rows from a manifest, reading image bytes into rows["image"].
ds = Dataset.from_dict(rows)
ds.save_to_disk("out/CDDB")
```

After `save_to_disk()`, write sidecars:

```python
mapping = {path: i for i, path in enumerate(rows["image_path"])}

split_payloads = {
    "train": {},
    "val": {},
    "test": {},
}

for path, label, split, subset in manifest_rows:
    split_payloads[split].setdefault(subset, {})[path] = int(label)
```

## Reading Flow

The reader normalizes each source directory into CAIDBench metadata.

For each configured source:

1. Download the Hugging Face repository or source directory.
2. Load the local source directory with `datasets.load_from_disk()`.
3. Read `mapping.json`.
4. Read all available split sidecars.
5. Build metadata rows from split sidecars:

   ```text
   path
   label
   split
   subset / aid_subset / task_hint
   _rowid
   ```

6. Merge optional Arrow row fields such as `md5`, `width`, `height`,
   `source_dataset` by `_rowid` or `image_path`.
7. Fill CAIDBench fields:

   | CAIDBench field | source |
   |---|---|
   | `path` | split sidecar path or `image_path`. |
   | `label` | split sidecar label, then Arrow row label. |
   | `split` | split sidecar filename, then Arrow row split. |
   | `dataset` | `source_dataset`, configured source name, or directory name. |
   | `domain` | `dataset` unless configured otherwise. |
   | `generator` | non-reserved subset name, then first path component. |
   | `subset` | semicolon-separated subset memberships. |
   | `task_hint` | same as `subset`. |
   | `_rowid` | `mapping.json` row index. |

8. Concatenate all configured sources.
9. Apply protocol YAML filters to form continual tasks.

If a source directory has Arrow `label` and `split` fields but no split
sidecars, the reader may fall back to those fields. This fallback loses subset
membership and is less expressive.

## CAIDBench Config Shape

Recommended future config for one Hugging Face repository containing multiple
source directories:

```yaml
scenario:
  data:
    backend: hf_arrow_collection
    repo_id: nebula/DF-arrow
    revision: main
    cache_dir: data/hf_cache
    sources:
      - name: CDDB
        path: CDDB
      - name: DIF
        path: DIF
      - name: DiffusionForensics
        path: DiffusionForensics
  protocol: protocols/examples/cddb_hard.yaml
```

Recommended future config for one source per Hugging Face repository:

```yaml
scenario:
  data:
    backend: hf_arrow_collection
    cache_dir: data/hf_cache
    sources:
      - name: CDDB
        repo_id: nebula/CDDB-arrow
      - name: DIF
        repo_id: nebula/DIF-arrow
      - name: Ojha
        repo_id: nebula/Ojha-arrow
  protocol: protocols/examples/cddb_hard.yaml
```

Current CAIDBench supports one remote Arrow/AID source through the regular
`aid_arrow` backend. Set `remote.platform` to choose between equivalent mirrors:
install optional dependencies with `pip install -e ".[arrow,hub]"`.

```yaml
scenario:
  data:
    backend: aid_arrow
    image_column: image
    remote:
      platform: huggingface   # or: modelscope
      repo_ids:
        huggingface: nebula/CDDB.arrow
        modelscope: yabinnng/CDDB.arrow
      local_dir: data/datasets/CDDB.arrow
      path_in_repo: .
  protocol: protocols/examples/cddb_arrow_subsets.yaml
```

If the upstream repository stores multiple source directories, set
`path_in_repo`, for example `path_in_repo: CDDB`. CAIDBench downloads only that
subdirectory when possible and passes the resulting local directory to the same
Arrow/AID reader.

The protocol continues to describe task order and task filtering:

```yaml
name: cddb_hard
tasks:
  - id: biggan
    name: BigGAN
    numeric_id: 0
    filter:
      include:
        subset: biggan
  - id: crn
    name: CRN
    numeric_id: 1
    filter:
      include:
        subset: crn
```

## Validation Rules

A packaged source is valid when:

- `dataset_info.json` and `state.json` exist;
- at least one `data-*.arrow` file exists;
- `mapping.json` exists;
- at least one split sidecar exists, unless Arrow rows contain both `label`
  and `split`;
- every split sidecar path exists in `mapping.json`;
- labels are integers in `{0, 1}`;
- paths are unique within the source;
- Arrow row count is compatible with `mapping.json` row indexes;
- if Arrow rows contain `image_path`, `mapping.json` keys match those paths.

## Design Notes

This convention deliberately keeps task definitions outside the dataset
package. A dataset package says what samples exist, which split they belong to,
what their labels are, and which subset memberships they have. A protocol YAML
then decides how these samples become continual-learning tasks.
