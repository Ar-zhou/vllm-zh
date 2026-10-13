# Long-context performance datasets

This branch contains three complete datasets in compressed, independently
verifiable archives. Each archive is split into files smaller than GitHub's
100 MB regular-Git file limit. Git LFS is not required to clone this branch.

| Dataset | Contents |
| --- | --- |
| `openai-mrcr` | Original [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr) Parquet files, README and download manifest. The upstream README declares an MIT license. |
| `glm53-length-sweep-v1` | Random-token, exact-length GLM-5.3 performance prompts. |
| `openai-mrcr-length-sweep-v1` | MRCR-derived, exact-length performance prompts. The cropped and padded prompts **must not** be used for answer-accuracy scoring. |

The two performance datasets each have 67 input lengths and 32 unique JSONL
prompts per length (2,144 records). Anchor lengths are 2,048; 8,192; 32,768;
65,536; 131,072; 524,288; and 1,000,000 input tokens, with ten sampled
lengths between each adjacent pair. `case_plan.csv` lists the 1,072 length ×
batch-size 1–16 combinations. Every record requests 16 output tokens.

The archive files live under `archives/`. To verify their SHA-256 checksums and
extract all datasets, install `zstd` and run:

```bash
python3 restore.py --output-dir ./data
```

Use `--dataset openai-mrcr` (or either other dataset name) to extract one
dataset. Use `--verify-only` to check all archive parts without extraction.
`archive_manifest.json` lists each part's size and SHA-256. After extraction,
each performance dataset also includes its own `manifest.json`,
`uniqueness_check.json`, generator, runner, and README. The original MRCR
`DOWNLOAD_INFO.json` records provenance and file hashes.

The generators verify input lengths with the deployed GLM-5.3 tokenizer and
chat template using `enable_thinking=false`. Generator and runner paths refer
to the original deployment and need adjustment on other machines. Capacity
measurements have not been run on this dataset branch.
