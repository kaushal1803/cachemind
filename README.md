# CacheMind research artifact

Research code and analysis notebooks for trace-grounded, natural-language
reasoning about cache replacement.

- [Installation, environment and required inputs](INSTALL.md)
- [Citation](#citation)
- [Archiving releases](ARCHIVING.md)

## Repository contents

| Path | Purpose |
| --- | --- |
| `code_files/RAG.yaml` | Exported Linux Conda environment |
| `code_files/make_rag_dataset.ipynb` | Prepare RAG input from traces and program information |
| `code_files/rag_source.py` | Retrieval and trace-processing helpers |
| `code_files/RAG_application*.ipynb` | Local, hosted Hugging Face and OpenAI notebook variants |
| `code_files/Finetune_OpenAI.ipynb` | Fine-tuning experiment notebook |
| `analysis/` | Evaluation notebooks |
| `Categorized Benchmark Questions.pdf` | Public benchmark questions |

The checked-in notebooks refer to data files and local paths that are not included
in this repository. Installation alone does not reproduce the paper's results;
see the input requirements in INSTALL.md. No software license is currently
provided in the repository; maintainers should specify reuse terms before a
software archive is published.

## Citation

Please cite the associated paper when using this artifact:

**CacheMind: From Miss Rates to Why — Natural-Language, Trace-Grounded Reasoning for Cache Replacement** (2026).  
Paper DOI: https://doi.org/10.1145/3779212.3790136

```bibtex
@inproceedings{cachemind2026,
  author = {Mhapsekar, Kaushal and Ghanbari, Azam and Aslrousta, Bita and Mirbagher-Ajorpaz, Samira},
  title = {{CacheMind: From Miss Rates to Why — Natural-Language, Trace-Grounded Reasoning for Cache Replacement}},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  year = {2026},
  doi = {10.1145/3779212.3790136}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
The paper DOI identifies the publication; it is not a software-release DOI.
For reproducibility, also record the Git commit and, when available, the archived
release DOI. See [ARCHIVING.md](ARCHIVING.md) for the release procedure.
