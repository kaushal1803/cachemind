# Installation and required inputs

## Environment

`code_files/RAG.yaml` is an exported Linux environment with Python 3.8.20,
PyTorch/CUDA and GPU FAISS. It contains platform-specific build pins and an
original machine's `prefix`; it is not a portable CPU-only lockfile.

With Conda installed, start from the checked-in environment:

```sh
git clone https://github.com/kaushal1803/cachemind.git
cd cachemind
conda env create --name cachemind --file code_files/RAG.yaml
conda activate cachemind
python -m pip check
```

The explicit environment name overrides the original export's location. A
successful installation has not been verified for this documentation change.
The export contains dependency versions that need maintainer review (including
PyTorch 2.4.1 alongside torchvision 0.20.0 and typing_extensions 4.5.0). If the
solver or `pip check` reports conflicts, preserve the error and obtain a corrected,
validated environment rather than treating a partial installation as complete.

A Jupyter front end is needed to use the notebooks. `ipykernel` is included in
the export; register it for an existing Jupyter installation with:

```sh
python -m ipykernel install --user --name cachemind --display-name CacheMind
```

Open the relevant notebook from `code_files/` and select that kernel. The
export does not list all optional notebook imports: particular cells also use
OpenAI, python-docx, PEFT, BERTScore, MoverScore, NLTK, LlamaIndex or Evidently.
The repository does not specify validated versions for those optional paths;
maintainers should provide them before claiming full reproduction support.

## Inputs needed before executing notebooks

The current checkout includes code and benchmark questions, but no `data/`
directory, raw eviction traces or workload-source directory.

| Notebook | Required input / adjustment |
| --- | --- |
| `make_rag_dataset.ipynb` | Raw eviction traces, workload source and disassembly files; update local paths |
| `RAG_application.ipynb` | `processed_data.pkl` relative to the working directory; local model resources |
| `RAG_application_OpenAI.ipynb` and `RAG_application_HF.ipynb` | `../data/processed_data.pkl`; experiment-specific CSV/question files and model configuration |
| `Finetune_OpenAI.ipynb` | Referenced training/example CSV and JSONL files |
| `analysis/*.ipynb` | Referenced evaluation CSV files |

Obtain permitted inputs from the maintainers or generate compatible inputs using
the dataset notebook. Do not assume the benchmark PDF replaces the machine-readable
question files. Resolve hard-coded paths before running cells. Load pickle files
only from a trusted source.

Hosted model paths require the user's own provider credentials and can incur
provider charges. Configure credentials locally; do not commit them. Local
model variants require hardware resources appropriate to the chosen model;
no verified minimum memory requirement is provided in this repository.

## Execution order and validation

1. Resolve and validate the environment for the selected notebook path.
2. Supply traces/program information and prepare the processed RAG dataset.
3. Configure one application notebook and run its cells interactively.
4. Save the generated outputs and use the matching analysis notebook.

Record the commit (`git rev-parse HEAD`), environment, model/version, dataset
version and notebook variant with results. These instructions describe the
checked-in files and outstanding prerequisites; full paper reproduction has
not been tested by this documentation update.
