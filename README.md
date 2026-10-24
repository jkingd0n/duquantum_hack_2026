# DuQuantum / Pasqal hackathon

## Setup

1. Install and start Docker, then install VS Code's **Dev Containers** extension.
2. Open this repository in VS Code and run **Dev Containers: Reopen in Container**.
3. Wait for setup to finish. Python 3.12 and the locked dependencies are installed
   automatically into `.venv`.
4. Open a notebook, select `.venv/bin/python` as the kernel, and run the cells
   from top to bottom.

The first build requires internet access. The pre-reading notebooks run locally
without credentials.

## Repository structure

| Path | Contents |
| --- | --- |
| [prereading/](prereading/) | Participant PDF and three local tutorial notebooks: Pulser programming, registers and Rydberg interactions, and QUBO Solver. |
| [pasqal_first_job.ipynb](pasqal_first_job.ipynb) | First cloud-emulator job; requires a Pasqal account and submits jobs when run. |
| [.devcontainer/](.devcontainer/) | Container image and VS Code setup. |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | Python dependencies and locked versions. |
| [.env_template](.env_template) | Cloud credential template; copy to `.env` for the cloud notebook. Never commit credentials. |
