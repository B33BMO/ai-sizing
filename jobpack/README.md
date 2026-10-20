# memlm job pack

Runs our training/eval jobs on a multi-GPU Windows machine with **Docker Desktop (WSL2 backend)**. No WSL shell
or Python setup needed on the host.

- `memlm-kb`: read-only Spero tool service (Spero's real `searchDocs`/`exploreGraph`) over a **search-only** DB:
  docs, FTS/vector indexes and the knowledge graph. No conversations, memories, users, sessions, secrets or runs
  (`kb/make_search_db.cjs` drops them; output is verified identical to the full DB for search).
- `memlm-train`: torch 2.11 cu128 + transformers/peft/bitsandbytes + our code + the training data bundle.
  No API keys, tokens or certificates are in either image.

## On the GPU machine (once)
1. NVIDIA Control Panel → Manage 3D settings → **CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback**
   (otherwise a VRAM overflow silently runs 10–50× slower instead of failing).
2. Give Docker enough RAM (Docker Desktop → Settings → Resources, or `memory=` in `%UserProfile%\.wslconfig`).
3. Disable sleep / automatic restarts for the window.
4. `docker load -i memlm-jobpack.tar.gz`

## Run
```powershell
powershell -ExecutionPolicy Bypass -File run-all.ps1 -Jobs jobs\tonight.jobs
powershell -File status.ps1            # container states + last log lines
```
One container per GPU (`gpu | name | command` lines in the jobs file). Results and logs go to `.\out\<job>\`;
models cache in `.\hf-cache\` (downloaded on first use).

## Build (on srv)
```sh
jobpack/build.sh
docker save memlm-kb:latest memlm-train:latest | gzip > memlm-jobpack.tar.gz
```
