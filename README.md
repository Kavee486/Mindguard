# MindGuard - Build Guide

An AI system that classifies text into five mental-health-related categories
(ADHD, OCD, Asperger's, depression, PTSD) and generates a supportive,
non-diagnostic response. This repo is a from-scratch, runnable rebuild of
the MindGuard project, sized for a personal machine with one GPU.

**This is a research/teaching demo, not a medical device. It must never be
used for real diagnosis or triage.**

---

## 0. What you need to do yourself (checklist)

I've written all the code. These are the parts that have to happen on your
machine, in order:

1. [ ] Install the prerequisites (Python 3.10, Node.js 20, Git, and
   optionally Docker + Ollama) - see Section 1.
2. [ ] Create a virtual environment and `pip install -r requirements.txt`,
   plus the correct `torch` build for your GPU - see Section 1.
3. [ ] Run `python src/data_prep.py` to download and clean the datasets -
   see Section 2. (Needs an internet connection; no login required.)
4. [ ] Run the training scripts in order (Section 3) - this is the part
   that takes real time (minutes to hours depending on your GPU) and you
   should run it yourself so you actually see the process working.
5. [ ] Install Ollama and pull a model, for the prompting stage and for
   response generation - see Section 4.
6. [ ] Start the backend and frontend (either locally or via Docker) and
   try the app in your browser - see Section 5 and 6.

I cannot run steps 3-6 for you - they need your GPU, your downloaded data,
and take real wall-clock time, so they have to happen on your machine. Open
this whole `MindGuard/` folder in VS Code and work through the sections
below in order.

---

## 1. Prerequisites & environment setup

Install:
- **Python 3.10** - https://www.python.org/downloads/
- **Node.js 20+** - https://nodejs.org (for the frontend)
- **Git** (optional but recommended)
- **Docker Desktop** (optional, only needed if you want the Docker path) -
  https://www.docker.com/products/docker-desktop/
- **Ollama** (needed for prompting + response generation) -
  https://ollama.com

Then, in a terminal inside this folder:

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install everything except torch
pip install -r requirements.txt

# Install torch matching YOUR GPU. Check your CUDA version first:
nvidia-smi
# Then go to https://pytorch.org/get-started/locally/, pick your OS + CUDA
# version, and run the command it gives you, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is visible to PyTorch:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` prints `False`, stop and fix that before
continuing to training - everything will still run on CPU but transformer
fine-tuning (Section 3) will be very slow.

---

## 2. Get the data

```bash
python src/data_prep.py
```

This downloads both datasets from HuggingFace (no login needed - see
`DATASETS.md` for details), cleans the text, filters out empty/deleted
posts, and writes:
- `data/processed/train.csv`, `val.csv`, `test.csv` (classification data)
- `data/processed/counseling.csv` (for response generation, Section 3.6)
- `data/processed/class_distribution.png`

This step needs an internet connection and will take a few minutes
depending on your connection (the raw dataset is a few hundred MB).

---

## 3. Train the models (run these in order)

Each script prints its own metrics as it runs and saves results under
`models/`. You don't have to run every stage before using the app - the
backend (Section 5) gracefully skips any model you haven't trained yet -
but running them in order is the intended learning path.

### 3.1 Baseline classifier (fast, do this first)
```bash
python src/baseline_tfidf.py
```
Takes under a minute. Expect ~85-87% accuracy.

### 3.2 Transformer fine-tuning (the main GPU stage)
```bash
python src/train_transformer.py --model bert-base-uncased --epochs 3
python src/train_transformer.py --model roberta-base --epochs 3
```
Each takes anywhere from ~20 minutes to a couple of hours depending on your
GPU. If you get "CUDA out of memory", add `--batch_size 8` (or lower).

### 3.3 Few-shot / transfer learning
```bash
python src/few_shot.py --model_dir models/bert-base-uncased/final
```

### 3.4 Prompt engineering (needs Ollama running - see Section 4)
```bash
python src/prompting.py --n_per_class 50
```

### 3.5 Generative augmentation (needs Ollama running)
```bash
python src/augmentation.py --target_classes aspergers ptsd --n_paraphrases 300
```

### 3.6 Advanced evaluation (needs 3.2 done for both models)
```bash
python src/evaluate.py
```
Produces the confusion matrix, ROC curves, t-SNE plot, and McNemar's
significance test under `models/evaluation/`.

### 3.7 Response generation (heaviest stage - see VRAM notes in the file)
```bash
python src/train_response_lora.py --epochs 3
python src/merge_lora.py --base_model microsoft/Phi-3-mini-4k-instruct \
    --adapter_dir models/response_lora/adapter \
    --out_dir models/response_lora/merged
```
Then follow "Serving the response model" in Section 4 to make it available
to the backend through Ollama.

---

## 4. Ollama (local LLM) setup

Used for: prompt-engineering (3.4), augmentation (3.5), and generating
empathetic responses in the running app.

```bash
# after installing from ollama.com, pull a general-purpose model:
ollama pull qwen2.5:7b
# (use qwen2.5:14b instead if you have 16GB+ VRAM, to match the report more closely)

# Ollama runs a local server automatically on http://localhost:11434
# Test it:
ollama run qwen2.5:7b "Say hello"
```

### Serving your fine-tuned response model

After Section 3.7 has produced `models/response_lora/merged/`, convert it to
GGUF (the format Ollama uses) using llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ../models/response_lora/merged \
    --outfile ../models/response_lora/mindguard-responder.gguf
```

Then create an Ollama model from it:

```bash
cd ../models/response_lora
cat > Modelfile << 'MODELFILE'
FROM ./mindguard-responder.gguf
PARAMETER temperature 0.7
MODELFILE

ollama create mindguard-responder -f Modelfile
```

The backend (`backend/app/llm.py`) already looks for a model named
`mindguard-responder` first and falls back to `qwen2.5:7b` if it isn't
found, so no code changes are needed - just run the steps above.

---

## 5. Run the backend (FastAPI)

```bash
# from the project root, with your venv activated
uvicorn backend.app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API docs (FastAPI
generates this automatically). Try `/health` first to confirm which models
loaded successfully.

---

## 6. Run the frontend (Next.js)

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Visit http://localhost:3000. You'll see three screens: Analyze, Compare
Models, and Interpretability.

---

## 7. Running everything with Docker instead

If you'd rather not install Python/Node locally, use Docker Compose. Train
your models first though (Section 3) - Docker here serves the finished app,
it doesn't replace the training step, since training benefits from you
watching it run and iterating on your own machine.

```bash
docker compose up --build
```

This starts three containers: `ollama` (port 11434), `backend` (port 8000),
`frontend` (port 3000). The backend and data folders are mounted from your
project directory, so models you trained in Section 3 are automatically
picked up - you don't need to retrain inside Docker.

After the containers are up, pull a model into the Ollama container once:
```bash
docker compose exec ollama ollama pull qwen2.5:7b
```

### GPU + Docker

By default the Docker setup runs on CPU (simplest, works everywhere). To
give containers GPU access:

1. Install the **NVIDIA Container Toolkit** on your host (Linux; WSL2 on
   Windows also supported) - see
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
2. Uncomment the `deploy: resources: reservations: devices:` block for the
   relevant service in `docker-compose.yml`.
3. In `backend/Dockerfile`, switch the base image to a CUDA-enabled one
   (see the comment at the top of that file) and install the matching
   `torch` build instead of the CPU one.

---

## Project layout

```
MindGuard/
├── data/               downloaded + cleaned datasets (gitignored)
├── models/             trained model artifacts + metrics (gitignored)
├── src/                the ML pipeline, run stage by stage (Section 3)
├── backend/             FastAPI app serving the trained models
├── frontend/             Next.js app (Analyze / Compare / Interpretability)
├── docker-compose.yml
├── requirements.txt
├── DATASETS.md
└── README.md            you are here
```

## Troubleshooting

- **"CUDA out of memory"** - lower `--batch_size` in the relevant script
  first, then try a smaller `--max_len`.
- **`/analyze` returns 503 "No trained models found"** - you haven't run
  Section 3 yet; run at least `src/baseline_tfidf.py`.
- **Ollama calls fail / time out** - make sure `ollama serve` is running
  (it usually auto-starts after install) and that you've pulled a model
  with `ollama pull qwen2.5:7b`.
- **Frontend shows a network error** - check `frontend/.env.local` has the
  right `NEXT_PUBLIC_API_URL` and that the backend is actually running on
  that port.
