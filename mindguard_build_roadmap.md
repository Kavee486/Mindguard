# Building MindGuard From Scratch — A Beginner's Roadmap

This is a full rebuild plan for the MindGuard project (mental-health text classifier + empathetic
response generator + web app), written for someone starting with no ML background but a
decent personal GPU. Realistic total timeline: **3–4 months part-time**, going stage by stage.
Don't skip stages — each one teaches skills the next one needs.

Before anything: run `nvidia-smi` in a terminal to see your GPU's VRAM. This number determines
what you can train locally vs. what needs a cloud GPU (free options: Google Colab, Kaggle).
As a rough guide: 6–8GB handles BERT/RoBERTa fine-tuning fine; 12GB+ is comfortable for
everything including the LLM fine-tuning stage; under 8GB, you'll want Colab for that last stage.

---

## Phase 0 — Foundations (skip anything you already know)

You need enough Python to read and modify scripts, not to write everything from zero.

1. **Python basics** (variables, functions, loops, lists/dicts, reading files) — official Python
   tutorial or freeCodeCamp's Python course. ~1–2 weeks.
2. **Command line basics** — navigating folders (`cd`, `ls`), running a script (`python file.py`).
3. **pandas + numpy** — the libraries you'll use to load and manipulate the Reddit dataset as
   tables. The pandas "10 minutes to pandas" doc is enough to start.
4. **Core ML concepts** — what train/validation/test splits are, what accuracy and F1 score
   mean, what overfitting is, what a confusion matrix shows. You don't need the math yet, just
   the vocabulary — Google's "Machine Learning Crash Course" (free) covers this in a few hours.
5. **Git basics** (optional but recommended) — `git init`, `git add`, `git commit`, and putting
   your project on GitHub so you don't lose work.

---

## Phase 1 — Environment Setup

1. Install **Python 3.10** and create a virtual environment (`conda create -n mindguard
   python=3.10` or `python -m venv mindguard`).
2. Install **Jupyter** (`pip install jupyterlab`) — you'll write most of this project as notebooks,
   same as the original.
3. Install **PyTorch with GPU support** — go to pytorch.org, pick your CUDA version, and use
   their install command. Verify it worked: `python -c "import torch; print(torch.cuda.is_available())"`
   should print `True`.
4. Install the core libraries:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install transformers datasets accelerate
   pip install gensim optuna shap
   ```
5. Install **Ollama** (ollama.com) — this is how you'll run a local LLM later for prompt
   engineering and as the base for the response generator, without paying for API calls.

---

## Phase 2 — Get and Explore the Data

1. Create a free Kaggle account and download:
   - `solomonk/reddit_mental_health_posts` — the classification dataset (five subreddits:
     ADHD, OCD, Asperger's, depression, PTSD).
   - `Amod/mental_health_counseling_conversations` — context→response pairs for the
     empathetic-reply stage later.
2. Load the CSV into pandas, look at it: how many rows, what columns, how many per
   category, how many empty/deleted posts.
3. Clean it: lower-case text, strip URLs and Reddit markup ([deleted]/[removed]), drop
   rows with too little text. Keep only posts with real body text (the original project found
   ~42% of posts had empty bodies and filtered them out).
4. Plot the class distribution (bar chart) — you'll notice it's imbalanced (some categories have
   nearly 2x more posts than others). Note this now; it matters later when you interpret results.

**Milestone:** a clean pandas DataFrame with text + label columns, saved to disk.

---

## Phase 3 — Baseline Classifier (your first working model)

Do this before touching any transformer — it's fast, teaches the sklearn workflow, and gives
you a number to beat.

1. Split data 80/10/10 (train/validation/test) using `sklearn.model_selection.train_test_split`.
2. Vectorize text with `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`.
3. Train a `LinearSVC` on the vectors.
4. Evaluate: accuracy, weighted F1, confusion matrix (`sklearn.metrics`).

Expect somewhere around 85–87% accuracy — a strong baseline because the five categories
have genuinely distinct vocabulary. **This is your reference number for everything after.**

*Optional side-quest:* try Word2Vec/FastText/GloVe embeddings (via `gensim`) with the same
SVM classifier, just to see how they compare to TF-IDF. In the original project, plain TF-IDF
actually won — a useful lesson that fancier isn't always better.

---

## Phase 4 — Transformer Fine-Tuning (BERT / RoBERTa)

This is the core "advanced AI" stage and where a GPU matters.

1. Read HuggingFace's text classification tutorial (huggingface.co/docs) — it walks through
   exactly this workflow.
2. Load `bert-base-uncased` via `AutoModelForSequenceClassification` with `num_labels=5`.
3. Tokenize your text with the model's tokenizer, max length 256 tokens (long enough to avoid
   cutting off most Reddit posts).
4. Fine-tune using HuggingFace's `Trainer` API — this handles the training loop for you. Use
   `EarlyStoppingCallback` so it stops once validation performance plateaus.
5. Repeat with `roberta-base`.
6. Compare both to your Phase 3 baseline. Expect roughly a 2–4 point F1 improvement.
7. *Optional:* use `optuna` to search learning rate / batch size / weight decay automatically
   instead of guessing.

**Gotcha to know upfront:** training transformers is VRAM-hungry. If you hit "out of memory,"
reduce batch size first (try 8 or 16) before anything else.

**Milestone:** two fine-tuned models saved locally, each with test-set accuracy/F1 logged.

---

## Phase 5 — Few-Shot / Transfer Learning

1. Reload your fine-tuned BERT, but drop its classification head — use it purely as a feature
   extractor (`AutoModel`, not `AutoModelForSequenceClassification`).
2. Extract the `[CLS]` embedding (a 768-number vector) for every post.
3. Train a plain `LogisticRegression` on top of these embeddings using only 5, 10, 20, 50, 100,
   200 examples per class, and plot accuracy vs. number of examples.
4. You'll likely see it saturate almost immediately — a real and interesting finding about how
   separable the fine-tuned features already are. Understand *why* (the encoder already saw
   this data during fine-tuning) rather than just reporting the number.

---

## Phase 6 — Prompt Engineering with a Local LLM

1. Pull a model through Ollama: `ollama pull qwen2.5:14b` (or a smaller variant like `7b` if
   VRAM is tight).
2. Write three prompt styles in a Python script calling Ollama's REST API:
   - zero-shot (just ask it to classify),
   - few-shot (give it 5–6 labeled examples in the prompt first),
   - chain-of-thought (ask it to reason about themes before answering).
3. Run all three against a few hundred held-out posts, compare accuracy. Use
   `temperature=0` for reproducibility.

**Milestone:** a comparison table showing prompting accuracy vs. your trained classifiers —
expect prompting to trail the fine-tuned models, which is the expected, honest result.

---

## Phase 7 — Generative Augmentation (optional but instructive)

1. Use your local LLM to paraphrase posts from the under-represented classes.
2. **Critical rule to avoid a very common mistake:** split your data into train/test *first*, and
   only paraphrase the training portion. Never let paraphrases leak into your test set — that
   would inflate your results dishonestly.
3. Filter paraphrases by cosine similarity to the original (keep ones that are similar but not
   identical, e.g. 0.3–0.9 range) so you're not just duplicating data.
4. Retrain your baseline on original vs. augmented data, compare. Don't be surprised if it
   doesn't help — that's a real, defensible finding when you already have plenty of data.

---

## Phase 8 — Advanced Evaluation

1. **Cross-validation:** run 5-fold CV on your baseline to check result stability.
2. **Ensembling:** soft-vote your BERT and RoBERTa predictions together (average their
   output probabilities) — usually a small but real accuracy bump.
3. **Statistical significance:** use McNemar's test (`statsmodels` has this) to check whether
   two classifiers are *actually* different or just noise.
4. **Interpretability:** use `shap` to see which words push predictions toward each class;
   visualize BERT's attention weights for a few examples.
5. **Embedding visualization:** run t-SNE (`sklearn.manifold.TSNE`) on your `[CLS]`
   embeddings and color by class — a nice visual sanity check of whether classes cluster.

---

## Phase 9 — Empathetic Response Generation (the heaviest stage)

This is the most GPU-demanding part. Check VRAM before starting.

1. Download `Mistral-7B-Instruct-v0.2` (or a smaller instruct model if VRAM-limited).
2. Install `peft` and `bitsandbytes` for LoRA + 4-bit quantization (`pip install peft
   bitsandbytes`) — this makes fine-tuning a 7B model feasible on a single consumer GPU.
3. Format your counseling-conversation dataset as instruction/response pairs.
4. Set up a LoRA adapter (rank 16, alpha 32, targeting attention projections is a good
   starting config) and fine-tune for a few epochs with gradient checkpointing enabled.
5. Merge the adapter into the base model, convert to GGUF format, and serve it through
   Ollama alongside your other local models.
6. Compare base vs. fine-tuned responses on a handful of fixed test prompts — you're
   checking for tone: warmer, less blunt, always non-diagnostic, always pointing to
   professional help.

**If your GPU can't handle this:** run this one stage on Google Colab (free T4 GPU) or Kaggle
notebooks (free, 30hrs/week GPU quota) instead of locally.

---

## Phase 10 — Build the Product

1. **Backend (FastAPI):** wrap your saved models behind endpoints — `/analyze` (classify +
   generate response), `/interpret` (attention/SHAP), `/compare` (all models side by side),
   `/health`. Install with `pip install fastapi uvicorn`.
2. **Frontend (Next.js):** install Node.js, scaffold with `npx create-next-app`, build a few
   screens that call your FastAPI endpoints: an analysis view, an interpretability view, a model
   comparison view.
3. Add a **persistent disclaimer and crisis-helpline message** on every screen — this is a
   safety requirement, not a nice-to-have, given the subject matter.

---

## Phase 11 — Write It Up

Once everything runs end to end, document it using the 11-section structure you already have
(Executive Summary through Appendices) — this is much easier once you have real numbers
and screenshots in hand rather than trying to write it alongside the code.

---

## Suggested Pacing (part-time, ~10hrs/week)

| Weeks | Focus |
|---|---|
| 1–2 | Phase 0–1: Python/ML basics, environment setup |
| 3 | Phase 2–3: Data + baseline classifier |
| 4–6 | Phase 4: Transformer fine-tuning |
| 7 | Phase 5–6: Few-shot + prompting |
| 8 | Phase 7–8: Augmentation + advanced evaluation |
| 9–11 | Phase 9: LLM response fine-tuning |
| 12–14 | Phase 10: FastAPI + Next.js product |
| 15–16 | Phase 11: Write-up, polish, appendix |

Start Phase 3 as early as possible even if Phase 0 feels incomplete — having a working, if
crude, classifier in week 3 will teach you more than reading tutorials in isolation.
