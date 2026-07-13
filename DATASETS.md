# Datasets

Both datasets used by this project are public and hosted on HuggingFace -
**no login or API key required**. `src/data_prep.py` downloads them for you
automatically the first time you run it. This file is just so you know what
they are and can inspect them yourself if you want.

## 1. Classification dataset (Reddit posts)

- **Identifier:** `solomonk/reddit_mental_health_posts`
- **Where:** https://huggingface.co/datasets/solomonk/reddit_mental_health_posts
- **Size:** ~151,000 posts
- **Fields:** `title`, `body`, `subreddit` (the label), plus metadata
  (`author`, `score`, `created_utc`, etc.)
- **Classes:** ADHD, OCD, Asperger's, depression, PTSD (one subreddit each)
- **How it's loaded:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("solomonk/reddit_mental_health_posts")
  ```

Note: an earlier version of the project report referred to this as a Kaggle
dataset, but it's only available on HuggingFace under this name - that's
what `src/data_prep.py` uses.

## 2. Counseling-conversations dataset (for the response generator)

- **Identifier:** `Amod/mental_health_counseling_conversations`
- **Where:** https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- **Size:** ~3,512 context/response pairs, sourced from counselchat.com
- **Fields:** `Context` (a person's question/message), `Response` (a
  counselor's reply)
- **License:** RAIL-D (free for academic/research use)
- **How it's loaded:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("Amod/mental_health_counseling_conversations")
  ```

## Fallback alternatives

If either dataset is ever unavailable, these are real alternatives you could
swap in with light code changes to `src/data_prep.py`:

- Kaggle: "Reddit Mental Health Dataset (RMHD)" -
  kaggle.com/datasets/entenam/reddit-mental-health-dataset (requires a free
  Kaggle account + API token, unlike the HuggingFace datasets above)
- HuggingFace: `hugginglearners/reddit-depression-cleaned` - smaller,
  pre-cleaned, no login needed
