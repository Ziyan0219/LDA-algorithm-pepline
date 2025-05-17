# Project README

This project contains two Python scripts:

1. **LDA——Algorithm.py**  — A pipeline for language detection, translation, and LDA topic modeling.
2. **Youtube_crawler.py** — A YouTube-comment crawler that reads video titles and URLs from a CSV file.

---

## 1. Prerequisites

* Python 3.8 or newer
* A virtual environment with dependencies installed (see instructions below)
* For `Youtube_crawler.py`: a valid YouTube Data API key

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # or Windows equivalent
pip install --upgrade pip
pip install pandas torch tqdm fasttext transformers \
            scikit-learn matplotlib google-api-python-client
```

---

## 2. Script: LDA_Algorithm.py

At the top of `LDA_Algorithm.py`, you will find configurable constants:

```python
INPUT_CSV        = 'merged_comments.csv'        # Path to your input CSV
TRANSLATED_CSV   = 'merged_comments_translated.csv'
PERPLEXITY_CSV   = 'perplexity_summary.csv'
TOPIC_ASSIGN_CSV = 'comments_topic_assignment.csv'
PLOT_FILE        = 'perplexity_plot.png'
CACHE_FILE       = 'translation_cache.json'
```

> **Action:** Change `INPUT_CSV` to match the filename or path of your own CSV file containing comments. You can also rename the other output paths if desired.

---

## 3. Script: Youtube_crawler.py

1. **API Key**

   ```python
   API_KEY = "YOUR_API_KEY_HERE"
   ```

   > **Action:** Replace `YOUR_API_KEY_HERE` with your own YouTube Data API key.

2. **Input CSV**

   ```python
   input_csv = "combined_short_sample.csv"
   ```

   > **Action:** Change `combined_short_sample.csv` to the path/name of your CSV file.

3. **Column detection**

   ```python
   # find the column whose name contains “title”
   title_field = next(f for f in reader.fieldnames if "title" in f.lower())

   # find the column whose name contains “video_id” or "video id"
   link_field  = next(
       f for f in reader.fieldnames
       if "video_id" in f.lower() or "video id" in f.lower()
   )
   ```

   > **Action:** If your CSV uses different column names for the video title or URL/ID, update these two lines. For example, replace `"title"` or `"video_id"` with the actual keywords in your header.

4. **Output folder** (optional)

   ```python
   output_folder = "output_comments"
   ```

   > **Action:** Change this folder name if you want CSV outputs saved elsewhere.

---

## 4. Running the scripts

### LDA pipeline

```bash
python LDA_Algorithm.py
```

This will:

* Read your `INPUT_CSV`
* Translate texts and save to `TRANSLATED_CSV`
* Compute perplexity grid and save to `PERPLEXITY_CSV`
* Plot and save `perplexity_plot.png`
* Assign topics and save to `TOPIC_ASSIGN_CSV`

### YouTube crawler

```bash
python Youtube_crawler.py
```

This will:

* Open your `input_csv`
* Extract video IDs
* Fetch top-level comments
* Save each video’s comments in `output_comments/*.csv`

---

## 5. Support

If you encounter any errors or need help customizing column names, please adjust the variables noted above or open an issue with details about your CSV headers and file structure.

Happy coding! 🎉
