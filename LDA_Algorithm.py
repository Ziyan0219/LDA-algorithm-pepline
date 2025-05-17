import os, json, re, math, sys, urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# ========== CONFIG ==========
INPUT_CSV        = 'merged_comments.csv'
TRANSLATED_CSV   = 'merged_comments_translated.csv'
PERPLEXITY_CSV   = 'perplexity_summary.csv'
TOPIC_ASSIGN_CSV = 'comments_topic_assignment.csv'
PLOT_FILE        = 'perplexity_plot.png'
CACHE_FILE       = 'translation_cache.json'

# mapping from language code to translation model
LANG_MODEL_MAP = {
    'zh': 'Helsinki-NLP/opus-mt-zh-en',
    'es': 'Helsinki-NLP/opus-mt-es-en',
    'fr': 'Helsinki-NLP/opus-mt-fr-en',
    'de': 'Helsinki-NLP/opus-mt-de-en',
    'pt': 'Helsinki-NLP/opus-mt-pt-en',
    'it': 'Helsinki-NLP/opus-mt-it-en',
    'ru': 'Helsinki-NLP/opus-mt-ru-en',
}
FALLBACK_MODEL = 'facebook/nllb-200-distilled-600M'

# set number of threads for torch
torch.set_num_threads(max(2, os.cpu_count() - 1))
DEVICE = -1  # use CPU
BATCH_SIZE = 24
MAX_WORKERS = min(8, os.cpu_count())
PROGRESS_STEP = 500

# LDA parameters
TOPIC_COUNTS = [9, 12, 15, 18, 21]
TOP_N_WORDS  = 10

# --------------------------------------------------
# 1. Ensure fastText language identifier
# --------------------------------------------------
MODEL_FILE = 'lid.176.ftz'
URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
if not Path(MODEL_FILE).exists() or Path(MODEL_FILE).stat().st_size < 800_000:
    print('Downloading lid.176.ftz…')
    urllib.request.urlretrieve(URL, MODEL_FILE)
FAST_MODEL = fasttext.load_model(MODEL_FILE)

re_zh = re.compile(r'[\u4e00-\u9fff]')
re_ja = re.compile(r'[\u3040-\u30ff]')
re_ko = re.compile(r'[\uac00-\ud7af]')
re_ru = re.compile(r'[А-Яа-я]')

def quick_lang(text):
    if re_zh.search(text): return 'zh'
    if re_ja.search(text): return 'ja'
    if re_ko.search(text): return 'ko'
    if re_ru.search(text): return 'ru'
    return ''

def fasttext_batch(texts):
    labels = FAST_MODEL.predict(texts, k=1)[0]
    return [lbl[0].replace('__label__', '') for lbl in labels]

# caching helpers

def load_cache():
    return json.load(open(CACHE_FILE, 'r', encoding='utf-8')) if os.path.exists(CACHE_FILE) else {}

def save_cache(c):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(c, f, ensure_ascii=False, indent=2)

# translator cache
_translators = {}

def get_translator(model):
    if model not in _translators:
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model)
        _translators[model] = pipeline('translation', model=mdl, tokenizer=tok, device=DEVICE, batch_size=BATCH_SIZE)
    return _translators[model]

# --------------------------------------------------
# 2. Main translation function (parallel per language)
# --------------------------------------------------

def translate_all(df):
    cache = load_cache()
    translated = [''] * len(df)
    langs = [''] * len(df)
    unknown_idx = []

    # Phase A: language detection
    for i, txt in tqdm(list(enumerate(df['text'].astype(str))), desc='lang-detect'):
        if txt in cache:
            translated[i] = cache[txt]; langs[i] = 'cached'; continue
        if not txt.strip() or len(txt.split()) <= 3:
            langs[i] = 'en'; translated[i] = txt; cache[txt] = txt; continue
        lg = quick_lang(txt)
        if lg:
            langs[i] = lg
        else:
            unknown_idx.append(i)
    if unknown_idx:
        preds = fasttext_batch([df.at[i, 'text'] for i in unknown_idx])
        for i, lang in zip(unknown_idx, preds): langs[i] = lang

    print('Language distribution top20:', Counter(langs).most_common(20))

    # group and translate in parallel
    groups = defaultdict(list)
    for idx, lg in enumerate(langs):
        if lg not in ('en', 'cached'): groups[lg].append(idx)

    def work(lang, idxs):
        model = LANG_MODEL_MAP.get(lang, FALLBACK_MODEL)
        trans = get_translator(model)
        total = len(idxs); done = 0
        for s in range(0, total, BATCH_SIZE):
            batch_idx = idxs[s:s+BATCH_SIZE]
            batch_txt = [df.at[i, 'text'] for i in batch_idx]
            out = trans(batch_txt, **({'src_lang': lang, 'tgt_lang': 'eng_Latn'} if model==FALLBACK_MODEL else {}))
            for i, res in zip(batch_idx, out):
                translated[i] = res['translation_text']; cache[df.at[i,'text']] = translated[i]
            done += len(batch_idx)
            if done % PROGRESS_STEP == 0 or done == total:
                print(f'[lang {lang}] {done}/{total}')
        return lang

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(work, lg, idxs): lg for lg, idxs in groups.items()}
        for f in as_completed(futures):
            print(f'Finished language {futures[f]}')

    save_cache(cache)
    df['text_en'] = translated
    return df

# --------------------------------------------------
# 3. LDA and topic utilities
# --------------------------------------------------

def lda_perplexity(texts):
    vec = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    res = []
    for k in TOPIC_COUNTS:
        lda = LatentDirichletAllocation(n_components=k, max_iter=10, learning_method='batch', random_state=42)
        lda.fit(dtm)
        perp = lda.perplexity(dtm)
        topics = [' '.join(vocab[idx] for idx in comp.argsort()[:-TOP_N_WORDS-1:-1]) for comp in lda.components_]
        res.append((k, perp, ' | '.join(topics)))
        print(f'k={k} perplexity={perp:.1f}')
    return res, vec, dtm

def plot_perp(results):
    ks, ps, _ = zip(*results)
    plt.figure()
    plt.plot(ks, ps, marker='o')
    plt.xlabel('k')
    plt.ylabel('perplexity')
    plt.grid(True)
    plt.title('LDA perplexity')
    plt.savefig(PLOT_FILE)

def assign_topics(df, vec, dtm, k_best):
    lda = LatentDirichletAllocation(n_components=k_best, max_iter=10, learning_method='batch', random_state=42)
    lda.fit(dtm)
    df['assigned_topic'] = lda.transform(vec.transform(df['text_en'])).argmax(axis=1)
    df.to_csv(TOPIC_ASSIGN_CSV, index=False)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    df_raw = pd.read_csv(INPUT_CSV, encoding='utf-8')
    print('\n=== Stage 1: Translate ===')
    df_tr = translate_all(df_raw.copy())
    df_tr.to_csv(TRANSLATED_CSV, index=False)

    print('\n=== Stage 2: LDA grid ===')
    res, vec, dtm = lda_perplexity(df_tr['text_en'])
    pd.DataFrame(res, columns=['num_topics','perplexity','topics']).to_csv(PERPLEXITY_CSV, index=False)

    print('\n=== Stage 3: Plot ===')
    plot_perp(res)

    k_best = 15
    print(f'\n=== Stage 4: Assign topics (k={k_best}) ===')
    assign_topics(df_tr, vec, dtm, k_best)
    print('Done.')