"""
=============================================================
  MFS APP REVIEW — FULL TEXT-BASED SENTIMENT PIPELINE
  Apps: bKash · Rocket · Upay · Nagad  (4 apps)
  Method: TF-IDF + Logistic Regression + Aspect-Based Scoring
=============================================================
"""

import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score,
                             precision_score, recall_score)

warnings.filterwarnings('ignore')

# ── Styling ────────────────────────────────────────────────────────────────
COLORS = {
    'bKash':    '#E2136E',
    'Rocket':   '#7B2FBE',
    'Upay':     '#0d9488',
    'Nagad':    '#f97316',
    'Positive': '#10b981',
    'Neutral':  '#f59e0b',
    'Negative': '#f43f5e',
}

plt.rcParams.update({
    'figure.facecolor': '#0f0f18',
    'axes.facecolor':   '#161622',
    'axes.edgecolor':   '#2a2a3e',
    'axes.labelcolor':  '#a0a0c0',
    'xtick.color':      '#6060a0',
    'ytick.color':      '#6060a0',
    'text.color':       '#d0d0f0',
    'grid.color':       '#1e1e32',
    'grid.linestyle':   '--',
    'grid.alpha':        0.5,
    'font.family':      'DejaVu Sans',
    'font.size':         10,
})

APPS = ['bKash', 'Rocket', 'Upay', 'Nagad']

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD & MERGE ALL FOUR DATASETS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 1 — DATA LOADING")
print("═"*60)

files = {
    'bKash':  '/mnt/user-data/uploads/bkash_2020-2026_.csv',
    'Rocket': '/mnt/user-data/uploads/rocket_2020-2026_.csv',
    'Upay':   '/mnt/user-data/uploads/upay_2020-2026_.csv',
    'Nagad':  '/mnt/user-data/uploads/nagad_most_relevent.csv',
}

frames = []
for app, path in files.items():
    df = pd.read_csv(path)
    df['app'] = app
    frames.append(df)
    print(f"  ✓ {app:8s}: {len(df):,} reviews loaded")

raw = pd.concat(frames, ignore_index=True)
print(f"\n  Total combined: {len(raw):,} reviews")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 2 — TEXT PREPROCESSING")
print("═"*60)

raw['at'] = pd.to_datetime(raw['at'])
raw['year']  = raw['at'].dt.year
raw['month'] = raw['at'].dt.to_period('M').astype(str)
raw = raw.dropna(subset=['content']).reset_index(drop=True)

# Rating-based sentiment (baseline)
def rating_sentiment(s):
    if s >= 4: return 'Positive'
    if s == 3: return 'Neutral'
    return 'Negative'

raw['rating_sentiment'] = raw['score'].apply(rating_sentiment)

# Text cleaning — preserves Bengali unicode (U+0980–U+09FF)
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

raw['clean_content'] = raw['content'].apply(clean_text)

# Bengali script detection
raw['has_bengali'] = raw['content'].apply(
    lambda t: bool(re.search(r'[\u0980-\u09FF]', str(t)))
)
bengali_pct = raw['has_bengali'].mean() * 100

print(f"  Reviews containing Bengali script : {bengali_pct:.1f}%")
print(f"  Reviews per app:")
for app in APPS:
    n = (raw['app'] == app).sum()
    print(f"    {app:8s}: {n:,}")

print(f"\n  Class distribution (rating-based):")
for s, c in raw['rating_sentiment'].value_counts().items():
    print(f"    {s:10s}: {c:,} ({c/len(raw)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — TOPIC CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 3 — TOPIC CLASSIFICATION")
print("═"*60)

TOPICS = {
    'Transaction/Payment':  ['send money','transaction','payment','transfer','pay',
                              'cash out','add money','failed','deducted','recharge','bill'],
    'App Performance':      ['crash','slow','lag','freeze','bug','error',
                              'not working','loading','update','install',
                              'time out','timeout','server'],
    'Login/Authentication': ['login','otp','pin','password','fingerprint',
                              'sign in','verification','log in','register','registration'],
    'Charges/Fees':         ['charge','fee','cost','expensive','commission',
                              'service charge','deduct'],
    'Customer Support':     ['customer service','helpline','support','complaint',
                              'agent','response','live chat','hotline'],
    'Security':             ['security','hack','fraud','scam','unauthorized',
                              'stolen','account block'],
    'UI/UX':                ['interface','design','ui','ux','layout','button',
                              'navigation','easy to use','user friendly'],
}

def get_topic(text):
    text = str(text).lower()
    for topic, kws in TOPICS.items():
        if any(kw in text for kw in kws):
            return topic
    return 'Other'

raw['topic'] = raw['content'].apply(get_topic)
print("  Topic distribution (all apps combined):")
for t, c in raw['topic'].value_counts().items():
    print(f"    {t:25s}: {c:,}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — ML MODEL (TF-IDF + LOGISTIC REGRESSION)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 4 — ML MODEL: TF-IDF + LOGISTIC REGRESSION")
print("═"*60)

# Word n-grams (1-2): phrase-level patterns
print("  Fitting word TF-IDF  (1-2 gram, 25k features)...")
tfidf_word = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2),
    max_features=25000, sublinear_tf=True,
    min_df=3, strip_accents='unicode',
)
X_word = tfidf_word.fit_transform(raw['clean_content'])
print(f"    → {X_word.shape[1]:,} features")

# Char n-grams (3-6): subword patterns, handles Bangla & misspellings
print("  Fitting char TF-IDF  (3-6 gram, 30k features)...")
tfidf_char = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 6),
    max_features=30000, sublinear_tf=True, min_df=5,
)
X_char = tfidf_char.fit_transform(raw['clean_content'])
print(f"    → {X_char.shape[1]:,} features")

X = hstack([X_word, X_char])
y = raw['rating_sentiment'].values
print(f"\n  Combined matrix: {X.shape[0]:,} rows × {X.shape[1]:,} features")

# 5-Fold Stratified Cross-Validation
print("\n  Running 5-fold stratified cross-validation...")
clf = LogisticRegression(
    C=1.0, max_iter=1000, solver='saga',
    n_jobs=-1, random_state=42
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)

acc = accuracy_score(y, y_pred_cv)
f1  = f1_score(y, y_pred_cv, average='weighted')

print(f"\n  ✓ Cross-Validation Results:")
print(f"    Overall Accuracy : {acc*100:.2f}%")
print(f"    Weighted F1 Score: {f1:.4f}")
print("\n  Per-Class Report:")
print(classification_report(y, y_pred_cv, digits=3))

raw['text_sentiment'] = y_pred_cv

# Confusion Matrix
cm = confusion_matrix(y, y_pred_cv, labels=['Positive','Neutral','Negative'])
print("  Confusion Matrix (rows=actual, cols=predicted):")
print("  Labels: Positive | Neutral | Negative")
for i, row_lbl in enumerate(['Positive','Neutral','Negative']):
    print(f"    {row_lbl:10s}: {cm[i]}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — AGREEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 5 — AGREEMENT ANALYSIS (Rating vs Text)")
print("═"*60)

raw['agreement'] = raw['rating_sentiment'] == raw['text_sentiment']

def disagree_type(row):
    r, t = row['rating_sentiment'], row['text_sentiment']
    if r == t: return 'Agree'
    pairs = {
        ('Positive','Negative'): 'Rated High, Sounds Negative',
        ('Negative','Positive'): 'Rated Low,  Sounds Positive',
        ('Positive','Neutral'):  'Rated High, Sounds Neutral',
        ('Neutral', 'Negative'): 'Rated Mid,  Sounds Negative',
        ('Neutral', 'Positive'): 'Rated Mid,  Sounds Positive',
        ('Negative','Neutral'):  'Rated Low,  Sounds Neutral',
    }
    return pairs.get((r, t), 'Other')

raw['disagree_type'] = raw.apply(disagree_type, axis=1)

total_agree = raw['agreement'].mean() * 100
print(f"  Overall agreement rate: {total_agree:.1f}%")
print(f"  Disagreement rate     : {100-total_agree:.1f}%\n")

print("  Disagreement types (all apps):")
for dtype, cnt in raw[~raw['agreement']]['disagree_type'].value_counts().items():
    print(f"    {dtype:45s}: {cnt:5,}  ({cnt/len(raw)*100:.1f}%)")

print("\n  Agreement by app:")
for app in APPS:
    sub = raw[raw['app'] == app]
    ag  = sub['agreement'].mean() * 100
    print(f"    {app:8s}: {ag:.1f}% agreement  |  {100-ag:.1f}% disagree")

# Most interesting disagreements
interesting = raw[
    raw['disagree_type'] == 'Rated High, Sounds Negative'
].sort_values('thumbsUpCount', ascending=False)
print(f"\n  Top 5 'Rated High but Sounds Negative' reviews:")
for _, row in interesting.head(5).iterrows():
    print(f"  [{row['app']:6s}] ★{row['score']} | {str(row['content'])[:110]}...")
    print()


# ══════════════════════════════════════════════════════════════════════════
# STAGE 6 — ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 6 — ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
print("═"*60)

absa_rows = []
for app in APPS:
    for topic in TOPICS.keys():
        sub = raw[(raw['app'] == app) & (raw['topic'] == topic)]
        if len(sub) < 10:
            continue
        pos_r = (sub['rating_sentiment'] == 'Positive').mean()
        neg_r = (sub['rating_sentiment'] == 'Negative').mean()
        pos_t = (sub['text_sentiment']   == 'Positive').mean()
        neg_t = (sub['text_sentiment']   == 'Negative').mean()
        absa_rows.append({
            'app': app, 'topic': topic, 'n': len(sub),
            'rating_pos_pct': round(pos_r * 100, 1),
            'rating_neg_pct': round(neg_r * 100, 1),
            'text_pos_pct':   round(pos_t * 100, 1),
            'text_neg_pct':   round(neg_t * 100, 1),
            'pos_gap': round((pos_t - pos_r) * 100, 1),
            'neg_gap': round((neg_t - neg_r) * 100, 1),
        })

absa_df = pd.DataFrame(absa_rows)
print(f"\n  {'App':8s} {'Topic':25s} {'N':>6s} | {'Rate+%':>7s} {'Text+%':>7s} {'Gap':>6s} | {'Rate-%':>7s} {'Text-%':>7s} {'Gap':>6s}")
print("  " + "─"*85)
for _, r in absa_df.sort_values(['app','neg_gap'], ascending=[True,False]).iterrows():
    print(f"  {r['app']:8s} {r['topic']:25s} {int(r['n']):>6,} | "
          f"{r['rating_pos_pct']:>6.1f}% {r['text_pos_pct']:>6.1f}% {r['pos_gap']:>+6.1f}% | "
          f"{r['rating_neg_pct']:>6.1f}% {r['text_neg_pct']:>6.1f}% {r['neg_gap']:>+6.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 7 — TOP DISCRIMINATING FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 7 — TOP DISCRIMINATING WORDS PER SENTIMENT CLASS")
print("═"*60)

clf.fit(X, y)
classes = clf.classes_
feature_names_word = tfidf_word.get_feature_names_out()
n_word = X_word.shape[1]

top_features = {}
for i, cls in enumerate(classes):
    coefs   = clf.coef_[i][:n_word]
    top_idx = np.argsort(coefs)[-20:][::-1]
    top_features[cls] = [(feature_names_word[j], round(coefs[j], 3)) for j in top_idx]

for cls in classes:
    words = [f"{w}({s:+.2f})" for w, s in top_features[cls][:15]]
    print(f"\n  → {cls}:")
    print("    " + "  ·  ".join(words))


# ══════════════════════════════════════════════════════════════════════════
# STAGE 8 — MONTHLY TEXT SENTIMENT TRENDS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 8 — MONTHLY TREND DATA PREPARATION")
print("═"*60)

def make_monthly(df, sentiment_col, pos_col, neg_col):
    m = df.groupby(['app','month', sentiment_col]).size().unstack(fill_value=0).reset_index()
    m['month_dt'] = pd.to_datetime(m['month'])
    m = m.sort_values(['app','month_dt'])
    cols = [c for c in ['Positive','Neutral','Negative'] if c in m.columns]
    m['total'] = m[cols].sum(axis=1)
    if 'Positive' in m.columns:
        m[pos_col] = m['Positive'] / m['total'] * 100
    if 'Negative' in m.columns:
        m[neg_col] = m['Negative'] / m['total'] * 100
    return m

monthly_rating = make_monthly(raw, 'rating_sentiment', 'rating_pos_pct', 'rating_neg_pct')
monthly_text   = make_monthly(raw, 'text_sentiment',   'text_pos_pct',   'text_neg_pct')
print("  ✓ Monthly trend data prepared")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 9 — VISUALISATIONS (5 Figures, now with 4 apps)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 9 — GENERATING VISUALISATIONS")
print("═"*60)

# ── Figure 1: Model Performance ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Text-Based Sentiment Model — Performance Overview (4 Apps)',
             fontsize=15, fontweight='bold', color='#e0e0ff', y=1.01)

# 1a. Confusion matrix
ax = axes[0]
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax.imshow(cm_pct, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
labels = ['Positive','Neutral','Negative']
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})',
                ha='center', va='center', fontsize=8,
                color='white' if cm_pct[i,j] < 50 else 'black')
ax.set_xlabel('Predicted', labelpad=8); ax.set_ylabel('Actual', labelpad=8)
ax.set_title('Confusion Matrix\n(5-Fold CV)', fontsize=11, color='#c0c0e0')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 1b. Accuracy by app (now 4 bars)
ax = axes[1]
app_accs = []
for app in APPS:
    sub = raw[raw['app'] == app]
    app_accs.append({
        'app': app,
        'accuracy': accuracy_score(sub['rating_sentiment'], sub['text_sentiment']) * 100,
        'f1':       f1_score(sub['rating_sentiment'], sub['text_sentiment'], average='weighted') * 100
    })
app_acc_df = pd.DataFrame(app_accs)
x = np.arange(len(APPS))
bars1 = ax.bar(x - 0.2, app_acc_df['accuracy'], 0.35,
               label='Accuracy',    color=[COLORS[a] for a in APPS], alpha=0.85, zorder=3)
bars2 = ax.bar(x + 0.2, app_acc_df['f1'], 0.35,
               label='Weighted F1', color=[COLORS[a] for a in APPS], alpha=0.45, zorder=3)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7.5, color='#c0c0e0')
ax.set_xticks(x); ax.set_xticklabels(APPS)
ax.set_ylim(50, 108); ax.yaxis.grid(True, zorder=0)
ax.set_title('Accuracy & F1 by App', fontsize=11, color='#c0c0e0')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylabel('Score (%)')

# 1c. Disagreement types
ax = axes[2]
disagree_counts = raw[~raw['agreement']]['disagree_type'].value_counts()
colors_d = ['#f43f5e','#f97316','#f59e0b','#10b981','#38bdf8','#8b5cf6']
wedges, texts, autotexts = ax.pie(
    disagree_counts.values, labels=None,
    autopct='%1.1f%%', colors=colors_d[:len(disagree_counts)],
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='#0f0f18', linewidth=2)
)
for at in autotexts: at.set_fontsize(8); at.set_color('white')
ax.legend(wedges, [t[:35] for t in disagree_counts.index],
          loc='lower center', bbox_to_anchor=(0.5, -0.28),
          fontsize=7, ncol=2, framealpha=0)
ax.set_title(f'Disagreement Breakdown\n({100-total_agree:.1f}% of all reviews)',
             fontsize=11, color='#c0c0e0')

plt.tight_layout()
plt.savefig('/home/claude/fig1_model_performance.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 1: Model performance")

# ── Figure 2: Monthly Sentiment — 4 subplots ─────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(20, 18))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Monthly Positive Sentiment — Rating-Based vs Text-Based (4 Apps)',
             fontsize=15, fontweight='bold', color='#e0e0ff', y=1.01)

for ax, app in zip(axes, APPS):
    color = COLORS[app]
    mr = monthly_rating[(monthly_rating['app'] == app) & (monthly_rating['total'] >= 10)]
    mt = monthly_text[(monthly_text['app'] == app) & (monthly_text['total'] >= 10)]

    if 'rating_pos_pct' in mr.columns:
        ax.plot(mr['month_dt'], mr['rating_pos_pct'],
                color=color, linewidth=2.5, label='Rating-Based % Positive', alpha=0.9, zorder=3)
        ax.fill_between(mr['month_dt'], mr['rating_pos_pct'], alpha=0.10, color=color)

    if 'text_pos_pct' in mt.columns:
        ax.plot(mt['month_dt'], mt['text_pos_pct'],
                color='white', linewidth=2, linestyle='--',
                label='Text-Based % Positive', alpha=0.8, zorder=4)

    ax.axhline(50, color='#6060a0', linewidth=1, linestyle=':', alpha=0.6)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title(app, fontsize=12, color=color, fontweight='bold')
    ax.set_ylabel('% Positive', color='#a0a0c0')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)

    # Annotate max gap month
    if 'rating_pos_pct' in mr.columns and 'text_pos_pct' in mt.columns:
        merged_m = mr.merge(mt[['month_dt','text_pos_pct']], on='month_dt', how='inner')
        if len(merged_m) > 0:
            merged_m['gap'] = abs(merged_m['rating_pos_pct'] - merged_m['text_pos_pct'])
            max_row = merged_m.loc[merged_m['gap'].idxmax()]
            ax.annotate(
                f"Max gap: {max_row['gap']:.1f}pp\n{max_row['month_dt'].strftime('%b %Y')}",
                xy=(max_row['month_dt'], max_row['rating_pos_pct']),
                xytext=(30, -30), textcoords='offset points',
                fontsize=8, color='#f59e0b',
                arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.2)
            )

plt.tight_layout()
plt.savefig('/home/claude/fig2_monthly_trends.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 2: Monthly trends")

# ── Figure 3: ABSA Heatmap — now 4 columns ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Aspect-Based Sentiment — Negativity Rate by App & Topic (4 Apps)',
             fontsize=14, fontweight='bold', color='#e0e0ff')

for ax, metric, label in [
    (axes[0], 'rating_neg_pct', 'Rating-Based Negativity %'),
    (axes[1], 'text_neg_pct',   'Text-Based Negativity %')
]:
    pivot = absa_df.pivot(index='topic', columns='app', values=metric)
    pivot = pivot.reindex(columns=APPS).fillna(0)
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.1f',
                linewidths=0.5, linecolor='#0f0f18',
                cbar_kws={'label': label}, vmin=0, vmax=100,
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title(label, fontsize=12, color='#c0c0e0', pad=10)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(colors='#a0a0c0')
    for lbl_obj in ax.get_xticklabels():
        lbl_obj.set_color(COLORS.get(lbl_obj.get_text(), '#a0a0c0'))
        lbl_obj.set_fontweight('bold')

plt.tight_layout()
plt.savefig('/home/claude/fig3_absa_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 3: ABSA heatmap")

# ── Figure 4: Top words — 4 rows × 3 sentiment columns ───────────────────
fig, axes = plt.subplots(4, 3, figsize=(22, 20))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Most Discriminating Words — Per App, Per Sentiment Class (4 Apps)',
             fontsize=14, fontweight='bold', color='#e0e0ff')

sentiment_colors = {
    'Positive': '#10b981',
    'Neutral':  '#f59e0b',
    'Negative': '#f43f5e',
}

for row_idx, app in enumerate(APPS):
    sub     = raw[raw['app'] == app]
    X_sub_w = tfidf_word.transform(sub['clean_content'])
    X_sub_c = tfidf_char.transform(sub['clean_content'])
    X_sub   = hstack([X_sub_w, X_sub_c])
    y_sub   = sub['rating_sentiment'].values

    clf_app = LogisticRegression(C=1.0, max_iter=500, solver='saga',
                                  n_jobs=-1, random_state=42)
    clf_app.fit(X_sub, y_sub)

    for col_idx, sentiment_cls in enumerate(['Positive','Neutral','Negative']):
        ax = axes[row_idx][col_idx]
        if sentiment_cls not in clf_app.classes_:
            ax.axis('off'); continue

        cls_idx = list(clf_app.classes_).index(sentiment_cls)
        coefs   = clf_app.coef_[cls_idx][:X_sub_w.shape[1]]
        top_idx = np.argsort(coefs)[-15:]
        top_words = [(feature_names_word[i], coefs[i]) for i in top_idx]
        top_words.sort(key=lambda x: x[1])

        words  = [w for w, _ in top_words]
        scores = [s for _, s in top_words]
        bar_colors = [sentiment_colors[sentiment_cls] if s > 0 else '#555577' for s in scores]

        ax.barh(words, scores, color=bar_colors, alpha=0.85, zorder=3)
        ax.axvline(0, color='#6060a0', linewidth=1, linestyle='-', alpha=0.5)
        ax.xaxis.grid(True, zorder=0); ax.yaxis.grid(False)
        ax.set_title(f'{app} — {sentiment_cls}', fontsize=9.5,
                     color=sentiment_colors[sentiment_cls], fontweight='bold')
        ax.tick_params(axis='y', labelsize=7.5)
        ax.tick_params(axis='x', labelsize=7)

plt.tight_layout()
plt.savefig('/home/claude/fig4_top_words.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 4: Top discriminating words")

# ── Figure 5: Agreement by year — 4 panels ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Rating vs Text Agreement Rate by Year — All 4 Apps',
             fontsize=14, fontweight='bold', color='#e0e0ff')

for ax, app in zip(axes, APPS):
    sub      = raw[raw['app'] == app]
    yearly_ag  = sub.groupby('year')['agreement'].mean() * 100
    yearly_cnt = sub.groupby('year').size()

    color = COLORS[app]
    bars  = ax.bar(yearly_ag.index, yearly_ag.values,
                   color=color, alpha=0.8, zorder=3, width=0.6)
    ax.plot(yearly_ag.index, yearly_ag.values,
            color='white', linewidth=2, marker='o', markersize=6, zorder=4)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color='#c0c0e0')

    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title(app, fontsize=12, color=color, fontweight='bold')
    ax.set_ylabel('Agreement %')
    ax.set_xlabel('Year')
    ax.axhline(75, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.6, label='75% ref')
    ax.legend(fontsize=8, framealpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig5_yearly_agreement.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 5: Yearly agreement")

# ── Figure 6 (NEW): Cross-App Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Cross-App Comparison — All 4 MFS Apps',
             fontsize=14, fontweight='bold', color='#e0e0ff')

# 6a: Avg rating by app
ax = axes[0]
avg_ratings = {app: raw[raw['app']==app]['score'].mean() for app in APPS}
bars = ax.bar(APPS, [avg_ratings[a] for a in APPS],
              color=[COLORS[a] for a in APPS], alpha=0.85, zorder=3, width=0.55)
for bar, app in zip(bars, APPS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{avg_ratings[app]:.2f}', ha='center', va='bottom', fontsize=10,
            color='#e0e0ff', fontweight='bold')
ax.set_ylim(1, 5.2); ax.yaxis.grid(True, zorder=0)
ax.axhline(3.0, color='#6060a0', linestyle=':', linewidth=1.5, alpha=0.7, label='3.0 benchmark')
ax.set_title('Average Star Rating', fontsize=11, color='#c0c0e0')
ax.set_ylabel('Avg Rating'); ax.legend(fontsize=8, framealpha=0.3)

# 6b: Sentiment mix by app (stacked 100%)
ax = axes[1]
sentiment_mix = {}
for app in APPS:
    sub = raw[raw['app'] == app]
    total = len(sub)
    sentiment_mix[app] = {
        'Positive': (sub['text_sentiment']=='Positive').sum() / total * 100,
        'Neutral':  (sub['text_sentiment']=='Neutral').sum()  / total * 100,
        'Negative': (sub['text_sentiment']=='Negative').sum() / total * 100,
    }
x = np.arange(len(APPS))
bottoms = np.zeros(len(APPS))
for s_cls, s_color in [('Positive','#10b981'),('Neutral','#f59e0b'),('Negative','#f43f5e')]:
    vals = [sentiment_mix[a][s_cls] for a in APPS]
    ax.bar(x, vals, bottom=bottoms, color=s_color, alpha=0.85, label=s_cls, zorder=3)
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
    bottoms += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(APPS)
ax.set_ylim(0, 105); ax.yaxis.grid(True, zorder=0)
ax.set_title('Text Sentiment Mix (100% stacked)', fontsize=11, color='#c0c0e0')
ax.legend(fontsize=8, loc='upper right', framealpha=0.3)
ax.set_ylabel('% of Reviews')

# 6c: Negativity rate by topic across apps (grouped)
ax = axes[2]
topic_list = ['App Performance','Login/Authentication','Transaction/Payment','Customer Support']
topic_short = ['App Perf','Login/Auth','Transaction','Cust Support']
x = np.arange(len(topic_list))
width = 0.2
for i, app in enumerate(APPS):
    neg_rates = []
    for topic in topic_list:
        sub = raw[(raw['app']==app) & (raw['topic']==topic)]
        if len(sub) > 0:
            neg_rates.append((sub['text_sentiment']=='Negative').mean() * 100)
        else:
            neg_rates.append(0)
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, neg_rates, width, label=app,
                  color=COLORS[app], alpha=0.85, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(topic_short, fontsize=8.5)
ax.set_ylim(0, 108); ax.yaxis.grid(True, zorder=0)
ax.axhline(50, color='#6060a0', linestyle=':', linewidth=1.2, alpha=0.6)
ax.set_title('Text Negativity Rate by Topic', fontsize=11, color='#c0c0e0')
ax.legend(fontsize=8, framealpha=0.3)
ax.set_ylabel('% Negative Reviews')

plt.tight_layout()
plt.savefig('/home/claude/fig6_cross_app_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 6: Cross-app comparison (NEW)")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 10 — EXPORT TO EXCEL (8 sheets)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 10 — EXPORTING RESULTS TO EXCEL")
print("═"*60)

output_path = '/home/claude/mfs_4app_text_sentiment_analysis.xlsx'

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    wb = writer.book

    # Shared formats
    hdr  = wb.add_format({'bold':True, 'bg_color':'#1a1a2e', 'font_color':'#e0e0ff',
                           'border':1, 'border_color':'#2a2a4e',
                           'align':'center', 'valign':'vcenter'})
    cell = wb.add_format({'bg_color':'#0f1020', 'font_color':'#c0c0d8',
                           'border':1, 'border_color':'#1e1e38'})
    kv   = wb.add_format({'bold':True, 'font_size':14, 'font_color':'#2dd4bf',
                           'bg_color':'#0f0f18'})
    lbl  = wb.add_format({'font_size':10, 'font_color':'#a0a0c0', 'bg_color':'#0f0f18'})
    ag_f = wb.add_format({'bg_color':'#052e16', 'font_color':'#6ee7b7',
                           'align':'center', 'border':1})
    dg_f = wb.add_format({'bg_color':'#2d0a12', 'font_color':'#fca5a5',
                           'align':'center', 'border':1})

    # ── Sheet 1: Full Data ──────────────────────────────────────────────
    cols = ['app','userName','content','score','rating_sentiment',
            'text_sentiment','agreement','disagree_type','topic',
            'thumbsUpCount','year','month','has_bengali']
    export_df = raw[cols].copy()
    export_df.to_excel(writer, sheet_name='Full Data', index=False)
    ws = writer.sheets['Full Data']
    for c, h in enumerate(cols): ws.write(0, c, h, hdr)
    ws.set_column('A:A', 9); ws.set_column('B:B', 20)
    ws.set_column('C:C', 65); ws.set_column('D:D', 7)
    ws.set_column('E:H', 16); ws.set_column('I:I', 24)
    ws.set_column('J:M', 14)
    ws.freeze_panes(1, 0)
    ws.conditional_format(1, 6, len(export_df), 6, {
        'type':'cell','criteria':'==','value':'"True"','format':ag_f})
    ws.conditional_format(1, 6, len(export_df), 6, {
        'type':'cell','criteria':'==','value':'"False"','format':dg_f})
    print("  ✓ Sheet 1: Full Data")

    # ── Sheet 2: Model Performance ──────────────────────────────────────
    perf_rows = []
    for app in APPS + ['ALL']:
        sub = raw if app == 'ALL' else raw[raw['app'] == app]
        for cls in ['Positive','Neutral','Negative']:
            y_true_b = (sub['rating_sentiment'] == cls).astype(int)
            y_pred_b = (sub['text_sentiment']   == cls).astype(int)
            perf_rows.append({
                'App': app, 'Class': cls,
                'N_reviews':   len(sub),
                'Accuracy':    accuracy_score(sub['rating_sentiment'], sub['text_sentiment']),
                'F1_weighted': f1_score(sub['rating_sentiment'], sub['text_sentiment'], average='weighted'),
                'Precision':   precision_score(y_true_b, y_pred_b, zero_division=0),
                'Recall':      recall_score(y_true_b, y_pred_b, zero_division=0),
                'Agreement_pct': sub['agreement'].mean(),
            })
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_excel(writer, sheet_name='Model Performance', index=False)
    ws2 = writer.sheets['Model Performance']
    for c, h in enumerate(perf_df.columns): ws2.write(0, c, h, hdr)
    ws2.set_column('A:B', 14); ws2.set_column('C:H', 16)
    ws2.conditional_format(1, 7, len(perf_df), 7, {
        'type':'3_color_scale',
        'min_color':'#5c0011','mid_color':'#78350f','max_color':'#052e16'})
    print("  ✓ Sheet 2: Model Performance")

    # ── Sheet 3: Disagreements ──────────────────────────────────────────
    disagree_export = raw[~raw['agreement']].sort_values(
        'thumbsUpCount', ascending=False
    )[['app','content','score','rating_sentiment','text_sentiment',
       'disagree_type','topic','thumbsUpCount','year']].head(6000)
    disagree_export.to_excel(writer, sheet_name='Disagreements', index=False)
    ws3 = writer.sheets['Disagreements']
    for c, h in enumerate(disagree_export.columns): ws3.write(0, c, h, hdr)
    ws3.set_column('A:A', 10); ws3.set_column('B:B', 72)
    ws3.set_column('C:I', 18); ws3.freeze_panes(1, 0)
    print("  ✓ Sheet 3: Disagreements")

    # ── Sheet 4: ABSA Results ───────────────────────────────────────────
    absa_df.to_excel(writer, sheet_name='ABSA Results', index=False)
    ws4 = writer.sheets['ABSA Results']
    for c, h in enumerate(absa_df.columns): ws4.write(0, c, h, hdr)
    ws4.set_column('A:B', 18); ws4.set_column('C:I', 15)
    ws4.conditional_format(1, 5, len(absa_df)+1, 5, {
        'type':'3_color_scale',
        'min_color':'#5c0011','mid_color':'#1a1a2e','max_color':'#052e16'})
    ws4.conditional_format(1, 7, len(absa_df)+1, 7, {
        'type':'3_color_scale',
        'min_color':'#052e16','mid_color':'#1a1a2e','max_color':'#5c0011'})
    print("  ✓ Sheet 4: ABSA Results")

    # ── Sheet 5: Monthly Trend ──────────────────────────────────────────
    monthly_merged = monthly_rating.merge(
        monthly_text[['app','month_dt','text_pos_pct','text_neg_pct']],
        on=['app','month_dt'], how='left'
    )
    avail = [c for c in ['app','month_dt','total','rating_pos_pct',
                          'text_pos_pct','text_neg_pct'] if c in monthly_merged.columns]
    monthly_out = monthly_merged[avail].copy()
    col_map = {'app':'App','month_dt':'Month','total':'Review Count',
               'rating_pos_pct':'Rating Positive %',
               'text_pos_pct':'Text Positive %','text_neg_pct':'Text Negative %'}
    monthly_out.columns = [col_map.get(c, c) for c in monthly_out.columns]
    monthly_out['Month'] = monthly_out['Month'].dt.strftime('%Y-%m')
    monthly_out.to_excel(writer, sheet_name='Monthly Trend', index=False)
    ws5 = writer.sheets['Monthly Trend']
    for c, h in enumerate(monthly_out.columns): ws5.write(0, c, h, hdr)
    ws5.set_column('A:B', 12); ws5.set_column('C:G', 20)
    ws5.freeze_panes(1, 0)
    print("  ✓ Sheet 5: Monthly Trend")

    # ── Sheet 6: Top Features ───────────────────────────────────────────
    word_rows = []
    for cls in classes:
        for rank, (word, score) in enumerate(top_features[cls], 1):
            word_rows.append({'Sentiment':cls, 'Rank':rank,
                              'Word/Phrase':word, 'TF-IDF Coefficient':score})
    word_df = pd.DataFrame(word_rows)
    word_df.to_excel(writer, sheet_name='Top Features', index=False)
    ws6 = writer.sheets['Top Features']
    for c, h in enumerate(word_df.columns): ws6.write(0, c, h, hdr)
    ws6.set_column('A:A', 14); ws6.set_column('B:B', 8)
    ws6.set_column('C:C', 30); ws6.set_column('D:D', 22)
    print("  ✓ Sheet 6: Top Features")

    # ── Sheet 7: Cross-App Summary ──────────────────────────────────────
    cross_rows = []
    for app in APPS:
        sub = raw[raw['app'] == app]
        cross_rows.append({
            'App':            app,
            'Total Reviews':  len(sub),
            'Avg Rating':     round(sub['score'].mean(), 3),
            'Rating Positive %': round((sub['rating_sentiment']=='Positive').mean()*100, 1),
            'Rating Negative %': round((sub['rating_sentiment']=='Negative').mean()*100, 1),
            'Text Positive %':   round((sub['text_sentiment']=='Positive').mean()*100, 1),
            'Text Negative %':   round((sub['text_sentiment']=='Negative').mean()*100, 1),
            'Agreement %':       round(sub['agreement'].mean()*100, 1),
            'Accuracy':          round(accuracy_score(sub['rating_sentiment'], sub['text_sentiment'])*100, 2),
            'F1 Weighted':       round(f1_score(sub['rating_sentiment'], sub['text_sentiment'], average='weighted'), 4),
        })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_excel(writer, sheet_name='Cross-App Summary', index=False)
    ws7 = writer.sheets['Cross-App Summary']
    for c, h in enumerate(cross_df.columns): ws7.write(0, c, h, hdr)
    ws7.set_column('A:A', 10); ws7.set_column('B:J', 20)
    ws7.conditional_format(1, 6, len(cross_df), 6, {
        'type':'3_color_scale',
        'min_color':'#052e16','mid_color':'#1a1a2e','max_color':'#5c0011'})
    print("  ✓ Sheet 7: Cross-App Summary")

    # ── Sheet 8: Executive Summary ──────────────────────────────────────
    ws8 = wb.add_worksheet('Executive Summary')
    ws8.set_tab_color('#f97316')
    big   = wb.add_format({'bold':True, 'font_size':18, 'font_color':'#e0e0ff', 'bg_color':'#0f0f18'})
    med   = wb.add_format({'bold':True, 'font_size':12, 'font_color':'#c0c0e0', 'bg_color':'#0f0f18'})
    ws8.set_column('A:A', 38); ws8.set_column('B:D', 22)

    ws8.write('A1', 'MFS App Review — 4-App Text Sentiment Analysis', big)
    ws8.write('A2', 'Method : TF-IDF (word 1-2gram + char 3-6gram) + Logistic Regression', lbl)
    ws8.write('A3', f'Dataset: bKash + Rocket + Upay + Nagad · {len(raw):,} reviews · 2020–2026', lbl)

    ws8.write('A5', 'GLOBAL METRICS', med)
    for i, (label, value) in enumerate([
        ('Total Reviews Analysed', f'{len(raw):,}'),
        ('Overall Accuracy (5-fold CV)', f'{acc*100:.2f}%'),
        ('Weighted F1 Score', f'{f1:.4f}'),
        ('Agreement Rate (text vs rating)', f'{total_agree:.1f}%'),
        ('Disagreement Rate', f'{100-total_agree:.1f}%'),
        ('Bengali-script reviews', f'{bengali_pct:.1f}%'),
        ('Feature Dimensions', f'{X.shape[1]:,}'),
    ], start=6):
        ws8.write(f'A{i}', label, lbl)
        ws8.write(f'B{i}', value, kv)

    ws8.write('A14', 'PER-APP ACCURACY', med)
    for i, row in enumerate(app_acc_df.itertuples(), 15):
        ws8.write(f'A{i}', row.app, lbl)
        ws8.write(f'B{i}', f'{row.accuracy:.1f}%', kv)
        ws8.write(f'C{i}', f'F1: {row.f1:.1f}%', lbl)

    ws8.write('A20', 'PER-APP SENTIMENT (TEXT-BASED)', med)
    for i, row in enumerate(cross_df.itertuples(), 21):
        ws8.write(f'A{i}', row.App, lbl)
        ws8.write(f'B{i}', f'Avg: {row._3:.2f}★', kv)
        ws8.write(f'C{i}', f'+{row._6:.1f}% pos', lbl)
        ws8.write(f'D{i}', f'-{row._7:.1f}% neg', lbl)

    ws8.write('A27', 'TOP DISAGREEMENT TYPES', med)
    for i, (dtype, cnt) in enumerate(
        raw[~raw['agreement']]['disagree_type'].value_counts().items(), 28
    ):
        ws8.write(f'A{i}', dtype, lbl)
        ws8.write(f'B{i}', f'{cnt:,}  ({cnt/len(raw)*100:.1f}%)', kv)

    print("  ✓ Sheet 8: Executive Summary")

print(f"\n  ✓ Excel saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PIPELINE COMPLETE — FINAL SUMMARY")
print("═"*60)

print(f"""
  DATASET
  ─────────────────────────────────────────────────────
  Apps            : {', '.join(APPS)}
  Total Reviews   : {len(raw):,}
  Date Range      : {raw['at'].min().strftime('%b %Y')} → {raw['at'].max().strftime('%b %Y')}
  Features        : {X.shape[1]:,}  (word TF-IDF + char TF-IDF)

  MODEL PERFORMANCE
  ─────────────────────────────────────────────────────
  Method          : Logistic Regression (saga, multinomial)
  Validation      : 5-Fold Stratified Cross-Validation
  Overall Accuracy: {acc*100:.2f}%
  Weighted F1     : {f1:.4f}
  Agreement Rate  : {total_agree:.1f}%
  Disagreement    : {100-total_agree:.1f}%

  PER-APP RESULTS
  ─────────────────────────────────────────────────────""")
for row in app_acc_df.itertuples():
    sub = raw[raw['app'] == row.app]
    print(f"  {row.app:8s} | Accuracy: {row.accuracy:.1f}%  |  "
          f"Avg rating: {sub['score'].mean():.2f}  |  "
          f"Agreement: {sub['agreement'].mean()*100:.1f}%")

print(f"""
  OUTPUTS
  ─────────────────────────────────────────────────────
  Excel (8 sheets): mfs_4app_text_sentiment_analysis.xlsx
  Figure 1        : fig1_model_performance.png
  Figure 2        : fig2_monthly_trends.png        (4 panels)
  Figure 3        : fig3_absa_heatmap.png           (4 cols)
  Figure 4        : fig4_top_words.png              (4 rows)
  Figure 5        : fig5_yearly_agreement.png       (4 panels)
  Figure 6        : fig6_cross_app_comparison.png   (NEW)
""")
