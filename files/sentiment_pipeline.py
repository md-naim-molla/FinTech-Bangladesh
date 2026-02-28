"""
=============================================================
  MFS APP REVIEW — FULL TEXT-BASED SENTIMENT PIPELINE
  Apps: bKash · Rocket · Upay
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ── Styling ────────────────────────────────────────────────────────────────
COLORS = {
    'bKash':   '#E2136E',
    'Rocket':  '#7B2FBE',
    'Upay':    '#0d9488',
    'Positive':'#10b981',
    'Neutral': '#f59e0b',
    'Negative':'#f43f5e',
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

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD & MERGE ALL THREE DATASETS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 1 — DATA LOADING")
print("═"*60)

files = {
    'bKash':  '/mnt/user-data/uploads/bkash_2020-2026_.csv',
    'Rocket': '/mnt/user-data/uploads/rocket_2020-2026_.csv',
    'Upay':   '/mnt/user-data/uploads/upay_2020-2026_.csv',
}

frames = []
for app, path in files.items():
    df = pd.read_csv(path)
    df['app'] = app
    frames.append(df)
    print(f"  ✓ {app}: {len(df):,} reviews loaded")

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

# --- Rating-based sentiment (our baseline) ---
def rating_sentiment(s):
    if s >= 4: return 'Positive'
    if s == 3: return 'Neutral'
    return 'Negative'

raw['rating_sentiment'] = raw['score'].apply(rating_sentiment)

# --- Text cleaning ---
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)        # remove URLs
    text = re.sub(r'[^\w\s\u0980-\u09FF]', ' ', text)  # keep alphanumeric + Bengali unicode
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

raw['clean_content'] = raw['content'].apply(clean_text)

# Language detection heuristic (Bengali unicode range U+0980–U+09FF)
def has_bengali(text):
    return bool(re.search(r'[\u0980-\u09FF]', str(text)))

raw['has_bengali'] = raw['content'].apply(has_bengali)
bengali_pct = raw['has_bengali'].mean() * 100
print(f"  Reviews containing Bengali script: {bengali_pct:.1f}%")
print(f"  Reviews — bKash: {(raw.app=='bKash').sum():,}  "
      f"Rocket: {(raw.app=='Rocket').sum():,}  "
      f"Upay: {(raw.app=='Upay').sum():,}")
print(f"  Class distribution (rating-based):")
for s, c in raw['rating_sentiment'].value_counts().items():
    print(f"    {s}: {c:,} ({c/len(raw)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — TOPIC CLASSIFIER (keyword-based, same as dashboards)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 3 — TOPIC CLASSIFICATION")
print("═"*60)

TOPICS = {
    'Transaction/Payment':  ['send money','transaction','payment','transfer','pay',
                              'cash out','add money','failed','deducted','recharge','bill'],
    'App Performance':      ['crash','slow','lag','freeze','bug','error',
                              'not working','loading','update','install'],
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
print("  Topic distribution:")
for t, c in raw['topic'].value_counts().items():
    print(f"    {t}: {c:,}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — ML SENTIMENT MODEL (TF-IDF + Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 4 — ML MODEL: TF-IDF + LOGISTIC REGRESSION")
print("═"*60)

# Build the sklearn Pipeline
# TF-IDF params tuned for mixed-language short texts:
#   - char_wb n-grams (3-6): captures subword patterns, works for Bangla
#   - word n-grams (1-2): captures phrases like "not working", "very good"
#   - sublinear_tf: dampens the effect of very frequent terms

tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=25000,
    sublinear_tf=True,
    min_df=3,
    strip_accents='unicode',
)

tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 6),
    max_features=30000,
    sublinear_tf=True,
    min_df=5,
)

from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack

print("  Fitting word TF-IDF...")
X_word = tfidf_word.fit_transform(raw['clean_content'])
print(f"    Word features: {X_word.shape[1]:,}")

print("  Fitting char TF-IDF...")
X_char = tfidf_char.fit_transform(raw['clean_content'])
print(f"    Char features: {X_char.shape[1]:,}")

# Combine both feature matrices
X = hstack([X_word, X_char])
y = raw['rating_sentiment'].values
print(f"  Combined feature matrix: {X.shape[0]:,} × {X.shape[1]:,}")

# ── 5-Fold Stratified Cross-Validation ────────────────────────────────────
print("\n  Running 5-fold stratified cross-validation...")
clf = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='saga',
    n_jobs=-1,
    random_state=42
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

# Store CV predictions
raw['text_sentiment'] = y_pred_cv

# ── Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y, y_pred_cv, labels=['Positive','Neutral','Negative'])
print("  Confusion Matrix (rows=actual, cols=predicted):")
print("  Labels: Positive | Neutral | Negative")
for i, row in enumerate(['Positive','Neutral','Negative']):
    print(f"    {row:10s}: {cm[i]}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — AGREEMENT ANALYSIS (Rating vs Text Sentiment)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 5 — AGREEMENT ANALYSIS")
print("═"*60)

raw['agreement'] = raw['rating_sentiment'] == raw['text_sentiment']
raw['disagree_type'] = 'Agree'
raw.loc[(raw['rating_sentiment']=='Positive') & (raw['text_sentiment']=='Negative'), 'disagree_type'] = 'Rated High, Sounds Negative'
raw.loc[(raw['rating_sentiment']=='Negative') & (raw['text_sentiment']=='Positive'), 'disagree_type'] = 'Rated Low, Sounds Positive'
raw.loc[(raw['rating_sentiment']=='Positive') & (raw['text_sentiment']=='Neutral'), 'disagree_type']  = 'Rated High, Sounds Neutral'
raw.loc[(raw['rating_sentiment']=='Neutral')  & (raw['text_sentiment']=='Negative'),'disagree_type']  = 'Rated Mid, Sounds Negative'
raw.loc[(raw['rating_sentiment']=='Neutral')  & (raw['text_sentiment']=='Positive'),'disagree_type']  = 'Rated Mid, Sounds Positive'
raw.loc[(raw['rating_sentiment']=='Negative') & (raw['text_sentiment']=='Neutral'), 'disagree_type']  = 'Rated Low, Sounds Neutral'

total_agree = raw['agreement'].mean() * 100
print(f"  Overall agreement rate: {total_agree:.1f}%")
print(f"  Disagreement rate     : {100-total_agree:.1f}%\n")

print("  Disagreement types:")
for dtype, cnt in raw[~raw['agreement']]['disagree_type'].value_counts().items():
    pct = cnt / len(raw) * 100
    print(f"    {dtype:45s}: {cnt:5,}  ({pct:.1f}%)")

print("\n  Agreement by app:")
for app in ['bKash','Rocket','Upay']:
    sub = raw[raw['app']==app]
    ag  = sub['agreement'].mean()*100
    print(f"    {app:8s}: {ag:.1f}% agreement  ({100-ag:.1f}% disagree)")

# Most interesting: high-rated but sounds negative
interesting = raw[raw['disagree_type']=='Rated High, Sounds Negative'].sort_values('thumbsUpCount', ascending=False)
print(f"\n  Top 5 'Rated High but Sounds Negative' reviews (most helpful):")
for _, row in interesting.head(5).iterrows():
    print(f"  [{row['app']}] ★{row['score']} | {str(row['content'])[:120]}...")
    print()


# ══════════════════════════════════════════════════════════════════════════
# STAGE 6 — ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 6 — ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
print("═"*60)

absa_rows = []
for app in ['bKash','Rocket','Upay']:
    for topic in [t for t in TOPICS.keys()]:
        sub = raw[(raw['app']==app) & (raw['topic']==topic)]
        if len(sub) < 10:
            continue
        pos_r = (sub['rating_sentiment']=='Positive').mean()
        neg_r = (sub['rating_sentiment']=='Negative').mean()
        pos_t = (sub['text_sentiment']=='Positive').mean()
        neg_t = (sub['text_sentiment']=='Negative').mean()
        absa_rows.append({
            'app': app, 'topic': topic, 'n': len(sub),
            'rating_pos_pct': round(pos_r*100,1),
            'rating_neg_pct': round(neg_r*100,1),
            'text_pos_pct':   round(pos_t*100,1),
            'text_neg_pct':   round(neg_t*100,1),
            'pos_gap': round((pos_t - pos_r)*100,1),
            'neg_gap': round((neg_t - neg_r)*100,1),
        })

absa_df = pd.DataFrame(absa_rows)
print("\n  ABSA Summary (rating vs text sentiment per app+topic):")
print(f"  {'App':8s} {'Topic':25s} {'N':>6s} | {'Rating+':>8s} {'Text+':>7s} {'Gap':>6s} | {'Rating-':>8s} {'Text-':>7s} {'Gap':>6s}")
print("  " + "-"*90)
for _, r in absa_df.sort_values(['app','neg_gap']).iterrows():
    print(f"  {r['app']:8s} {r['topic']:25s} {int(r['n']):>6,} | "
          f"{r['rating_pos_pct']:>7.1f}% {r['text_pos_pct']:>6.1f}% {r['pos_gap']:>+6.1f}% | "
          f"{r['rating_neg_pct']:>7.1f}% {r['text_neg_pct']:>6.1f}% {r['neg_gap']:>+6.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 7 — TOP FEATURES (Most Informative Words Per Sentiment)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 7 — TOP DISCRIMINATING WORDS PER SENTIMENT CLASS")
print("═"*60)

# Refit on full data to get coefficients
clf.fit(X, y)
classes = clf.classes_
feature_names_word = tfidf_word.get_feature_names_out()
# Only use word features for interpretability
n_word = X_word.shape[1]

top_features = {}
for i, cls in enumerate(classes):
    coefs = clf.coef_[i][:n_word]  # word TF-IDF coefficients only
    top_idx = np.argsort(coefs)[-20:][::-1]
    top_features[cls] = [(feature_names_word[j], round(coefs[j],3)) for j in top_idx]

for cls in classes:
    print(f"\n  Top words → {cls}:")
    words = [f"{w}({s:+.2f})" for w, s in top_features[cls][:15]]
    print("    " + "  ·  ".join(words))


# ══════════════════════════════════════════════════════════════════════════
# STAGE 8 — MONTHLY TEXT SENTIMENT TREND
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 8 — MONTHLY TEXT SENTIMENT TRENDS")
print("═"*60)

monthly_text = raw.groupby(['app','month','text_sentiment']).size().unstack(fill_value=0).reset_index()
monthly_text['month_dt'] = pd.to_datetime(monthly_text['month'])
monthly_text = monthly_text.sort_values(['app','month_dt'])
if 'Positive' in monthly_text.columns and 'Negative' in monthly_text.columns:
    monthly_text['total'] = monthly_text[['Positive','Neutral','Negative']].sum(axis=1)
    monthly_text['text_pos_pct'] = monthly_text['Positive'] / monthly_text['total'] * 100
    monthly_text['text_neg_pct'] = monthly_text['Negative'] / monthly_text['total'] * 100

monthly_rating = raw.groupby(['app','month','rating_sentiment']).size().unstack(fill_value=0).reset_index()
monthly_rating['month_dt'] = pd.to_datetime(monthly_rating['month'])
monthly_rating = monthly_rating.sort_values(['app','month_dt'])
if 'Positive' in monthly_rating.columns and 'Negative' in monthly_rating.columns:
    monthly_rating['total'] = monthly_rating[['Positive','Neutral','Negative']].sum(axis=1)
    monthly_rating['rating_pos_pct'] = monthly_rating['Positive'] / monthly_rating['total'] * 100

print("  ✓ Monthly trend data prepared")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 9 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 9 — GENERATING VISUALISATIONS")
print("═"*60)

# ── Figure 1: Model Performance ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Text-Based Sentiment Model — Performance Overview', 
             fontsize=16, fontweight='bold', color='#e0e0ff', y=1.02)

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

# 1b. Accuracy by app
ax = axes[1]
app_accs = []
for app in ['bKash','Rocket','Upay']:
    sub = raw[raw['app']==app]
    app_accs.append({
        'app': app,
        'accuracy': accuracy_score(sub['rating_sentiment'], sub['text_sentiment'])*100,
        'f1': f1_score(sub['rating_sentiment'], sub['text_sentiment'], average='weighted')*100
    })
app_acc_df = pd.DataFrame(app_accs)
x = np.arange(3)
bars1 = ax.bar(x - 0.2, app_acc_df['accuracy'], 0.35, 
               label='Accuracy', color=[COLORS[a] for a in ['bKash','Rocket','Upay']], 
               alpha=0.85, zorder=3)
bars2 = ax.bar(x + 0.2, app_acc_df['f1'], 0.35, 
               label='Weighted F1', color=[COLORS[a] for a in ['bKash','Rocket','Upay']], 
               alpha=0.45, zorder=3)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color='#c0c0e0')
ax.set_xticks(x); ax.set_xticklabels(['bKash','Rocket','Upay'])
ax.set_ylim(50, 105); ax.yaxis.grid(True, zorder=0)
ax.set_title('Accuracy & F1 by App', fontsize=11, color='#c0c0e0')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylabel('Score (%)')

# 1c. Disagreement types
ax = axes[2]
disagree_counts = raw[~raw['agreement']]['disagree_type'].value_counts()
colors_d = ['#f43f5e','#f97316','#f59e0b','#10b981','#38bdf8','#8b5cf6']
wedges, texts, autotexts = ax.pie(
    disagree_counts.values,
    labels=None,
    autopct='%1.1f%%',
    colors=colors_d[:len(disagree_counts)],
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='#0f0f18', linewidth=2)
)
for at in autotexts: at.set_fontsize(8); at.set_color('white')
ax.legend(wedges, [t[:35] for t in disagree_counts.index],
          loc='lower center', bbox_to_anchor=(0.5, -0.25),
          fontsize=7, ncol=2, framealpha=0)
ax.set_title(f'Disagreement Breakdown\n({100-total_agree:.1f}% of all reviews)', 
             fontsize=11, color='#c0c0e0')

plt.tight_layout()
plt.savefig('/home/claude/fig1_model_performance.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 1: Model performance saved")

# ── Figure 2: Monthly Sentiment Trend — Rating vs Text ────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Monthly Positive Sentiment — Rating-Based vs Text-Based',
             fontsize=15, fontweight='bold', color='#e0e0ff', y=1.01)

for ax, app in zip(axes, ['bKash','Rocket','Upay']):
    color = COLORS[app]
    mr = monthly_rating[monthly_rating['app']==app]
    mt = monthly_text[monthly_text['app']==app]
    
    # Only plot months with enough data (>= 10 reviews)
    mr = mr[mr['total'] >= 10]
    mt = mt[mt['total'] >= 10]
    
    if 'rating_pos_pct' in mr.columns:
        ax.plot(mr['month_dt'], mr['rating_pos_pct'],
                color=color, linewidth=2.5, label='Rating-Based % Positive',
                alpha=0.9, zorder=3)
        ax.fill_between(mr['month_dt'], mr['rating_pos_pct'], alpha=0.1, color=color)
    
    if 'text_pos_pct' in mt.columns:
        ax.plot(mt['month_dt'], mt['text_pos_pct'],
                color='white', linewidth=2, label='Text-Based % Positive',
                linestyle='--', alpha=0.8, zorder=4)
    
    ax.axhline(50, color='#6060a0', linewidth=1, linestyle=':', alpha=0.6)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title(f'{app}', fontsize=12, color=color, fontweight='bold')
    ax.set_ylabel('% Positive', color='#a0a0c0')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)
    
    # Annotate max disagreement month
    if 'rating_pos_pct' in mr.columns and 'text_pos_pct' in mt.columns:
        merged_m = mr.merge(mt[['month_dt','text_pos_pct','text_neg_pct']], on='month_dt', how='inner')
        if len(merged_m) > 0:
            merged_m['gap'] = abs(merged_m['rating_pos_pct'] - merged_m['text_pos_pct'])
            max_gap_row = merged_m.loc[merged_m['gap'].idxmax()]
            ax.annotate(f"Max gap: {max_gap_row['gap']:.1f}pp\n{max_gap_row['month_dt'].strftime('%b %Y')}",
                        xy=(max_gap_row['month_dt'], max_gap_row['rating_pos_pct']),
                        xytext=(30, -30), textcoords='offset points',
                        fontsize=8, color='#f59e0b',
                        arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.2))

plt.tight_layout()
plt.savefig('/home/claude/fig2_monthly_trends.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 2: Monthly trends saved")

# ── Figure 3: ABSA Heatmap ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Aspect-Based Sentiment Analysis — Negativity Rate by App & Topic',
             fontsize=14, fontweight='bold', color='#e0e0ff')

for ax, metric, label in [
    (axes[0], 'rating_neg_pct', 'Rating-Based Negativity %'),
    (axes[1], 'text_neg_pct',   'Text-Based Negativity %')
]:
    pivot = absa_df.pivot(index='topic', columns='app', values=metric).fillna(0)
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.1f',
                linewidths=0.5, linecolor='#0f0f18',
                cbar_kws={'label': label},
                vmin=0, vmax=100,
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title(label, fontsize=12, color='#c0c0e0', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(colors='#a0a0c0')
    for label_obj in ax.get_xticklabels():
        label_obj.set_color(COLORS.get(label_obj.get_text(), '#a0a0c0'))
        label_obj.set_fontweight('bold')

plt.tight_layout()
plt.savefig('/home/claude/fig3_absa_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 3: ABSA heatmap saved")

# ── Figure 4: Top words per sentiment per app ─────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Most Discriminating Words — Per App, Per Sentiment Class',
             fontsize=14, fontweight='bold', color='#e0e0ff')

sentiment_colors = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#f43f5e'}

for row_idx, app in enumerate(['bKash','Rocket','Upay']):
    sub     = raw[raw['app']==app]
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
        
        bars = ax.barh(words, scores, color=bar_colors, alpha=0.85, zorder=3)
        ax.axvline(0, color='#6060a0', linewidth=1, linestyle='-', alpha=0.5)
        ax.yaxis.grid(False); ax.xaxis.grid(True, zorder=0)
        
        title_color = sentiment_colors[sentiment_cls]
        ax.set_title(f'{app} — {sentiment_cls}', fontsize=10,
                     color=title_color, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7)

plt.tight_layout()
plt.savefig('/home/claude/fig4_top_words.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 4: Top words per sentiment saved")

# ── Figure 5: Agreement rate by year ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f0f18')
fig.suptitle('Rating vs Text Agreement Rate by Year — Per App',
             fontsize=14, fontweight='bold', color='#e0e0ff')

for ax, app in zip(axes, ['bKash','Rocket','Upay']):
    sub = raw[raw['app']==app]
    yearly_ag = sub.groupby('year')['agreement'].mean() * 100
    yearly_cnt = sub.groupby('year').size()
    
    color = COLORS[app]
    bars = ax.bar(yearly_ag.index, yearly_ag.values, color=color, alpha=0.8, zorder=3, width=0.6)
    ax.plot(yearly_ag.index, yearly_ag.values, color='white', linewidth=2,
            marker='o', markersize=6, zorder=4)
    
    for bar, (yr, cnt) in zip(bars, yearly_cnt.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color='#c0c0e0')
    
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title(f'{app}', fontsize=12, color=color, fontweight='bold')
    ax.set_ylabel('Agreement %')
    ax.set_xlabel('Year')
    ax.axhline(75, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.6, label='75% reference')
    ax.legend(fontsize=8, framealpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig5_yearly_agreement.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f18')
plt.close()
print("  ✓ Figure 5: Yearly agreement saved")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 10 — EXPORT TO EXCEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STAGE 10 — EXPORTING RESULTS TO EXCEL")
print("═"*60)

output_path = '/home/claude/mfs_text_sentiment_analysis.xlsx'

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    wb = writer.book
    
    # Formats
    hdr_fmt  = wb.add_format({'bold':True, 'bg_color':'#1a1a2e', 'font_color':'#e0e0ff',
                               'border':1, 'border_color':'#2a2a4e', 'align':'center', 'valign':'vcenter'})
    title_fmt = wb.add_format({'bold':True, 'font_color':'#e0e0ff', 'font_size':14,
                                'bg_color':'#0f0f18'})
    pos_fmt  = wb.add_format({'bg_color':'#052e16', 'font_color':'#6ee7b7', 'align':'center'})
    neg_fmt  = wb.add_format({'bg_color':'#2d0a12', 'font_color':'#fca5a5', 'align':'center'})
    neu_fmt  = wb.add_format({'bg_color':'#1c1300', 'font_color':'#fcd34d', 'align':'center'})
    num_fmt  = wb.add_format({'num_format':'#,##0', 'align':'center'})
    pct_fmt  = wb.add_format({'num_format':'0.1%', 'align':'center'})
    cell_fmt = wb.add_format({'bg_color':'#0f1020', 'font_color':'#c0c0d8', 'border':1,
                               'border_color':'#1e1e38'})
    agree_fmt   = wb.add_format({'bg_color':'#052e16','font_color':'#6ee7b7','align':'center','border':1})
    disagree_fmt= wb.add_format({'bg_color':'#2d0a12','font_color':'#fca5a5','align':'center','border':1})

    # ── Sheet 1: Full Data ──────────────────────────────────────────────
    export_cols = ['app','userName','content','score','rating_sentiment',
                   'text_sentiment','agreement','disagree_type','topic',
                   'thumbsUpCount','year','month','has_bengali']
    export_df = raw[export_cols].copy()
    export_df.to_excel(writer, sheet_name='Full Data', index=False)
    ws = writer.sheets['Full Data']
    for col_num, col in enumerate(export_cols):
        ws.write(0, col_num, col, hdr_fmt)
    ws.set_column('A:A', 8); ws.set_column('B:B', 20)
    ws.set_column('C:C', 60); ws.set_column('D:D', 7)
    ws.set_column('E:G', 14); ws.set_column('H:H', 35)
    ws.set_column('I:I', 22); ws.set_column('J:L', 14)
    ws.freeze_panes(1, 0)
    # Conditional formatting for agreement
    ws.conditional_format(1, 6, len(export_df), 6, {
        'type':'cell','criteria':'==','value':'"True"','format':agree_fmt})
    ws.conditional_format(1, 6, len(export_df), 6, {
        'type':'cell','criteria':'==','value':'"False"','format':disagree_fmt})
    print("  ✓ Sheet 1: Full Data")

    # ── Sheet 2: Model Performance ──────────────────────────────────────
    perf_rows = []
    for app in ['bKash','Rocket','Upay','ALL']:
        sub = raw if app == 'ALL' else raw[raw['app']==app]
        for cls in ['Positive','Neutral','Negative']:
            from sklearn.metrics import precision_score, recall_score
            y_true_bin = (sub['rating_sentiment'] == cls).astype(int)
            y_pred_bin = (sub['text_sentiment']   == cls).astype(int)
            perf_rows.append({
                'App': app, 'Class': cls,
                'N_reviews': len(sub),
                'Accuracy':  accuracy_score(sub['rating_sentiment'], sub['text_sentiment']),
                'F1_weighted': f1_score(sub['rating_sentiment'], sub['text_sentiment'], average='weighted'),
                'Precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
                'Recall':    recall_score(y_true_bin, y_pred_bin, zero_division=0),
                'Agreement_pct': sub['agreement'].mean(),
            })
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_excel(writer, sheet_name='Model Performance', index=False)
    ws2 = writer.sheets['Model Performance']
    for c, h in enumerate(perf_df.columns):
        ws2.write(0, c, h, hdr_fmt)
    ws2.set_column('A:B', 14); ws2.set_column('C:C', 12); ws2.set_column('D:H', 15)
    ws2.conditional_format(1, 7, len(perf_df), 7, {
        'type':'3_color_scale','min_color':'#5c0011','mid_color':'#78350f','max_color':'#052e16'})
    print("  ✓ Sheet 2: Model Performance")

    # ── Sheet 3: Disagreement Analysis ─────────────────────────────────
    disagree_df = raw[~raw['agreement']].sort_values('thumbsUpCount', ascending=False)
    disagree_export = disagree_df[['app','content','score','rating_sentiment',
                                   'text_sentiment','disagree_type','topic',
                                   'thumbsUpCount','year']].head(5000)
    disagree_export.to_excel(writer, sheet_name='Disagreements', index=False)
    ws3 = writer.sheets['Disagreements']
    for c, h in enumerate(disagree_export.columns):
        ws3.write(0, c, h, hdr_fmt)
    ws3.set_column('B:B', 70); ws3.set_column('A:A', 9)
    ws3.set_column('C:I', 16); ws3.set_column('H:H', 25)
    ws3.freeze_panes(1, 0)
    print("  ✓ Sheet 3: Disagreements")

    # ── Sheet 4: ABSA Results ───────────────────────────────────────────
    absa_df.to_excel(writer, sheet_name='ABSA Results', index=False)
    ws4 = writer.sheets['ABSA Results']
    for c, h in enumerate(absa_df.columns):
        ws4.write(0, c, h, hdr_fmt)
    ws4.set_column('A:B', 16); ws4.set_column('C:I', 14)
    # Colour the gap columns
    ws4.conditional_format(1, 6, len(absa_df), 6, {
        'type':'3_color_scale','min_color':'#5c0011','mid_color':'#1a1a2e','max_color':'#052e16'})
    ws4.conditional_format(1, 7, len(absa_df), 7, {
        'type':'3_color_scale','min_color':'#052e16','mid_color':'#1a1a2e','max_color':'#5c0011'})
    print("  ✓ Sheet 4: ABSA Results")

    # ── Sheet 5: Monthly Sentiment ──────────────────────────────────────
    monthly_merged = monthly_rating.merge(
        monthly_text[['app','month_dt','text_pos_pct','text_neg_pct']],
        on=['app','month_dt'], how='left'
    )
    monthly_out = monthly_merged[['app','month_dt','total','rating_pos_pct','text_pos_pct','text_neg_pct']].copy()
    monthly_out.columns = ['App','Month','Review Count','Rating Positive %','Text Positive %','Text Negative %']
    monthly_out['Month'] = monthly_out['Month'].dt.strftime('%Y-%m')
    monthly_out.to_excel(writer, sheet_name='Monthly Trend', index=False)
    ws5 = writer.sheets['Monthly Trend']
    for c, h in enumerate(monthly_out.columns):
        ws5.write(0, c, h, hdr_fmt)
    ws5.set_column('A:B', 12); ws5.set_column('C:G', 18)
    ws5.freeze_panes(1, 0)
    print("  ✓ Sheet 5: Monthly Trend")

    # ── Sheet 6: Top Words ──────────────────────────────────────────────
    word_rows = []
    for cls in classes:
        for rank, (word, score) in enumerate(top_features[cls], 1):
            word_rows.append({'Sentiment': cls, 'Rank': rank, 'Word/Phrase': word,
                              'TF-IDF Coefficient': score})
    word_df = pd.DataFrame(word_rows)
    word_df.to_excel(writer, sheet_name='Top Features', index=False)
    ws6 = writer.sheets['Top Features']
    for c, h in enumerate(word_df.columns):
        ws6.write(0, c, h, hdr_fmt)
    ws6.set_column('A:A', 14); ws6.set_column('B:B', 8)
    ws6.set_column('C:C', 28); ws6.set_column('D:D', 20)
    print("  ✓ Sheet 6: Top Features")

    # ── Sheet 7: Executive Summary ──────────────────────────────────────
    ws7 = wb.add_worksheet('Executive Summary')
    ws7.set_tab_color('#E2136E')
    bold_big  = wb.add_format({'bold':True,'font_size':18,'font_color':'#e0e0ff','bg_color':'#0f0f18'})
    bold_med  = wb.add_format({'bold':True,'font_size':12,'font_color':'#c0c0e0','bg_color':'#0f0f18'})
    val_fmt_l = wb.add_format({'font_size':11,'font_color':'#a0a0c0','bg_color':'#0f0f18'})
    kpi_val   = wb.add_format({'bold':True,'font_size':16,'font_color':'#2dd4bf','bg_color':'#0f0f18'})
    
    ws7.set_column('A:A', 35); ws7.set_column('B:D', 20)
    ws7.write('A1', 'MFS App Review — Text Sentiment Analysis', bold_big)
    ws7.write('A2', 'Method: TF-IDF (word 1-2gram + char 3-6gram) + Logistic Regression', val_fmt_l)
    ws7.write('A3', f'Dataset: bKash + Rocket + Upay · {len(raw):,} reviews · 2020–2026', val_fmt_l)
    ws7.write('A5', 'KEY METRICS', bold_med)
    
    kpi_data = [
        ('Total Reviews Analysed', f'{len(raw):,}'),
        ('Overall Accuracy (5-fold CV)', f'{acc*100:.2f}%'),
        ('Weighted F1 Score', f'{f1:.4f}'),
        ('Agreement Rate (text vs rating)', f'{total_agree:.1f}%'),
        ('Disagreement Rate', f'{100-total_agree:.1f}%'),
        ('Bengali-script reviews', f'{bengali_pct:.1f}%'),
        ('Feature dimensions', f'{X.shape[1]:,}'),
    ]
    for i, (label, value) in enumerate(kpi_data, 6):
        ws7.write(f'A{i}', label, val_fmt_l)
        ws7.write(f'B{i}', value, kpi_val)
    
    ws7.write('A14', 'ACCURACY BY APP', bold_med)
    for i, row in enumerate(app_acc_df.itertuples(), 15):
        ws7.write(f'A{i}', row.app, val_fmt_l)
        ws7.write(f'B{i}', f'{row.accuracy:.1f}%', kpi_val)
        ws7.write(f'C{i}', f'F1: {row.f1:.1f}%', val_fmt_l)
    
    ws7.write('A19', 'TOP DISAGREEMENT TYPES', bold_med)
    for i, (dtype, cnt) in enumerate(disagree_counts.items(), 20):
        ws7.write(f'A{i}', dtype, val_fmt_l)
        ws7.write(f'B{i}', f'{cnt:,} ({cnt/len(raw)*100:.1f}%)', kpi_val)
    
    print("  ✓ Sheet 7: Executive Summary")

print(f"\n  ✓ Excel file saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PIPELINE COMPLETE — SUMMARY")
print("═"*60)
print(f"""
  Dataset         : {len(raw):,} reviews (bKash + Rocket + Upay)
  Features        : {X.shape[1]:,} (word TF-IDF + char TF-IDF)
  Model           : Logistic Regression (multinomial, saga solver)
  Validation      : 5-Fold Stratified Cross-Validation

  PERFORMANCE
  ───────────────────────────────────────
  Overall Accuracy: {acc*100:.2f}%
  Weighted F1     : {f1:.4f}
  Agreement Rate  : {total_agree:.1f}%
  Disagreement    : {100-total_agree:.1f}%

  OUTPUTS
  ───────────────────────────────────────
  Excel (7 sheets) : mfs_text_sentiment_analysis.xlsx
  Figure 1         : fig1_model_performance.png
  Figure 2         : fig2_monthly_trends.png
  Figure 3         : fig3_absa_heatmap.png
  Figure 4         : fig4_top_words.png
  Figure 5         : fig5_yearly_agreement.png
""")
