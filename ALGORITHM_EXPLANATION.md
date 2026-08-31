# Article Extraction Algorithm - Complete Explanation

## Overview

The algorithm uses **message passing on a graph** with **spatial + semantic signals** to assign each paragraph (text box) to the correct article title.

---

## Step-by-Step Process

### **Step 1: Text Embedding (Semantic Understanding)**

**Model:** `paraphrase-multilingual-mpnet-base-v2`
- **Dimensions:** 768
- **Purpose:** Convert Portuguese text to semantic vectors
- **Training:** Pre-trained on paraphrase detection (finds similar/related text)

**For each text box:**
```python
# Clean text
cleaned = remove_hyphenation(text)
          .normalize_whitespace()
          .fix_punctuation()

# Convert to 768-dimensional vector
embedding = model.encode(cleaned)  # [0.234, -0.123, ..., 0.456] (768 numbers)
```

---

### **Step 2: Title Affinity Computation (Which Title Does Each Box Belong To?)**

**This answers your question!** For each text box, we compute affinity to ALL titles using:

**40% Semantic Similarity:**
```python
semantic_score = cosine_similarity(text_embedding, title_embedding)
# Measures topic relatedness
# Example: "Grémio text" vs "Grémio title" → 0.65
#          "Grémio text" vs "Funcionalismo title" → 0.30
```

**60% Spatial Position (NEW - Just Added!):**
```python
# Is box directly below title?
vertical_gap = text_box.y1 - title.y2

if box is BELOW title:
    if vertical_gap < 100px AND horizontally_aligned:
        spatial_score = 1.0  # Directly below = perfect!
    elif vertical_gap < 200px:
        spatial_score = 0.9 - (gap / 400)  # Close = high
    elif vertical_gap < 400px:
        spatial_score = 0.7 - (gap / 800)  # Medium
    else:
        spatial_score = 0.5 - (distance / 1000)  # Far
else:
    spatial_score = 0.1  # Above title = unlikely

# COMBINE
affinity = 0.4 * semantic + 0.6 * spatial
```

**Result:** Each text box knows its **best-matching title**
```python
title_affinity[box_10] = (Title_1_Grémio, score=0.707)
title_affinity[box_15] = (Title_0_Funcionalismo, score=0.498)
```

---

### **Step 3: Graph Construction (Which Boxes Can Connect?)**

**Build edges between boxes** using 4 signals:

**For Text↔Text connections:**
```python
edge_weight = (
    0.4 * spatial_proximity +         # How close geometrically?
    0.3 * full_text_similarity +      # Same topic (full paragraph embeddings)?
    0.3 * continuation_score          # Text flows naturally (last sentence A → first sentence B)?
) × title_affinity_multiplier

# Title affinity multiplier:
if box_a and box_b both prefer SAME title:
    multiplier = 1.8  # Boost - likely same article
elif box_a prefers Title X, box_b prefers Title Y (X ≠ Y):
    if both have score > 0.5:
        multiplier = 0.05  # Heavy penalty - different articles
```

**Example edges:**
```
Box 10 (Grémio text) → Box 11 (Grémio text):
  - Both prefer Title 1 (Grémio)
  - Edge weight = (0.4*0.8 + 0.3*0.7 + 0.3*0.75) * 1.8 = 1.17 (strong connection!)

Box 10 (Grémio text) → Box 5 (Funcionalismo text):
  - Box 10 prefers Title 1, Box 5 prefers Title 0
  - Edge weight = (0.4*0.6 + 0.3*0.5 + 0.3*0.4) * 0.05 = 0.025 (blocked!)
```

---

### **Step 4: Message Passing (Assign Boxes to Articles)**

**Phase 1: Initialize**
```python
Article 0 = Title 0 (Funcionalismo)
Article 1 = Title 1 (Grémio)
```

**Phase 2: Iterative Label Propagation**

```python
for iteration in 1..10:
    for each unassigned box:
        # STRONG OVERRIDE (NEW)
        if box has affinity > 0.50 to Title X:
            # Box can ONLY join Title X's article
            # Wait for neighbors in that article to be assigned
            if Article X has assigned neighbors:
                join Article X

        # Otherwise, use voting
        else:
            # Collect votes from assigned neighbors
            for neighbor in box.neighbors:
                if neighbor is in Article Y:
                    vote = edge_weight(box, neighbor)

                    # Apply title affinity penalty
                    if box prefers different title than Article Y:
                        vote *= penalty  # or block completely

                    votes[Article Y] += vote

            # Join article with highest vote
            if max(votes) >= 0.5:
                join best_article
```

**Phase 3: Spatial Bounds Refinement**
```python
# For each assigned box, check if it fits article's spatial extent
bounds = get_article_bounds(article)  # top, bottom, left, right

if box is outside bounds (+ 500px tolerance):
    # Re-vote with title affinity filtering
    reassign if better match found
```

**Phase 4: Orphan Handling**
```python
# Boxes with no assignment → assign to nearest article
```

---

## Current Results (5 Pages Tested)

| Page | Titles | Articles | Ratio | Clean | Contaminated | Notes |
|------|--------|----------|-------|-------|--------------|-------|
| 001 | 5 | 5 | 1.00:1 | 5 | 0 | ✅ Perfect |
| 002 | 2 | 2 | 1.00:1 | 1 | 1 | ❌ Article 1 has both topics |
| 003 | 3 | 3 | 1.00:1 | 2 | 1 | ⚠️  Large article (32p, multiple topics) |
| 004 | 5 | 5 | 1.00:1 | 5 | 0 | ✅ Perfect |
| 005 | 3 | 3 | 1.00:1 | 3 | 0 | ✅ Perfect |

**Overall:** 16 titles → 16 articles (1.00:1 ratio) ✅
**Contamination:** 2/16 = 12.5%
**Reading Order:** ✅ Correct on all pages

---

## Page 002 Issue

**Problem:** Article 1 (Funcionalismo) is absorbing some boxes that should belong to Article 2 (Grémio).

**Diagnosis:**
- Title affinity computation works (4 boxes prefer Grémio)
- But message passing still assigns them to Funcionalismo
- Possible causes:
  1. Grémio boxes have more/stronger neighbors in Funcionalismo article
  2. Spatial layout makes Grémio boxes closer to Funcionalismo boxes
  3. Title affinity penalty not strong enough during voting

**Current constraints:**
- ✅ Spatial affinity (boxes below title = high affinity)
- ✅ Title affinity override (affinity > 0.50 → can only join that title's article)
- ✅ Blocked voting (if box prefers Title X, blocks votes from Article Y)

---

## How to Visualize (As You Requested)

### **Pages 1 and 3:**
```bash
source .venv/bin/activate

# Page 1 (5 clean articles)
python visualize_articles.py data/1974/04/24/page_001.png visualizations/page_001_final.png

# Page 3 (3 articles)
python visualize_articles.py data/1974/04/24/page_003.png visualizations/page_003_final.png

# Page 2 (to see the contamination issue)
python visualize_articles.py data/1974/04/24/page_002.png visualizations/page_002_debug.png
```

**What you'll see:**
- Each article in unique color
- Boxes labeled "Art 001", "Art 002", etc.
- Legend showing article IDs and titles
- **On page 002:** You'll see some boxes labeled "Art 001" (Funcionalismo) that should be "Art 002" (Grémio)

### **Check Reading Order:**
```bash
python test_reading_order.py
```

Shows coherent text flow and topic purity for each article.

---

## Parameters You Can Tune

| Parameter | Location | Current | Effect |
|-----------|----------|---------|--------|
| **Spatial weight in title affinity** | Line 474 | 0.6 (60%) | Higher = position matters more |
| **Semantic weight in title affinity** | Line 474 | 0.4 (40%) | Higher = topic matters more |
| **Strong override threshold** | Line 1119 | 0.50 | Lower = more boxes get direct assignment |
| **Vote blocking threshold** | Line 1055 | 0.55 | Lower = blocks more cross-article votes |
| **Title affinity boost (same title)** | Line 631 | 1.8× | Higher = stronger same-article connections |
| **Title affinity penalty (diff title)** | Line 636 | 0.05× | Lower = stronger separation |

---

## Recommendations for Page 002 Fix

**Option A: Increase spatial weight in title affinity**
- Change line 474: `0.4 * semantic + 0.6 * spatial` → `0.3 * semantic + 0.7 * spatial`
- Makes position dominate even more

**Option B: Lower strong override threshold**
- Change line 1119: `affinity_score > 0.50` → `affinity_score > 0.45`
- More boxes get direct assignment to their best title

**Option C: Visualize and manually inspect**
- See actual spatial layout
- Understand why Grémio boxes are near Funcionalismo boxes
- Adjust spatial rules based on layout

**Option D: Accept 12.5% contamination**
- 14/16 articles are perfect (87.5% success)
- Page 002 might have unusual layout
- Focus on scaling to more pages

Which approach would you prefer?
