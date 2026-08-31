# Article Extraction Research Plan
## Diário de Lisboa OCR Processing

**Date:** 2026-01-10
**Status:** Research Planning Phase
**Goal:** Transform OCR-detected layout elements (paragraphs, titles, images) into coherent, searchable articles

---

## Executive Summary

This research plan addresses the challenge of extracting structured articles from OCR-processed newspaper pages. The Diário de Lisboa archive (1921-1990) contains ~150,000 pages processed with DOTS OCR, producing bounding boxes with categories (Title, Text, Section-header, Picture, Caption, etc.). The goal is to:

1. **Connect related elements** (paragraphs, titles, images) into single articles
2. **Track cross-page continuations** (articles spanning multiple pages)
3. **Detect and fix OCR errors** (empty boxes, overlaps, incorrect text/sizes)

This will enable building a searchable article database for historical research.

---

## Current State Analysis

### OCR Output Structure

Each processed page generates a JSON file with:
```json
{
  "image_path": "data/1974/04/24/page_001.png",
  "ocr_result": "[{\"bbox\": [x1,y1,x2,y2], \"category\": \"Title\", \"text\": \"...\"}, ...]",
  "timestamp": "2025-12-13T11:21:31.780722",
  "model": "rednote-hilab/dots.ocr",
  "status": "success"
}
```

**Bounding Box Elements:**
- `bbox`: [x1, y1, x2, y2] coordinates in pixels
- `category`: Layout element type (12 categories)
- `text`: Extracted Portuguese text (empty for Pictures)

### Layout Element Categories (from 50 sample files)

| Category | Count | % of Total | Purpose |
|----------|-------|------------|---------|
| Text | 1,852 | 73.2% | Article paragraphs, body text |
| Section-header | 364 | 14.4% | Article sub-headings |
| Page-header | 150 | 5.9% | Page number, date, newspaper name |
| Picture | 84 | 3.3% | Photographs, illustrations |
| List-item | 58 | 2.3% | Bullet points, numbered lists |
| Title | 34 | 1.3% | Main article headlines |
| Table | 20 | 0.8% | Tabular data |
| Caption | 19 | 0.8% | Image captions |
| Page-footer | 1 | <0.1% | Footer information |

### Newspaper Layout Characteristics

**Physical Layout:**
- **Columns:** 3-4 columns per page
- **Resolution:** ~2870×3832 pixels (typical PNG)
- **Reading order:** Top-to-bottom, left-to-right within columns
- **Column flow:** Articles can span multiple columns on same page

**Article Patterns Observed:**
1. **Single-column article:** Title + paragraphs in one column
2. **Multi-column article:** Article flows across 2-3 columns on same page
3. **Continued articles:** Explicit "Continua na pág. X" markers (12% of pages)
4. **Mixed content:** Articles interspersed with images, advertisements, tables

**Example from Page 1 (1974-04-24):**
- Large masthead "Diário de Lisboa" (Title category)
- Main headline article with large picture
- Multiple 2-3 column articles
- Continuation markers: "Continua na pág. 10", "Continua na pág. 4"

### OCR Quality Issues (from analysis of 50 files)

| Issue Type | Prevalence | Impact |
|------------|------------|--------|
| **Empty text boxes** | 14% of files | Missing text in non-Picture categories |
| **Overlapping boxes** | 6% of files | Duplicate or incorrect bbox detection |
| **100% overlap** | Present | Complete bbox duplication (OCR error) |
| **Continuation markers** | 12% of files | Explicit cross-page article linking |
| **Unknown category** | <1% | OCR categorization failure |

**Specific Issues Identified:**

1. **Empty Text Boxes:**
   - Example: `{"bbox": [995, 626, 1405, 742], "category": "Text", "text": ""}`
   - Possible causes: OCR failure, image-only regions misclassified
   - Impact: Missing article content

2. **Overlapping Bboxes:**
   - Example: Box 33 (Text) overlaps Box 34 (Text) by 100%
   - Possible causes: Double detection, column boundary confusion
   - Impact: Duplicate text, article segmentation errors

3. **Malformed Bboxes:**
   - Example: `{"bbox": [2327, 2353, 2731], ...}` (only 3 coordinates)
   - Impact: Cannot process geometry

4. **Large Bboxes:**
   - 4% of files have boxes >7.7M pixels (~70% of page)
   - Possible cause: Failed column segmentation
   - Impact: Entire page treated as single element

---

## Research Objective 1: Article Segmentation & Assembly

**Goal:** Connect related layout elements (titles, paragraphs, images, captions) into coherent articles

### Challenge Description

Newspapers have complex layouts where:
- **Articles span columns:** A single article may flow across 2-3 columns
- **Column precedence:** Reading order is top-to-bottom within column, then next column
- **Visual hierarchy:** Titles, section headers, and paragraphs form nested structure
- **Mixed content:** Articles interrupted by ads, other articles, images

**Key Questions:**
1. How to determine which Text boxes belong to the same article?
2. How to associate Titles/Section-headers with their article content?
3. How to link Pictures/Captions to their parent article?
4. How to handle advertisements vs. editorial content?

### Proposed Research Approaches

#### Approach 1: Column-Based Reading Order

**Hypothesis:** Articles follow column-wise reading order with geometric proximity

**Method:**
1. **Detect columns:** Cluster bboxes by X-coordinate into vertical columns
2. **Sort within column:** Order by Y-coordinate (top-to-bottom)
3. **Identify article boundaries:**
   - New Title → Start new article
   - Large vertical gap → Potential article boundary
   - Section-header → Sub-section of current article
4. **Merge across columns:** If article reaches column bottom, check top of next column

**Pros:**
- Simple geometric approach
- Aligns with natural reading order
- Works for standard multi-column layouts

**Cons:**
- Fails on complex layouts (mixed column widths)
- Ads/images disrupt column flow
- Cannot handle articles that "jump" columns

**Research Tasks:**
- [ ] Implement column detection algorithm (X-coordinate clustering)
- [ ] Test on 20 sample pages with varied layouts
- [ ] Measure accuracy: manual annotation vs. algorithm
- [ ] Document failure cases

#### Approach 2: Hierarchical Clustering (XY-Cut Algorithm)

**Hypothesis:** Recursive spatial partitioning mirrors visual layout

**Method:**
1. **Recursive XY-cut:**
   - Find largest horizontal/vertical whitespace
   - Split page into regions
   - Recursively subdivide until homogeneous
2. **Build hierarchy tree:** Regions → Articles → Paragraphs
3. **Reading order traversal:** DFS/BFS to extract articles

**Pros:**
- Handles complex layouts
- Naturally creates hierarchy (article → sections → paragraphs)
- Well-studied algorithm (document layout analysis)

**Cons:**
- Sensitive to whitespace detection threshold
- May over-segment (split single article into multiple)
- Requires tuning for newspaper-specific layouts

**Research Tasks:**
- [ ] Implement XY-cut algorithm
- [ ] Tune whitespace threshold on 10 sample pages
- [ ] Compare against column-based approach
- [ ] Evaluate on pages with irregular layouts

#### Approach 3: Machine Learning (Layout Analysis Model)

**Hypothesis:** Fine-tuned vision model can learn article boundaries

**Method:**
1. **Training data creation:**
   - Manually annotate 100-200 pages with article boundaries
   - Label each bbox with article_id
2. **Model options:**
   - Fine-tune LayoutLMv3 or similar document AI model
   - Train CNN classifier on bbox features (position, size, category, text)
3. **Inference:** Model predicts article_id for each bbox

**Pros:**
- Handles arbitrary layout complexity
- Learns implicit rules (e.g., "articles under same title belong together")
- Can incorporate text semantics (not just geometry)

**Cons:**
- Requires significant annotation effort (100-200 pages)
- Training infrastructure needed
- Risk of overfitting to 1970s newspaper style

**Research Tasks:**
- [ ] Annotate 20 pages manually (pilot dataset)
- [ ] Evaluate annotation difficulty and time required
- [ ] Research existing document layout models (LayoutLMv3, VILA)
- [ ] Compare effort vs. accuracy gain over geometric methods

#### Approach 4: Graph-Based Connectivity

**Hypothesis:** Bboxes form a graph where edges represent "belongs to same article"

**Method:**
1. **Build graph:** Nodes = bboxes, Edges = spatial/semantic similarity
2. **Edge features:**
   - Geometric: proximity, alignment, column membership
   - Semantic: category compatibility (Title → Text, Picture → Caption)
   - Textual: continuation markers, topic similarity
3. **Community detection:** Find connected components (articles)

**Pros:**
- Flexible framework (can combine multiple signals)
- Naturally handles multi-column flow
- Explicitly models relationships

**Cons:**
- Requires careful edge weight tuning
- Complex implementation
- May need labeled data to learn edge weights

**Research Tasks:**
- [ ] Design graph schema (node/edge features)
- [ ] Implement graph construction
- [ ] Test community detection algorithms (Louvain, label propagation)
- [ ] Compare with simpler geometric methods

### Recommended Initial Approach

**Start with Approach 1 (Column-Based) + Approach 4 (Graph enhancement):**

1. **Phase 1 (Week 1):** Implement column detection + reading order
2. **Phase 2 (Week 2):** Add graph edges for special cases (continuation markers, cross-column titles)
3. **Phase 3 (Week 3):** Evaluate on 50 annotated pages
4. **Phase 4 (Week 4):** Iterate based on error analysis

**Rationale:** Simple baseline first, incrementally add sophistication. Avoid ML overhead unless geometric methods fail.

---

## Research Objective 2: Cross-Page Article Tracking

**Goal:** Detect and link articles that continue across multiple pages

### Challenge Description

**Explicit Continuation Markers (12% of pages):**
- Pattern: "Continua na pág. X" at end of text block
- Example: "Continua na pág. 10" → Article resumes on page 10
- Challenge: Find resumption point on target page

**Implicit Continuations (unknown frequency):**
- No explicit marker, but article continues
- Common in long feature articles
- Challenge: Detect semantic/topic continuity

### Current Evidence

From sample analysis:
- **Explicit markers found:**
  - "Continua na pág. 20" (pages 1 of multiple dates)
  - "Continua na pág. 4" (page 3)
  - "Continua na pág. 18" (page 9)

- **Marker placement:** Typically last Text box in article
- **Format variations:** "Continua na pág.", "Continua na página", "Cont. pág."

### Proposed Research Approaches

#### Approach 1: Regex-Based Marker Detection

**Method:**
1. **Extract markers:**
   ```python
   pattern = r'continua\s+na\s+p[aá]g(?:ina)?\.?\s+(\d+)'
   ```
2. **For each marker:**
   - Record: source_page, source_bbox, target_page
   - Find article on target page (look for "Vem da pág. X" or matching title)
3. **Link articles:** Append target page text to source article

**Pros:**
- Simple and fast
- Works for ~12% of cases (explicit markers)

**Cons:**
- Misses implicit continuations
- Requires finding resumption point on target page

**Research Tasks:**
- [ ] Build regex patterns for all marker variations
- [ ] Extract all continuations from 50 sample pages
- [ ] Manually verify target page matching
- [ ] Document cases where resumption is ambiguous

#### Approach 2: Title Matching Across Pages

**Hypothesis:** Continued articles repeat title (or shortened version) on new page

**Method:**
1. **Extract titles:** All Title category boxes
2. **Fuzzy match:** Compare titles across consecutive/nearby pages
   - Use edit distance or semantic similarity
3. **If match:** Treat as continuation

**Pros:**
- Catches implicit continuations
- No reliance on explicit markers

**Cons:**
- False positives if multiple articles share similar titles
- Requires tuning similarity threshold

**Research Tasks:**
- [ ] Extract all titles from 100 pages
- [ ] Measure title repetition frequency
- [ ] Test fuzzy matching (Levenshtein, TF-IDF cosine similarity)
- [ ] Evaluate precision/recall

#### Approach 3: Semantic Continuity Detection

**Hypothesis:** Continued articles maintain topic/vocabulary consistency

**Method:**
1. **Embed text:** Use sentence embeddings (e.g., multilingual BERT for Portuguese)
2. **Compute similarity:** Last paragraph of page N vs. first paragraph of page N+1
3. **High similarity:** Likely continuation

**Pros:**
- Catches all continuations (explicit + implicit)
- No pattern matching needed

**Cons:**
- Computationally expensive
- May link unrelated but topically similar articles

**Research Tasks:**
- [ ] Select Portuguese embedding model (BERTimbau, mBERT)
- [ ] Compute embeddings for 50 pages
- [ ] Test similarity threshold on known continuations
- [ ] Measure false positive rate

### Recommended Initial Approach

**Phase 1:** Regex-based marker detection (covers 12% of cases, high precision)
**Phase 2:** Title matching (extends coverage, medium complexity)
**Phase 3:** Semantic analysis (if first two approaches have low coverage)

**Evaluation:**
- Manually annotate 50 pages with continuation ground truth
- Measure precision/recall for each approach
- Combine approaches for maximum coverage

---

## Research Objective 3: OCR Error Detection & Correction

**Goal:** Identify and fix OCR quality issues affecting article extraction

### Issue 1: Empty Text Boxes (14% of files)

**Problem:** Bboxes marked as Text/Section-header but with empty `text` field

**Observed Examples:**
```json
{"bbox": [995, 626, 1405, 742], "category": "Text", "text": ""}
{"bbox": [2322, 3294, 2738, 3384], "category": "Text", "text": ""}
```

**Possible Causes:**
1. **OCR failure:** Model detected layout but failed to extract text
2. **Misclassification:** Picture/Table region misclassified as Text
3. **Low contrast:** Faded/poor quality scan

**Proposed Solutions:**

#### Solution 1A: Re-run OCR on Empty Boxes

**Method:**
1. Crop bbox region from original image
2. Re-run OCR with different parameters (higher sensitivity)
3. If still empty, downgrade category to "Unknown" or merge with adjacent box

**Research Tasks:**
- [ ] Extract empty box regions from 10 sample pages
- [ ] Manually inspect: are they truly empty or OCR failures?
- [ ] Test re-OCR with different DOTS parameters
- [ ] Measure recovery rate

#### Solution 1B: Visual Classification

**Method:**
1. Use vision model to classify empty box: Text vs. Picture vs. Noise
2. If Picture: reclassify, don't expect text
3. If Text: flag for manual review or re-OCR

**Research Tasks:**
- [ ] Collect examples of empty boxes with visual context
- [ ] Train/fine-tune classifier (CLIP, ViT)
- [ ] Evaluate reclassification accuracy

### Issue 2: Overlapping Bounding Boxes (6% of files)

**Problem:** Multiple boxes with >30% area overlap (some 100% duplicates)

**Observed Examples:**
```
Box 33 (Text) overlaps Box 34 (Text) by 100%
Box 27 (Picture) overlaps Box 33 (List-item) by 100%
```

**Possible Causes:**
1. **Double detection:** OCR model detected same region twice
2. **Column boundary errors:** Boxes spanning column gaps
3. **Hierarchical elements:** Section-header inside larger Text block

**Proposed Solutions:**

#### Solution 2A: Merge Duplicate Boxes

**Method:**
1. Detect overlaps >80% (near-duplicates)
2. Merge strategy:
   - Keep higher-confidence box (if scores available)
   - Or keep more specific category (Title > Text)
   - Or merge text fields (concatenate)

**Research Tasks:**
- [ ] Analyze overlap patterns: which categories overlap most?
- [ ] Define merge rules (priority: Title > Section-header > Text)
- [ ] Test on 10 pages with overlaps
- [ ] Measure impact on article extraction

#### Solution 2B: Spatial NMS (Non-Maximum Suppression)

**Method:**
1. Apply NMS algorithm (from object detection):
   - Sort boxes by confidence/size
   - Remove boxes overlapping >threshold with higher-ranked boxes
2. Preserve hierarchical relationships (allow Title to contain Text)

**Research Tasks:**
- [ ] Implement NMS with category-aware rules
- [ ] Tune overlap threshold (30%? 50%? 80%?)
- [ ] Evaluate on overlapping cases

### Issue 3: Malformed Bounding Boxes

**Problem:** Bboxes with incorrect number of coordinates or invalid geometry

**Observed Examples:**
```json
{"bbox": [2327, 2353, 2731], ...}  // Only 3 coordinates
{"bbox": [x1, y1, x2, y2], ...} where x1 > x2  // Inverted
```

**Proposed Solutions:**

**Method:**
1. **Validation pass:** Check all bboxes for:
   - Exactly 4 coordinates
   - x1 < x2, y1 < y2 (fix by swapping if needed)
   - Within image bounds
2. **Reject invalid:** Remove or flag for manual review

**Research Tasks:**
- [ ] Count malformed bbox frequency
- [ ] Implement validation + auto-correction
- [ ] Log rejected boxes for analysis

### Issue 4: Incorrect Text / Garbled OCR

**Problem:** Text field contains nonsense or incorrect characters

**Detection Methods:**
1. **Language model perplexity:** High perplexity → likely garbled
2. **Dictionary check:** High % of unknown words → likely errors
3. **Character distribution:** Excessive special characters

**Proposed Solutions:**

**Method:**
1. Flag suspicious text for review
2. Optionally re-OCR flagged boxes
3. Build correction model (e.g., spell-checker for Portuguese)

**Research Tasks:**
- [ ] Sample 100 text boxes, manually label quality
- [ ] Test perplexity-based detection (use Portuguese LM)
- [ ] Evaluate false positive rate

### Recommended Prioritization

**High Priority (Week 1-2):**
1. ✅ Empty text box analysis (Solution 1A: re-inspect + flag)
2. ✅ Overlapping box merging (Solution 2A: deduplication)
3. ✅ Malformed bbox validation

**Medium Priority (Week 3-4):**
4. Title matching for continuations
5. Incorrect text detection (sample-based)

**Low Priority (Future):**
6. Visual classification for empty boxes (requires training)
7. Semantic continuity detection (high complexity)

---

## Implementation Plan

### Phase 1: Data Exploration & Baseline (Week 1)

**Objectives:**
- Understand OCR output patterns across 100-200 sample pages
- Build baseline article extraction (column-based)
- Document common layout patterns

**Deliverables:**
1. **Annotated sample set:** 50 pages manually labeled with article boundaries
2. **Baseline extractor:** Python script using column detection
3. **Analysis report:** Layout pattern frequency, OCR error rates

**Tasks:**
- [ ] Manually annotate 50 pages (article boundaries, continuations)
- [ ] Implement column detection algorithm
- [ ] Implement reading order sort
- [ ] Evaluate baseline accuracy vs. manual annotations

### Phase 2: Article Segmentation Refinement (Week 2)

**Objectives:**
- Improve article boundary detection
- Handle multi-column articles
- Associate images/captions with articles

**Deliverables:**
1. **Enhanced extractor:** Hierarchical clustering or graph-based
2. **Evaluation metrics:** Precision/recall on article boundaries
3. **Error analysis:** Document failure modes

**Tasks:**
- [ ] Implement XY-cut or graph-based approach
- [ ] Test on 100 pages
- [ ] Compare with baseline
- [ ] Iterate based on errors

### Phase 3: Cross-Page Linking (Week 3)

**Objectives:**
- Detect explicit continuation markers
- Link articles across pages
- Build multi-page article database

**Deliverables:**
1. **Continuation detector:** Regex + target page matching
2. **Linked article database:** SQLite or JSON with cross-page IDs
3. **Validation report:** Manual check of 50 continuations

**Tasks:**
- [ ] Build continuation regex library
- [ ] Extract all continuations from 500 pages
- [ ] Implement target page matching
- [ ] Manually validate 50 random continuations

### Phase 4: OCR Error Handling (Week 4)

**Objectives:**
- Clean OCR outputs
- Fix overlaps, empty boxes, malformed bboxes
- Re-run article extraction on cleaned data

**Deliverables:**
1. **OCR cleaner script:** Automated error detection + correction
2. **Cleaned OCR outputs:** New JSON files with fixes
3. **Quality report:** Before/after error rates

**Tasks:**
- [ ] Implement bbox validation
- [ ] Implement overlap detection + merging
- [ ] Re-inspect empty boxes
- [ ] Re-run article extraction on cleaned data

### Phase 5: Evaluation & Iteration (Week 5)

**Objectives:**
- Comprehensive evaluation on large sample (500+ pages)
- Build search index prototype
- Document findings

**Deliverables:**
1. **Final extraction pipeline:** End-to-end script (OCR → articles)
2. **Article database:** SQLite with ~10,000 articles (50 newspapers × 200 articles/issue)
3. **Search prototype:** Basic keyword search interface
4. **Research report:** Methodology, results, limitations

**Tasks:**
- [ ] Process 500 pages end-to-end
- [ ] Build SQLite database with articles
- [ ] Implement basic search (FTS5 full-text search)
- [ ] Write final report

---

## Evaluation Metrics

### Article Segmentation Quality

**Metrics:**
1. **Boundary Precision:** % of predicted boundaries that match ground truth
2. **Boundary Recall:** % of true boundaries detected
3. **Article Completeness:** % of article text captured (no missing paragraphs)
4. **Contamination Rate:** % of articles containing text from other articles

**Evaluation Method:**
- Manual annotation of 50 pages
- Compare predicted article boundaries with ground truth
- Report precision/recall/F1

### Cross-Page Linking Quality

**Metrics:**
1. **Continuation Detection Recall:** % of true continuations detected
2. **Continuation Precision:** % of detected continuations that are correct
3. **Target Page Accuracy:** % where continuation correctly identifies resumption page

**Evaluation Method:**
- Manual annotation of 50 pages with continuations
- Compare predicted links with ground truth

### OCR Error Correction Quality

**Metrics:**
1. **Error Detection Recall:** % of true errors detected
2. **Error Detection Precision:** % of flagged boxes that are actual errors
3. **Correction Success Rate:** % of errors successfully fixed

**Evaluation Method:**
- Manual review of 100 boxes flagged as errors
- Verify fixes on 50 re-OCRed boxes

---

## Tools & Infrastructure

### Required Software

**Python Libraries:**
- `json` - Parse OCR outputs
- `Pillow (PIL)` - Image processing
- `scikit-learn` - Clustering (for column detection)
- `networkx` - Graph algorithms (for graph-based approach)
- `regex` - Advanced pattern matching (continuation markers)
- `transformers` - BERT embeddings (for semantic similarity)
- `sqlite3` - Article database
- `numpy` - Numeric operations

**Optional:**
- `spacy` or `stanza` - Portuguese NLP (for text analysis)
- `layoutparser` - Document layout library
- `sentence-transformers` - Multilingual embeddings

### Development Environment

**Hardware:**
- CPU: Any modern processor (no GPU needed for baseline)
- RAM: 16 GB (for processing large batches)
- Storage: 50 GB (for OCR outputs + article database)

**Software:**
- Python 3.10+
- Jupyter notebooks (for exploration)
- Git (version control)

---

## Risk & Mitigation

### Risk 1: Layout Complexity Exceeds Geometric Methods

**Risk:** Newspaper layouts too irregular for column/XY-cut algorithms
**Likelihood:** Medium
**Impact:** High (baseline approaches fail)

**Mitigation:**
- Early evaluation on diverse page samples (front page, classified ads, feature articles)
- Fallback to ML-based approach if geometric methods <70% accuracy
- Hybrid approach: geometric + ML for edge cases

### Risk 2: Insufficient Continuation Coverage

**Risk:** Many articles continue without explicit markers
**Likelihood:** Medium (unknown frequency)
**Impact:** Medium (incomplete articles)

**Mitigation:**
- Measure implicit continuation frequency on sample
- Implement title matching + semantic similarity if >20% implicit
- Accept some incompleteness in initial version, iterate later

### Risk 3: Annotation Effort Underestimated

**Risk:** Manual annotation too time-consuming
**Likelihood:** High
**Impact:** Medium (delays evaluation)

**Mitigation:**
- Start with small sample (20 pages) to estimate effort
- Build annotation tool to speed up process
- Recruit help or accept smaller evaluation set

### Risk 4: OCR Errors Too Severe

**Risk:** >50% of text boxes have errors, undermining extraction
**Likelihood:** Low (DOTS OCR is high-quality)
**Impact:** High (entire pipeline fails)

**Mitigation:**
- Sample-based error rate measurement in Phase 1
- Re-OCR problematic pages if error rate >30%
- Focus on high-quality pages for initial prototype

---

## Success Criteria

### Minimum Viable Outcome (MVP)

**After 5 weeks:**
1. ✅ **Article extraction working on 70%+ of pages** (standard layouts)
2. ✅ **Explicit continuations detected** (12% of pages)
3. ✅ **OCR errors reduced by 50%** (overlaps, empty boxes)
4. ✅ **Search prototype functional** (keyword search over 500 pages)

### Stretch Goals

**If time permits:**
1. **Implicit continuation detection** (semantic similarity)
2. **ML-based layout analysis** (LayoutLM fine-tuning)
3. **Full archive processing** (150,000 pages → 500,000+ articles)
4. **Web interface** (browse/search historical articles)

---

## Next Steps

**Immediate Actions (This Week):**
1. ✅ Review and approve this research plan
2. ⬜ Manually annotate 20 pilot pages (article boundaries)
3. ⬜ Implement column detection algorithm
4. ⬜ Extract continuation markers from 50 pages
5. ⬜ Run OCR error analysis on 100 pages

**First Milestone (End of Week 1):**
- Baseline article extractor working on simple layouts
- Annotated sample set of 50 pages
- Documented error patterns and frequencies

---

## References & Related Work

**Document Layout Analysis:**
- PubLayNet: Large-scale document layout analysis dataset
- LayoutLMv3: Pre-trained model for document AI
- XY-Cut algorithm: Classic geometric layout segmentation

**Newspaper-Specific:**
- Historical newspaper digitization (Europeana project)
- Article segmentation in newspaper archives
- OCR post-correction for historical documents

**Portuguese NLP:**
- BERTimbau: Portuguese BERT model
- spaCy Portuguese models
- Semantic similarity for Portuguese text

---

**Document Version:** 1.0
**Author:** Claude (Research Planning Agent)
**Last Updated:** 2026-01-10
