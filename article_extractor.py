#!/usr/bin/env python3
"""
Article Extraction Module for Diário de Lisboa
Extracts coherent articles from OCR-processed newspaper pages

Features:
1. Column detection (geometric clustering)
2. Reading order sorting (top-to-bottom, left-to-right)
3. Article boundary detection (titles, gaps, continuation markers)
4. Cross-page continuation tracking
"""

import json
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import statistics
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class BBox:
    """Bounding box with layout information"""
    x1: float
    y1: float
    x2: float
    y2: float
    category: str
    text: str
    index: int  # Original position in OCR output

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Article:
    """Extracted article with metadata"""
    article_id: str
    title: str
    text: str
    page: str
    bboxes: List[BBox] = field(default_factory=list)
    images: List[BBox] = field(default_factory=list)
    captions: List[str] = field(default_factory=list)
    continuation_to: Optional[str] = None  # "pág. X" if continues
    continuation_from: Optional[str] = None  # "pág. X" if continued from

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'article_id': self.article_id,
            'title': self.title,
            'text': self.text,
            'page': self.page,
            'num_paragraphs': len(self.bboxes),
            'num_images': len(self.images),
            'captions': self.captions,
            'continuation_to': self.continuation_to,
            'continuation_from': self.continuation_from,
        }


class ColumnDetector:
    """Detects column layout using X-coordinate clustering"""

    def __init__(self, min_column_gap: float = 50):
        """
        Args:
            min_column_gap: Minimum horizontal gap to separate columns (pixels)
        """
        self.min_column_gap = min_column_gap

    def detect_columns(self, bboxes: List[BBox]) -> List[List[BBox]]:
        """
        Cluster bboxes into columns based on X-coordinates

        Returns:
            List of columns, each containing bboxes sorted top-to-bottom
        """
        if not bboxes:
            return []

        # Sort by center X coordinate
        sorted_boxes = sorted(bboxes, key=lambda b: b.center_x)

        # Simple gap-based clustering
        columns = []
        current_column = [sorted_boxes[0]]

        for i in range(1, len(sorted_boxes)):
            prev_box = sorted_boxes[i - 1]
            curr_box = sorted_boxes[i]

            # Check gap between rightmost edge of prev and leftmost edge of current
            gap = curr_box.x1 - prev_box.x2

            if gap > self.min_column_gap:
                # Start new column
                columns.append(current_column)
                current_column = [curr_box]
            else:
                current_column.append(curr_box)

        # Add last column
        if current_column:
            columns.append(current_column)

        # Sort each column top-to-bottom
        for column in columns:
            column.sort(key=lambda b: b.y1)

        return columns


class ContinuationDetector:
    """Detects cross-page article continuations"""

    # Regex patterns for continuation markers
    CONTINUATION_PATTERNS = [
        r'continua\s+na\s+p[aá]g(?:ina)?\.?\s+(\d+)',
        r'cont\.?\s+(?:na\s+)?p[aá]g\.?\s+(\d+)',
        r'segue\s+na\s+p[aá]g(?:ina)?\.?\s+(\d+)',
    ]

    FROM_PATTERNS = [
        r'vem\s+da\s+p[aá]g(?:ina)?\.?\s+(\d+)',
        r'continuação\s+da\s+p[aá]g(?:ina)?\.?\s+(\d+)',
    ]

    def __init__(self):
        self.continuation_regex = [re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PATTERNS]
        self.from_regex = [re.compile(p, re.IGNORECASE) for p in self.FROM_PATTERNS]

    def find_continuation_marker(self, text: str) -> Optional[int]:
        """
        Extract page number from continuation marker

        Args:
            text: Text to search

        Returns:
            Page number if found, None otherwise
        """
        for pattern in self.continuation_regex:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        return None

    def find_from_marker(self, text: str) -> Optional[int]:
        """
        Extract page number from "continued from" marker

        Args:
            text: Text to search

        Returns:
            Page number if found, None otherwise
        """
        for pattern in self.from_regex:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        return None


class ArticleExtractor:
    """Main article extraction engine"""

    def __init__(self,
                 min_column_gap: float = 50,
                 min_article_gap: float = 100,
                 title_categories: List[str] = None):
        """
        Args:
            min_column_gap: Minimum gap to separate columns
            min_article_gap: Minimum vertical gap to separate articles
            title_categories: Categories that indicate article start
        """
        self.column_detector = ColumnDetector(min_column_gap)
        self.continuation_detector = ContinuationDetector()
        self.min_article_gap = min_article_gap
        self.title_categories = title_categories or ['Title', 'Section-header']

        # Load embedding model for semantic similarity
        print("Loading embedding model (first time only)...")
        # Using mpnet for better paraphrase detection and continuation understanding
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.embedding_cache = {}
        print(f"Model loaded successfully ({self.embedding_model.get_sentence_embedding_dimension()} dims)")

    def load_ocr_file(self, file_path: Path) -> List[BBox]:
        """Load OCR JSON file and convert to BBox objects"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data['status'] != 'success':
                return []

            ocr_result = json.loads(data['ocr_result'])

            bboxes = []
            for i, box in enumerate(ocr_result):
                if not isinstance(box, dict):
                    continue

                bbox_coords = box.get('bbox', [])
                if len(bbox_coords) != 4:
                    continue

                bboxes.append(BBox(
                    x1=bbox_coords[0],
                    y1=bbox_coords[1],
                    x2=bbox_coords[2],
                    y2=bbox_coords[3],
                    category=box.get('category', 'Unknown'),
                    text=box.get('text', ''),
                    index=i
                ))

            return bboxes

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def filter_metadata_boxes(self, bboxes: List[BBox]) -> List[BBox]:
        """Remove page headers/footers that are metadata, not content"""
        # Simple heuristic: remove Page-header and Page-footer categories
        return [b for b in bboxes if b.category not in ['Page-header', 'Page-footer']]

    def filter_masthead(self, bboxes: List[BBox], page_height: float = 3832) -> List[BBox]:
        """
        Remove masthead elements (newspaper name, date, issue number)

        Uses heuristics to distinguish masthead metadata from real article titles.

        Args:
            bboxes: List of bounding boxes
            page_height: Typical page height in pixels (default: 3832)

        Returns:
            Filtered list without masthead metadata
        """
        masthead_zone = page_height * 0.12  # Top 460px (~12%)

        # Masthead keywords (Portuguese)
        masthead_keywords = {
            'diário de lisboa', 'fundador', 'director', 'quarta-feira',
            'segunda-feira', 'terça-feira', 'quinta-feira', 'sexta-feira',
            'sábado', 'domingo', 'ano', 'preço', 'n.º'
        }

        filtered = []
        for bbox in bboxes:
            # Keep if below masthead region
            if bbox.y1 > masthead_zone:
                filtered.append(bbox)
                continue

            # In masthead zone - check if it's real content or metadata
            is_masthead = False

            # Check for masthead keywords
            text_lower = bbox.text.lower()
            if any(keyword in text_lower for keyword in masthead_keywords):
                # Contains masthead keywords - likely metadata
                if bbox.category in ['Title', 'Text'] and len(bbox.text) < 100:
                    is_masthead = True

            # Keep substantive content even in masthead zone
            if not is_masthead:
                # Always keep media
                if bbox.category in ['Picture', 'Caption', 'Table']:
                    filtered.append(bbox)
                # Keep long text (real content)
                elif len(bbox.text.strip()) > 100:
                    filtered.append(bbox)
                # Keep Section-headers (almost always real article titles, not masthead)
                elif bbox.category == 'Section-header':
                    filtered.append(bbox)
                # Keep Titles if they look like article titles (longer than newspaper name)
                elif bbox.category == 'Title' and len(bbox.text.strip()) > 20:
                    filtered.append(bbox)

        print(f"Filtered masthead: {len(bboxes)} → {len(filtered)} boxes")
        return filtered

    def euclidean_distance(self, box_a: BBox, box_b: BBox) -> float:
        """Calculate Euclidean distance between box centers"""
        return math.sqrt((box_a.center_x - box_b.center_x)**2 +
                        (box_a.center_y - box_b.center_y)**2)

    def clean_text_for_embedding(self, text: str) -> str:
        """
        Clean text for optimal embedding quality

        Args:
            text: Raw OCR text with newlines, hyphenation

        Returns:
            Cleaned text suitable for embedding
        """
        # Remove hyphenation at line breaks (e.g., "partici-\npam" → "participam")
        text = re.sub(r'-\s*\n\s*', '', text)

        # Normalize whitespace (multiple spaces/newlines → single space)
        text = re.sub(r'\s+', ' ', text)

        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Strip leading/trailing whitespace
        return text.strip()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding with caching

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector (mpnet-base)
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Clean text using enhanced cleaning
        cleaned = self.clean_text_for_embedding(text)

        # Skip if too short
        if len(cleaned) < 3:
            # Return zero vector for very short text
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())

        # Encode
        embedding = self.embedding_model.encode(cleaned, show_progress_bar=False)

        self.embedding_cache[text] = embedding
        return embedding

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            emb_a, emb_b: Embedding vectors

        Returns:
            Similarity from -1 to 1 (usually 0 to 1 for text)
        """
        # Handle zero vectors
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(emb_a, emb_b) / (norm_a * norm_b)

    def get_continuation_score(self, box_a: BBox, box_b: BBox) -> float:
        """
        Detect if box_b continues the text from box_a
        Compares last sentence of A with first sentence of B

        Args:
            box_a, box_b: Adjacent text boxes to compare

        Returns:
            Continuation score 0.0-1.0 (higher = more likely continuation)
        """
        # Split into sentences (handle Portuguese punctuation)
        sents_a = [s.strip() for s in re.split(r'[.!?]+', box_a.text) if s.strip()]
        sents_b = [s.strip() for s in re.split(r'[.!?]+', box_b.text) if s.strip()]

        if not sents_a or not sents_b:
            return 0.5  # Neutral

        # Get boundary sentences
        last_a = self.clean_text_for_embedding(sents_a[-1])
        first_b = self.clean_text_for_embedding(sents_b[0])

        # Need reasonable length for meaningful comparison
        if len(last_a) < 10 or len(first_b) < 10:
            return 0.5  # Neutral

        # Embed boundary sentences
        emb_last = self.embedding_model.encode(last_a, show_progress_bar=False)
        emb_first = self.embedding_model.encode(first_b, show_progress_bar=False)

        # Continuation similarity
        similarity = self.cosine_similarity(emb_last, emb_first)

        return similarity

    def get_title_affinity_score(self, text_box: BBox, title: BBox,
                                 embeddings: Dict[int, np.ndarray]) -> float:
        """
        Compute how related a text box is to a title
        Uses BOTH semantic similarity AND spatial position

        Args:
            text_box: Text box to evaluate
            title: Title box
            embeddings: Pre-computed embeddings

        Returns:
            Affinity score 0.0-1.0 (higher = more related)
        """
        # 1. SEMANTIC COMPONENT
        semantic_score = 0.5  # Default neutral

        if text_box.index in embeddings and title.index in embeddings:
            semantic_score = self.cosine_similarity(embeddings[text_box.index], embeddings[title.index])

        # 2. SPATIAL COMPONENT (NEW!)
        # Text boxes directly below a title are very likely to belong to that article

        distance = self.euclidean_distance(text_box, title)

        # Check if text box is below title (reading order constraint)
        if text_box.y1 < title.y2:
            # Text is ABOVE or overlapping title - unlikely to belong
            spatial_score = 0.1
        else:
            # Text is below title - compute spatial affinity

            # Vertical distance from title bottom to text top
            vertical_gap = text_box.y1 - title.y2

            # Horizontal alignment (how much X-overlap) - CRITICAL!
            x_overlap = min(title.x2, text_box.x2) - max(title.x1, text_box.x1)
            x_overlap_ratio = x_overlap / min(title.width, text_box.width) if min(title.width, text_box.width) > 0 else 0

            # IMPORTANT: If no horizontal overlap, they're in different columns
            # Even if text is "below" title in Y, if X is far apart → NOT related
            if x_overlap_ratio < 0.3:
                # Different columns with poor alignment = very low affinity
                spatial_score = 0.1
            elif vertical_gap < 100 and x_overlap_ratio > 0.5:
                # Directly below with good alignment = very high spatial affinity
                spatial_score = 1.0
            elif vertical_gap < 200 and x_overlap_ratio > 0.5:
                # Close below with good alignment = high affinity
                spatial_score = 0.9 - (vertical_gap / 400)
            elif vertical_gap < 400 and x_overlap_ratio > 0.4:
                # Moderate distance with some alignment = medium affinity
                spatial_score = 0.7 - (vertical_gap / 800)
            else:
                # Far away or poor alignment = low spatial affinity
                spatial_score = 0.1

        # 3. COMBINE SEMANTIC + SPATIAL
        # For title affinity, spatial position is VERY important
        # 40% semantic, 60% spatial (favor position over topic)
        combined = 0.4 * semantic_score + 0.6 * spatial_score

        return combined

    def compute_title_affinity_map(self, bboxes: List[BBox], titles: List[BBox],
                                   embeddings: Dict[int, np.ndarray]) -> Dict[int, Tuple[int, float]]:
        """
        Build map of text boxes to their best-matching titles

        Args:
            bboxes: All bounding boxes
            titles: List of title boxes
            embeddings: Pre-computed embeddings

        Returns:
            Dict mapping text_box_index → (best_title_index, affinity_score)
        """
        title_affinity = {}

        for bbox in bboxes:
            if bbox.category != 'Text' or bbox.index not in embeddings:
                continue

            best_title = None
            best_score = -1

            for title in titles:
                if title.index not in embeddings:
                    continue

                score = self.get_title_affinity_score(bbox, title, embeddings)
                if score > best_score:
                    best_score = score
                    best_title = title

            if best_title is not None:
                title_affinity[bbox.index] = (best_title.index, best_score)

        return title_affinity

    def build_article_graph(self, bboxes: List[BBox], columns: List[List[BBox]]) -> nx.Graph:
        """
        Build graph where edges connect bboxes likely in same article
        Uses BOTH spatial proximity AND semantic similarity

        Args:
            bboxes: List of all bounding boxes
            columns: List of columns (each column is list of bboxes)

        Returns:
            Graph where nodes = bbox indices, edges = same-article relationships
        """
        print(f"Computing embeddings for {len(bboxes)} text boxes...")

        # Compute embeddings for all text boxes
        embeddings = {}
        for bbox in bboxes:
            if bbox.text.strip() and bbox.category not in ['Picture', 'Page-header', 'Page-footer']:
                embeddings[bbox.index] = self.get_embedding(bbox.text)

        print(f"Embedded {len(embeddings)} boxes.")

        # Identify titles and compute title affinities
        titles = [b for b in bboxes if b.category in self.title_categories]
        print(f"Computing title affinities for {len(titles)} titles...")

        title_affinity = self.compute_title_affinity_map(bboxes, titles, embeddings)
        print(f"Computed affinities for {len(title_affinity)} text boxes")

        # Detect parallel titles (side-by-side articles at same Y position)
        parallel_groups = self.detect_parallel_titles(titles)

        # Build graph
        G = nx.Graph()

        # Add all boxes as nodes
        for bbox in bboxes:
            G.add_node(bbox.index, bbox=bbox)

        # Build column map for spatial reasoning
        column_map = {}
        for col_idx, column in enumerate(columns):
            for bbox in column:
                column_map[bbox.index] = col_idx

        # Add edges based on spatial + semantic + continuation + title affinity + parallel blocking
        print("Building graph edges...")
        edges_created = 0
        for i, bbox_i in enumerate(bboxes):
            for j in range(i + 1, len(bboxes)):
                bbox_j = bboxes[j]

                # Skip if too far apart vertically (optimization)
                if abs(bbox_i.center_y - bbox_j.center_y) > 500:
                    continue

                weight = self.compute_edge_weight(bbox_i, bbox_j, column_map, columns,
                                                 embeddings, title_affinity, parallel_groups, titles)

                if weight > 0.3:  # Threshold for edge creation
                    G.add_edge(bbox_i.index, bbox_j.index, weight=weight)
                    edges_created += 1

        print(f"Created {edges_created} edges in graph")

        return G, embeddings

    def compute_edge_weight(self, box_a: BBox, box_b: BBox,
                           column_map: dict, columns: List[List[BBox]],
                           embeddings: Dict[int, np.ndarray],
                           title_affinity: Dict[int, Tuple[int, float]],
                           parallel_groups: List[set],
                           all_titles: List[BBox]) -> float:
        """
        Compute likelihood that two boxes belong to same article
        Uses: spatial + full semantic + continuation + title affinity + parallel blocking

        Args:
            box_a, box_b: Bounding boxes to compare
            column_map: Dict mapping bbox index to column index
            columns: List of columns
            embeddings: Dict mapping bbox index to embedding vector
            title_affinity: Dict mapping text_box_index → (title_index, score)
            parallel_groups: List of sets of parallel title indices
            all_titles: List of all title boxes

        Returns:
            Weight from 0.0 (no connection) to 1.0 (definitely same article)
        """
        # BLOCK connections between parallel articles (side-by-side articles at same Y)
        if (box_a.index in title_affinity and box_b.index in title_affinity and parallel_groups):
            title_a_idx, score_a = title_affinity[box_a.index]
            title_b_idx, score_b = title_affinity[box_b.index]

            # If boxes belong to different titles
            if title_a_idx != title_b_idx:
                # Check if these titles are in the same parallel group
                for group in parallel_groups:
                    if title_a_idx in group and title_b_idx in group:
                        # Parallel articles detected - BLOCK all connections
                        return 0.0  # No connection allowed between side-by-side articles

        # Continue with existing edge weight computation...
        # Handle Picture-Caption connections (spatial only)
        if box_a.category == 'Picture' or box_b.category == 'Picture':
            if ((box_a.category == 'Caption' and box_b.category == 'Picture') or
                (box_a.category == 'Picture' and box_b.category == 'Caption')):
                distance = self.euclidean_distance(box_a, box_b)
                return 0.85 if distance < 150 else 0.0
            return 0.0

        # For Text-Text and Title-Text, use spatial + semantic

        # 1. SPATIAL COMPONENT
        distance = self.euclidean_distance(box_a, box_b)

        # Early exit if too far
        if distance > 1000:
            return 0.0

        # Spatial score: decays from 1.0 at 0px to 0.0 at 600px (CHANGED: was 500px)
        spatial_score = max(0, 1.0 - (distance / 600))

        # Same column boost
        same_column = (box_a.index in column_map and box_b.index in column_map and
                      column_map[box_a.index] == column_map[box_b.index])
        if same_column:
            spatial_score = min(1.0, spatial_score * 1.3)

        # 2. SEMANTIC COMPONENT
        semantic_score = 0.5  # Default neutral

        if (box_a.index in embeddings and box_b.index in embeddings and
            len(box_a.text.strip()) >= 10 and len(box_b.text.strip()) >= 10):
            similarity = self.cosine_similarity(embeddings[box_a.index], embeddings[box_b.index])
            # Cosine similarity is usually 0.0-1.0 for text in same language
            semantic_score = max(0, min(1, similarity))

        # 3. COMBINE SPATIAL + SEMANTIC with category-specific rules

        # Title → Text: STRONG connection to keep title and body together
        if box_a.category in self.title_categories and box_b.category == 'Text':
            # Must be below title
            if box_b.y1 < box_a.y2:
                return 0.0

            # 60% spatial, 40% semantic (CHANGED: was 70/30)
            combined = 0.6 * spatial_score + 0.4 * semantic_score

            # STRONG boost if close (CHANGED: 150→250px, 1.5x→1.8x)
            if distance < 250:
                combined = min(1.0, combined * 1.8)

            # Additional same-column bonus
            if same_column:
                combined = min(1.0, combined * 1.3)

            return combined

        # Text → Text: Use full semantic + continuation + title affinity
        if box_a.category == 'Text' and box_b.category == 'Text':
            # Get full text semantic similarity
            full_semantic = semantic_score  # Already computed above

            # Get continuation score (boundary sentences)
            continuation_score = self.get_continuation_score(box_a, box_b)

            # SPATIAL-FIRST: 70% spatial, 15% semantic, 15% continuation
            # Position dominates over topic similarity
            combined = (0.7 * spatial_score +
                       0.15 * full_semantic +
                       0.15 * continuation_score)

            # Apply title affinity multiplier (STRONG constraint)
            title_affinity_multiplier = 1.0
            if (box_a.index in title_affinity and box_b.index in title_affinity):
                title_a_idx, score_a = title_affinity[box_a.index]
                title_b_idx, score_b = title_affinity[box_b.index]

                if title_a_idx == title_b_idx:
                    # Same title = boost connection significantly
                    title_affinity_multiplier = 1.8  # Increased from 1.5
                elif title_a_idx != title_b_idx:
                    # Different titles = strong penalty to prevent cross-article contamination
                    if score_a > 0.5 and score_b > 0.5:
                        # BOTH have decent affinity to DIFFERENT titles = very heavy penalty
                        title_affinity_multiplier = 0.05  # Increased from 0.1
                    elif score_a > 0.4 or score_b > 0.4:
                        # One has some affinity to different title = heavy penalty
                        title_affinity_multiplier = 0.2  # Decreased from 0.4

            combined *= title_affinity_multiplier

            # Same column AND close vertically boost
            if same_column:
                vertical_gap = min(abs(box_b.y1 - box_a.y2), abs(box_a.y1 - box_b.y2))
                if vertical_gap < 50:
                    combined = min(1.0, combined * 1.2)

            # Penalty if far apart vertically
            vertical_distance = abs(box_a.center_y - box_b.center_y)
            if vertical_distance > 400:
                combined *= 0.6

            return combined

        return 0.0

    def extract_articles_from_graph(self, G: nx.Graph, bboxes: List[BBox],
                                   edge_threshold: float = 0.60) -> List[List[BBox]]:
        """
        Extract articles as connected components in graph

        Args:
            G: Graph with bbox nodes and same-article edges
            bboxes: List of all bboxes (for indexing)
            edge_threshold: Minimum weight to keep edge (lower = more connections)

        Returns:
            List of articles (each article is list of bboxes)
        """
        # Filter edges: keep only strong connections (weight >= edge_threshold)
        # Lower threshold = more connections = fewer, larger articles
        # Higher threshold = fewer connections = more, smaller articles
        strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= edge_threshold]

        print(f"Filtering edges: {len(strong_edges)} edges above threshold {edge_threshold}")

        strong_graph = nx.Graph()
        strong_graph.add_nodes_from(G.nodes())
        strong_graph.add_edges_from(strong_edges)

        # Find connected components (each = one article)
        components = nx.connected_components(strong_graph)

        # Create mapping from bbox.index to BBox object
        bbox_map = {bbox.index: bbox for bbox in bboxes}

        # Convert to lists of BBox objects
        articles = []
        for component in components:
            article_bboxes = [bbox_map[idx] for idx in sorted(component) if idx in bbox_map]
            # Sort by reading order within article (top-to-bottom, left-to-right)
            article_bboxes.sort(key=lambda b: (b.y1, b.x1))
            if article_bboxes:  # Only add non-empty articles
                articles.append(article_bboxes)

        return articles

    def detect_article_boundaries(self, column: List[BBox]) -> List[List[BBox]]:
        """
        Split column into articles based on titles and gaps

        Args:
            column: List of bboxes in reading order

        Returns:
            List of articles (each article is list of bboxes)
        """
        if not column:
            return []

        articles = []
        current_article = []

        for i, bbox in enumerate(column):
            # Start new article if:
            # 1. Current box is a Title
            # 2. Large vertical gap from previous box
            start_new = False

            if bbox.category in self.title_categories:
                start_new = True
            elif current_article and i > 0:
                prev_bbox = column[i - 1]
                gap = bbox.y1 - prev_bbox.y2

                if gap > self.min_article_gap:
                    start_new = True

            if start_new and current_article:
                # Save previous article
                articles.append(current_article)
                current_article = [bbox]
            else:
                current_article.append(bbox)

        # Add last article
        if current_article:
            articles.append(current_article)

        return articles

    def build_article_object(self,
                            bboxes: List[BBox],
                            page_id: str,
                            article_num: int) -> Article:
        """
        Construct Article object from bboxes

        Args:
            bboxes: List of bboxes belonging to this article
            page_id: Page identifier (e.g., "1974/04/24/page_001")
            article_num: Article number on this page

        Returns:
            Article object
        """
        # Extract title (first Title/Section-header, or first text if none)
        title = ""
        for bbox in bboxes:
            if bbox.category in self.title_categories and bbox.text.strip():
                title = bbox.text.strip()
                break

        if not title and bboxes:
            # Use first text as title (truncated)
            first_text = bboxes[0].text.strip()
            title = first_text[:100] + "..." if len(first_text) > 100 else first_text

        # Collect text paragraphs
        text_parts = []
        images = []
        captions = []

        for bbox in bboxes:
            if bbox.category == 'Picture':
                images.append(bbox)
            elif bbox.category == 'Caption':
                captions.append(bbox.text.strip())
            elif bbox.text.strip():
                text_parts.append(bbox.text.strip())

        # Combine text
        full_text = '\n\n'.join(text_parts)

        # Check for continuation markers
        continuation_to = None
        continuation_from = None

        for bbox in bboxes:
            if bbox.text:
                page_num = self.continuation_detector.find_continuation_marker(bbox.text)
                if page_num:
                    continuation_to = f"pág. {page_num}"

                from_num = self.continuation_detector.find_from_marker(bbox.text)
                if from_num:
                    continuation_from = f"pág. {from_num}"

        # Generate article ID
        article_id = f"{page_id}_article_{article_num:03d}"

        return Article(
            article_id=article_id,
            title=title,
            text=full_text,
            page=page_id,
            bboxes=[b for b in bboxes if b.category not in ['Picture', 'Caption']],
            images=images,
            captions=captions,
            continuation_to=continuation_to,
            continuation_from=continuation_from
        )

    def assign_boxes_to_titles(self, bboxes: List[BBox], columns: List[List[BBox]]) -> Dict[int, List[BBox]]:
        """
        Assign each box to the most appropriate title using spatial reasoning

        Returns:
            Dict mapping title index to list of boxes belonging to that article
        """
        # Find all titles
        titles = [b for b in bboxes if b.category in self.title_categories]

        # If no titles, treat all content as one article
        if not titles:
            return {-1: bboxes}

        # Build column map for constraint checking
        column_map = {}
        for col_idx, column in enumerate(columns):
            for bbox in column:
                column_map[bbox.index] = col_idx

        # Assign each non-title box to nearest title
        assignments = defaultdict(list)

        for bbox in bboxes:
            if bbox.category in self.title_categories:
                # Title belongs to itself
                assignments[bbox.index].append(bbox)
                continue

            # Find best title for this box
            best_title = None
            best_score = -1

            for title in titles:
                score = self.compute_assignment_score(title, bbox, column_map, columns)
                if score > best_score:
                    best_score = score
                    best_title = title

            # Assign to best title if score is good enough
            if best_title and best_score > 0.3:
                assignments[best_title.index].append(bbox)
            else:
                # Orphan box - create separate article
                assignments[bbox.index].append(bbox)

        return assignments

    def compute_assignment_score(self, title: BBox, bbox: BBox,
                                 column_map: dict, columns: List[List[BBox]]) -> float:
        """
        Compute how likely bbox belongs to article with this title

        Returns:
            Score from 0.0 (doesn't belong) to 1.0 (definitely belongs)
        """
        # Distance factor
        distance = self.euclidean_distance(title, bbox)
        if distance > 1000:  # Too far
            return 0.0

        # Same column bonus
        same_column = (title.index in column_map and bbox.index in column_map and
                      column_map[title.index] == column_map[bbox.index])

        # Box must be BELOW title (reading order)
        if bbox.y1 < title.y2:
            return 0.0  # Box is above title - can't belong to this article

        # Calculate base score from distance
        distance_score = max(0, 1.0 - (distance / 1000))

        # Boost if same column
        if same_column:
            distance_score *= 1.5  # 50% boost

        # Penalize if different column and not adjacent
        if not same_column:
            if bbox.index in column_map and title.index in column_map:
                col_diff = abs(column_map[bbox.index] - column_map[title.index])
                if col_diff > 1:  # More than 1 column away
                    distance_score *= 0.3  # Heavy penalty

        return min(1.0, distance_score)

    def filter_empty_articles(self, articles: List[Article]) -> List[Article]:
        """
        Remove articles that are just empty images with no text content

        Empty image articles are artifacts from isolated Picture boxes that
        don't connect to any text. Real articles should have text paragraphs.

        Args:
            articles: List of extracted articles

        Returns:
            Filtered list without empty image articles
        """
        filtered = []
        for article in articles:
            # Keep if has text bboxes (paragraphs)
            if len(article.bboxes) > 0:
                filtered.append(article)
            # Or if has meaningful captions
            elif article.captions and any(len(c) > 20 for c in article.captions):
                filtered.append(article)
            # Or if has actual text content
            elif article.text and len(article.text.strip()) > 15:
                filtered.append(article)

        if len(filtered) < len(articles):
            print(f"Filtered empty articles: {len(articles)} → {len(filtered)} articles")

        return filtered

    def get_article_bounds(self, boxes: List[BBox]) -> Optional[dict]:
        """
        Compute spatial extent of an article

        Args:
            boxes: List of bounding boxes in article

        Returns:
            Dict with top, bottom, left, right, center_x, center_y, num_boxes
        """
        if not boxes:
            return None

        return {
            'top': min(b.y1 for b in boxes),
            'bottom': max(b.y2 for b in boxes),
            'left': min(b.x1 for b in boxes),
            'right': max(b.x2 for b in boxes),
            'center_x': sum(b.center_x for b in boxes) / len(boxes),
            'center_y': sum(b.center_y for b in boxes) / len(boxes),
            'num_boxes': len(boxes)
        }

    def fits_in_bounds(self, box: BBox, bounds: Optional[dict], tolerance: float = 400) -> bool:
        """
        Check if box fits spatially within article bounds

        Args:
            box: Box to check
            bounds: Article bounds from get_article_bounds()
            tolerance: How far outside bounds is acceptable (pixels)

        Returns:
            True if box is spatially consistent with article
        """
        if not bounds:
            return True

        # Check if box is within expanded bounds
        if box.y1 < bounds['top'] - tolerance or box.y2 > bounds['bottom'] + tolerance:
            return False  # Vertically distant

        if box.x1 < bounds['left'] - tolerance or box.x2 > bounds['right'] + tolerance:
            return False  # Horizontally distant

        return True

    def find_box_directly_above(self, box: BBox, bboxes: List[BBox],
                               article_assignment: Dict[int, int],
                               max_gap: float = 120) -> Optional[BBox]:
        """
        Find the box directly above this box (same column, close vertically)
        Used for spatial-first message passing

        Args:
            box: Current box
            bboxes: All boxes
            article_assignment: Current assignments
            max_gap: Maximum vertical gap to consider "directly above"

        Returns:
            Box directly above, or None
        """
        candidates = []

        for other in bboxes:
            if other.index not in article_assignment:
                continue

            # Check if above (other's bottom < box's top)
            if other.y2 > box.y1:
                continue

            # Check vertical gap
            gap = box.y1 - other.y2
            if gap > max_gap:
                continue

            # Check horizontal overlap (same column)
            x_overlap = min(box.x2, other.x2) - max(box.x1, other.x1)
            if box.width > 0 and other.width > 0:
                overlap_ratio = x_overlap / min(box.width, other.width)
                if overlap_ratio > 0.5:
                    candidates.append((other, gap))

        # Return closest box above
        return min(candidates, key=lambda x: x[1])[0] if candidates else None

    def find_column_index(self, box: BBox, columns: List[List[BBox]]) -> int:
        """Find which column a box belongs to"""
        for i, column in enumerate(columns):
            if any(b.index == box.index for b in column):
                return i
        return -1

    def is_at_column_top(self, box: BBox, column: List[BBox], threshold: float = 200) -> bool:
        """Check if box is near top of its column"""
        if not column:
            return False
        column_top = min(b.y1 for b in column)
        return (box.y1 - column_top) < threshold

    def is_at_column_bottom(self, box: BBox, column: List[BBox], threshold: float = 200) -> bool:
        """Check if box is near bottom of its column"""
        if not column:
            return False
        column_bottom = max(b.y2 for b in column)
        return (column_bottom - box.y2) < threshold

    def no_title_between(self, box_a: BBox, box_b: BBox, titles: List[BBox]) -> bool:
        """
        Check if any title exists spatially between two boxes
        """
        min_y = min(box_a.y2, box_b.y1)
        max_y = max(box_a.y2, box_b.y1)

        for title in titles:
            if min_y < title.y1 < max_y:
                return False
        return True

    def detect_parallel_titles(self, titles: List[BBox], y_threshold: float = 250) -> List[set]:
        """
        Detect titles that are at similar Y positions (parallel articles)

        Parallel articles are side-by-side articles that should never mix.
        Example: Page 2 has two articles at Y~380px in different columns.

        Args:
            titles: List of title boxes
            y_threshold: Maximum Y difference to consider parallel (pixels)

        Returns:
            List of sets, each set contains indices of titles that are parallel
        """
        if len(titles) <= 1:
            return []

        groups = []

        for title in titles:
            placed = False

            # Try to place in existing group
            for group in groups:
                for existing_title_idx in group:
                    existing_title = next((t for t in titles if t.index == existing_title_idx), None)
                    if existing_title and abs(title.y1 - existing_title.y1) < y_threshold:
                        group.add(title.index)
                        placed = True
                        break
                if placed:
                    break

            # Create new group if not placed
            if not placed:
                groups.append({title.index})

        # Only return groups with 2+ titles (actual parallels)
        parallel_groups = [g for g in groups if len(g) >= 2]

        if parallel_groups:
            print(f"Detected {len(parallel_groups)} parallel article group(s)")
            for i, group in enumerate(parallel_groups):
                print(f"  Group {i+1}: {len(group)} parallel titles")

        return parallel_groups

    def estimate_font_size(self, bbox: BBox) -> float:
        """
        Estimate font size from bounding box dimensions

        Font size ≈ bbox_height / number_of_lines

        Args:
            bbox: Bounding box

        Returns:
            Estimated font size in pixels
        """
        # Count lines in text
        num_lines = bbox.text.count('\n') + 1

        # Font size ≈ height / lines
        font_size = bbox.height / num_lines if num_lines > 0 else bbox.height

        return font_size

    def detect_title_like_text(self, bboxes: List[BBox], existing_titles: List[BBox]) -> List[BBox]:
        """
        Detect Text boxes that should have been classified as Section-headers by OCR

        Uses heuristics to identify misclassified titles:
        1. ALL CAPS text
        2. Short text (< 100 characters)
        3. No sentence-ending punctuation
        4. Larger font size (> 40px)
        5. Isolated box (not part of continuous text flow)

        Args:
            bboxes: All bounding boxes
            existing_titles: Already-detected title boxes

        Returns:
            List of Text boxes that look like titles
        """
        title_candidates = []

        # Get existing title Y-positions to avoid duplicates
        existing_title_regions = set()
        for title in existing_titles:
            existing_title_regions.add((title.center_x // 100, title.center_y // 100))

        for bbox in bboxes:
            if bbox.category != 'Text':
                continue

            # Skip very small or very large boxes
            if bbox.height < 20 or bbox.height > 300:
                continue

            # Check if near existing title (avoid duplicates)
            region = (bbox.center_x // 100, bbox.center_y // 100)
            if region in existing_title_regions:
                continue

            # Extract text for analysis
            text = bbox.text.strip()
            if len(text) < 3:
                continue

            # HEURISTIC 1: ALL CAPS (strong signal)
            is_all_caps = text.isupper() and len(text) > 3

            # HEURISTIC 2: Short text
            is_short = len(text) < 100

            # HEURISTIC 3: No sentence-ending punctuation
            has_no_sentence_end = not any(text.endswith(p) for p in ['.', '!', '?', ',', ';'])

            # HEURISTIC 4: Font size estimation
            font_size = self.estimate_font_size(bbox)
            is_large_font = font_size > 40

            # HEURISTIC 5: Not part of continuous paragraph (no lowercase at start)
            starts_with_upper = text[0].isupper() if text else False

            # Compute score
            score = 0
            if is_all_caps:
                score += 2  # Strong signal
            if is_short:
                score += 1
            if has_no_sentence_end:
                score += 1
            if is_large_font:
                score += 2  # Strong signal
            if starts_with_upper and is_short:
                score += 1

            # Threshold: need at least 3 points
            if score >= 3:
                title_candidates.append(bbox)
                print(f"  Detected title-like text (score={score}): {text[:50]}... (font≈{font_size:.0f}px)")

        return title_candidates

    def promote_to_titles(self, bboxes: List[BBox], title_like: List[BBox]) -> List[BBox]:
        """
        Create pseudo-Section-header boxes from title-like Text boxes

        Args:
            bboxes: All bounding boxes
            title_like: Text boxes detected as title-like

        Returns:
            List of promoted boxes (category changed to Section-header)
        """
        promoted = []

        for bbox in title_like:
            # Create copy with Section-header category
            pseudo_title = BBox(
                x1=bbox.x1,
                y1=bbox.y1,
                x2=bbox.x2,
                y2=bbox.y2,
                category='Section-header',  # Promote from Text!
                text=bbox.text,
                index=bbox.index
            )
            promoted.append(pseudo_title)

        return promoted

    def merge_subsections(self, titles: List[BBox]) -> Tuple[List[BBox], Dict[int, int]]:
        """
        Identify sub-sections and merge into parent articles

        Uses FONT SIZE (bbox_height / num_lines) as primary signal:
        - Large font (>60px/line) = main article title
        - Small font (<40px/line) = sub-section title

        Also uses: position on page, ALL-CAPS, clustering

        Args:
            titles: List of all title/section-header boxes

        Returns:
            (article_level_titles, subsection_to_parent_map)
        """
        if len(titles) <= 1:
            return titles, {}

        # Sort by Y position
        sorted_titles = sorted(titles, key=lambda t: t.y1)

        article_titles = []
        subsection_map = {}

        for title in sorted_titles:
            # Estimate font size
            font_size = self.estimate_font_size(title)

            # Compute subtitle score
            subtitle_score = 0.0

            # Signal 1: Font size (MOST IMPORTANT)
            if font_size < 40:
                subtitle_score += 0.4  # Small font = likely subtitle
            elif font_size < 60:
                subtitle_score += 0.2  # Medium font
            # Large font (>60) gets score += 0

            # Signal 2: Position on page (lower = more likely subtitle)
            relative_y = title.y1 / 3832
            if relative_y > 0.7:
                subtitle_score += 0.2  # Bottom 30%
            elif relative_y > 0.5:
                subtitle_score += 0.1  # Middle

            # Signal 3: ALL-CAPS and short
            if title.text.isupper() and len(title.text.strip()) < 30:
                subtitle_score += 0.2

            # Signal 4: Clustered with other titles nearby
            nearby = [t for t in titles if abs(t.y1 - title.y1) < 200 and t != title]
            if len(nearby) > 0:
                subtitle_score += 0.2

            # Classify as subtitle or article-level
            if subtitle_score > 0.5:
                # This is a sub-title - find parent article above it
                parent = None
                for potential_parent in reversed(article_titles):  # Search backwards (nearest above)
                    if potential_parent.y2 < title.y1:  # Parent must be above
                        parent = potential_parent
                        break

                if parent:
                    subsection_map[title.index] = parent.index
                else:
                    # No parent found - treat as article (shouldn't happen often)
                    article_titles.append(title)
            else:
                # Article-level title
                article_titles.append(title)

        if subsection_map:
            print(f"Detected {len(subsection_map)} sub-section(s) with font-size analysis")
            for sub_idx, parent_idx in subsection_map.items():
                sub = next(t for t in titles if t.index == sub_idx)
                parent = next(t for t in titles if t.index == parent_idx)
                sub_font = self.estimate_font_size(sub)
                parent_font = self.estimate_font_size(parent)
                print(f"  Sub-title (font={sub_font:.0f}px): {sub.text[:30]} → Parent (font={parent_font:.0f}px)")

        return article_titles, subsection_map

    def compute_adaptive_thresholds(self, bboxes: List[BBox]) -> Dict[str, float]:
        """
        Compute adaptive distance thresholds based on actual box distribution

        Instead of hard-coding thresholds, analyze the inter-box distances
        to determine what "nearby" means for this specific page.

        Args:
            bboxes: All bounding boxes on the page

        Returns:
            Dict with 'horizontal_gap', 'vertical_gap', 'cluster_horizontal', 'cluster_vertical'
        """
        if len(bboxes) < 10:
            # Not enough data, use defaults
            return {
                'horizontal_gap': 300,
                'vertical_gap': 200,
                'cluster_horizontal': 400,
                'cluster_vertical': 400
            }

        # Get content boxes only
        content_boxes = [b for b in bboxes
                        if b.category not in ['Picture', 'Page-header', 'Page-footer']]

        if len(content_boxes) < 5:
            return {
                'horizontal_gap': 300,
                'vertical_gap': 200,
                'cluster_horizontal': 400,
                'cluster_vertical': 400
            }

        # Calculate horizontal distances between boxes
        h_distances = []
        for i, box_a in enumerate(content_boxes):
            for box_b in content_boxes[i+1:]:
                # Only consider boxes that overlap vertically
                v_overlap = min(box_a.y2, box_b.y2) - max(box_a.y1, box_b.y1)
                if v_overlap > -100:  # Allow some vertical tolerance
                    h_dist = abs(box_a.center_x - box_b.center_x)
                    h_distances.append(h_dist)

        # Calculate vertical distances between boxes
        v_distances = []
        for i, box_a in enumerate(content_boxes):
            for box_b in content_boxes[i+1:]:
                # Only consider boxes that overlap horizontally
                h_overlap = min(box_a.x2, box_b.x2) - max(box_a.x1, box_b.x1)
                if h_overlap > 0:  # Must have some horizontal overlap
                    # Distance from bottom of one to top of other
                    if box_a.y2 < box_b.y1:
                        v_dist = box_b.y1 - box_a.y2
                        v_distances.append(v_dist)
                    elif box_b.y2 < box_a.y1:
                        v_dist = box_a.y1 - box_b.y2
                        v_distances.append(v_dist)

        if not h_distances or not v_distances:
            return {
                'horizontal_gap': 300,
                'vertical_gap': 200,
                'cluster_horizontal': 400,
                'cluster_vertical': 400
            }

        # Use percentiles to determine thresholds
        h_distances.sort()
        v_distances.sort()

        # For "nearby" checking (orphan detection):
        # Use 40th percentile for horizontal (below median to capture within-column)
        # Use 75th percentile for vertical (more lenient)
        horizontal_gap = h_distances[int(len(h_distances) * 0.4)]
        vertical_gap = v_distances[int(len(v_distances) * 0.75)] if v_distances else 200

        # For clustering (grouping boxes together):
        # Use 50th percentile (median) for horizontal
        # Use 60th percentile for vertical
        cluster_horizontal = h_distances[len(h_distances) // 2]
        cluster_vertical = v_distances[int(len(v_distances) * 0.6)] if v_distances else 400

        # Apply sensible bounds
        thresholds = {
            'horizontal_gap': max(200, min(400, horizontal_gap)),  # Cap at 400px
            'vertical_gap': max(100, min(400, vertical_gap)),
            'cluster_horizontal': max(300, min(600, cluster_horizontal)),
            'cluster_vertical': max(200, min(500, cluster_vertical))
        }

        print(f"Adaptive thresholds: H_gap={thresholds['horizontal_gap']:.0f}px, "
              f"V_gap={thresholds['vertical_gap']:.0f}px, "
              f"Cluster_H={thresholds['cluster_horizontal']:.0f}px, "
              f"Cluster_V={thresholds['cluster_vertical']:.0f}px")

        return thresholds

    def detect_orphan_regions(self, bboxes: List[BBox],
                             titles: List[BBox],
                             min_boxes: int = 3,
                             thresholds: Dict[str, float] = None) -> List[List[BBox]]:
        """
        Detect spatial regions with coherent content but no title

        Uses spatial clustering to find groups of text boxes that:
        1. Are close together (adaptive thresholds based on page layout)
        2. Have at least min_boxes content boxes
        3. Have NO title boxes nearby

        This is more robust than column-based detection for complex layouts.

        Args:
            bboxes: All bounding boxes
            titles: List of detected titles
            min_boxes: Minimum boxes to consider orphan region
            thresholds: Optional pre-computed thresholds

        Returns:
            List of orphan regions (each is list of boxes)
        """
        # Compute adaptive thresholds if not provided
        if thresholds is None:
            thresholds = self.compute_adaptive_thresholds(bboxes)

        # Get content boxes only (exclude media and headers/footers)
        content_boxes = [b for b in bboxes
                        if b.category not in ['Picture', 'Page-header', 'Page-footer', 'Caption']
                        and b.category not in ['Title', 'Section-header']]

        if len(content_boxes) < min_boxes:
            return []

        # Build spatial clusters of nearby boxes (column-aware)
        clusters = []
        used = set()

        cluster_h = thresholds['cluster_horizontal']
        cluster_v = thresholds['cluster_vertical']

        for seed_box in content_boxes:
            if seed_box.index in used:
                continue

            # Start new cluster from this seed
            cluster = [seed_box]
            used.add(seed_box.index)

            # Expand cluster by finding nearby boxes in same "column region"
            changed = True
            while changed:
                changed = False
                for box in content_boxes:
                    if box.index in used:
                        continue

                    # Check if box is close to any box in cluster
                    for cluster_box in cluster:
                        v_dist = abs(box.center_y - cluster_box.center_y)
                        h_dist = abs(box.center_x - cluster_box.center_x)

                        # Conservative clustering: closer vertically than horizontally
                        # AND must have some X-overlap (same column)
                        x_overlap = min(box.x2, cluster_box.x2) - max(box.x1, cluster_box.x1)

                        if v_dist < cluster_v and h_dist < cluster_h and x_overlap > -50:
                            # Boxes are vertically close, horizontally close, and in same column
                            cluster.append(box)
                            used.add(box.index)
                            changed = True
                            break

            if len(cluster) >= min_boxes:
                clusters.append(cluster)

        # Filter clusters: keep only those WITHOUT nearby titles
        orphan_regions = []

        for cluster in clusters:
            # Get cluster bounds
            cluster_center_x = sum(b.center_x for b in cluster) / len(cluster)
            cluster_center_y = sum(b.center_y for b in cluster) / len(cluster)
            cluster_y_min = min(b.y1 for b in cluster)
            cluster_y_max = max(b.y2 for b in cluster)

            # Check if any title is near this cluster
            has_nearby_title = False
            for title in titles:
                # Check if title is above or within cluster's Y-range
                # (Title at top of article won't overlap with boxes below it)
                vertical_proximity = 0.0

                if title.y2 <= cluster_y_min:
                    # Title is above cluster - check gap
                    vertical_proximity = cluster_y_min - title.y2
                elif title.y1 >= cluster_y_max:
                    # Title is below cluster - check gap
                    vertical_proximity = title.y1 - cluster_y_max
                else:
                    # Title overlaps cluster vertically
                    vertical_proximity = 0.0

                # Title is nearby if: within adaptive thresholds vertically AND horizontally
                v_threshold = thresholds['vertical_gap']
                h_threshold = thresholds['horizontal_gap']

                if vertical_proximity < v_threshold:
                    h_dist = abs(title.center_x - cluster_center_x)
                    if h_dist < h_threshold:
                        has_nearby_title = True
                        break

            if not has_nearby_title:
                # Check for continuation markers
                has_continuation = any(
                    'vem da pág' in box.text.lower() or
                    'continuação' in box.text.lower() or
                    'continua da pág' in box.text.lower()
                    for box in cluster
                )

                if not has_continuation:
                    orphan_regions.append(cluster)

        return orphan_regions

    def detect_continuation_from_previous_page(self, boxes: List[BBox]) -> Optional[int]:
        """
        Check if page contains continuation from previous page

        Looks for markers like "Vem da pág. X" or "Continuação da pág. X"
        in the first few boxes of the page.

        Args:
            boxes: All bounding boxes on the page

        Returns:
            Page number if continuation detected, None otherwise
        """
        # Regex patterns for continuation markers
        patterns = [
            r'vem\s+da\s+p[aá]g(?:ina)?\.?\s+(\d+)',
            r'continuação\s+da\s+p[aá]g(?:ina)?\.?\s+(\d+)',
            r'continua(?:ção)?\s+da\s+p[aá]g(?:ina)?\.?\s+(\d+)'
        ]

        # Check first 5 boxes (continuation markers usually at top)
        for box in boxes[:5]:
            for pattern in patterns:
                match = re.search(pattern, box.text.lower())
                if match:
                    return int(match.group(1))

        return None

    def seed_orphan_regions(self, orphan_regions: List[List[BBox]],
                           article_assignment: Dict[int, int],
                           next_article_id: int) -> int:
        """
        Create pseudo-articles for orphan regions

        Seeds articles from the first box in each orphan region.
        This ensures titleless content gets its own article instead of
        being absorbed into nearby titled articles.

        Args:
            orphan_regions: List of orphan regions (each is list of boxes)
            article_assignment: Current assignments (will be updated in-place)
            next_article_id: Next available article ID

        Returns:
            Updated next_article_id (incremented for each seeded article)
        """
        for region_idx, region in enumerate(orphan_regions):
            if not region:
                continue

            # Sort region boxes by reading order (top-to-bottom, left-to-right)
            region_sorted = sorted(region, key=lambda b: (b.y1, b.x1))

            # Seed first box in region
            seed_box = region_sorted[0]
            article_assignment[seed_box.index] = next_article_id

            # Get region location for logging
            x_center = sum(b.center_x for b in region) / len(region)
            y_center = sum(b.center_y for b in region) / len(region)

            print(f"  Orphan region {region_idx} (X≈{x_center:.0f}, Y≈{y_center:.0f}): Seeded article {next_article_id} from box {seed_box.index}")
            print(f"    Region has {len(region)} boxes")
            print(f"    Preview: {seed_box.text[:50]}")

            next_article_id += 1

        return next_article_id

    def get_neighbor_votes(self, box: BBox, G: nx.Graph,
                          article_assignment: Dict[int, int],
                          title_affinity: Dict[int, Tuple[int, float]] = None,
                          titles_by_article: Dict[int, BBox] = None,
                          parallel_groups: List[set] = None,
                          all_bboxes: List[BBox] = None,
                          h_threshold: float = 350) -> Dict[int, float]:
        """
        Collect weighted votes from neighbors for article assignment
        Applies title affinity constraint to prevent cross-article contamination

        Args:
            box: Box to get votes for
            G: Graph with edge weights
            article_assignment: Current article assignments
            title_affinity: Optional title affinity map
            titles_by_article: Optional mapping of article_id to title box
            all_bboxes: All boxes for cross-column check

        Returns:
            Dict mapping article_id → total vote weight
        """
        votes = defaultdict(float)

        for neighbor_idx in G.neighbors(box.index):
            if neighbor_idx in article_assignment:
                article_id = article_assignment[neighbor_idx]
                edge_weight = G[box.index][neighbor_idx]['weight']

                # BLOCK votes from boxes in far-away columns (adaptive X-distance constraint)
                if all_bboxes:
                    neighbor_box = next((b for b in all_bboxes if b.index == neighbor_idx), None)
                    if neighbor_box:
                        h_dist = abs(box.center_x - neighbor_box.center_x)
                        if h_dist > h_threshold:
                            # Too far horizontally - different column, block vote
                            edge_weight = 0.0

                # FIRST: Block votes from parallel articles (side-by-side articles)
                if (parallel_groups and title_affinity and titles_by_article and
                    box.index in title_affinity and article_id in titles_by_article):

                    box_best_title_idx, box_title_score = title_affinity[box.index]
                    article_title = titles_by_article[article_id]

                    # Check if box's title and article's title are in same parallel group
                    if box_best_title_idx != article_title.index:
                        for group in parallel_groups:
                            if box_best_title_idx in group and article_title.index in group:
                                # Parallel articles - BLOCK vote completely
                                edge_weight = 0.0
                                break

                # THEN: Apply title affinity filtering (if available)
                if title_affinity and titles_by_article and box.index in title_affinity:
                    box_best_title_idx, box_title_score = title_affinity[box.index]

                    # Get the title of the article this neighbor belongs to
                    if article_id in titles_by_article:
                        article_title = titles_by_article[article_id]

                        # If box's best title DOESN'T match article's title
                        if article_title.index != box_best_title_idx:
                            # STRICT: If box has ANY meaningful preference for different title, block vote
                            if box_title_score > 0.55:
                                # Box clearly belongs to different article - BLOCK vote completely
                                edge_weight = 0.0  # Was *= 0.02, now complete block
                            elif box_title_score > 0.45:
                                # Moderate preference = very heavy penalty
                                edge_weight *= 0.05  # Was 0.1
                            elif box_title_score > 0.35:
                                # Weak preference = heavy penalty
                                edge_weight *= 0.2  # Was 0.3
                            else:
                                # Very weak/neutral preference = medium penalty
                                edge_weight *= 0.5  # Was 0.6

                votes[article_id] += edge_weight

        return dict(votes)

    def extract_articles_message_passing(self, bboxes: List[BBox], G: nx.Graph,
                                        titles: List[BBox],
                                        embeddings: Dict[int, np.ndarray],
                                        max_iterations: int = 10) -> tuple:
        """
        Extract articles using message passing label propagation

        Each title seeds an article. Labels propagate through weighted edges.
        Boxes vote on article membership based on neighbors' assignments.

        Args:
            bboxes: All bounding boxes
            G: Graph with edge weights (spatial + semantic + continuation + title affinity)
            titles: List of title boxes (seeds for articles)
            embeddings: Pre-computed embeddings for title affinity
            max_iterations: Maximum propagation iterations

        Returns:
            Tuple of (articles, subsection_map) where articles is list of bbox lists
        """
        # Detect and merge sub-sections
        article_titles, subsection_map = self.merge_subsections(titles)

        print(f"Article-level titles: {len(article_titles)}")
        if subsection_map:
            print(f"Sub-sections detected: {len(subsection_map)}")

        # Compute title affinity map (using only article-level titles)
        title_affinity = self.compute_title_affinity_map(bboxes, article_titles, embeddings)

        # Detect parallel titles (among article-level titles only)
        parallel_groups = self.detect_parallel_titles(article_titles)

        # Build titles_by_article mapping (using article-level titles)
        titles_by_article = {i: title for i, title in enumerate(article_titles)}

        # PHASE 1: Initialize with article-level titles
        article_assignment = {}

        for i, title in enumerate(article_titles):
            article_assignment[title.index] = i

        # Assign sub-sections to their parent article
        for subsection_idx, parent_idx in subsection_map.items():
            if parent_idx in article_assignment:
                parent_article = article_assignment[parent_idx]
                article_assignment[subsection_idx] = parent_article
                print(f"  Sub-section at index {subsection_idx} → Article {parent_article}")

        print(f"Initialized {len(article_titles)} articles from titles")

        # Compute adaptive thresholds based on page layout
        thresholds = self.compute_adaptive_thresholds(bboxes)

        # Detect orphan regions (titleless content clusters)
        # Check if this is a continuation page first
        continuation_from = self.detect_continuation_from_previous_page(bboxes)

        if continuation_from:
            print(f"Page is continuation from page {continuation_from}")
            print(f"  Orphan regions will be treated as continuation content")
            # Don't seed new articles - let them join via message passing
        else:
            # Detect orphan regions (spatial clusters without nearby titles)
            orphan_regions = self.detect_orphan_regions(bboxes, article_titles, thresholds=thresholds)

            if orphan_regions:
                print(f"Detected {len(orphan_regions)} orphan region(s) without titles")

                # Seed new articles for orphan regions
                next_article_id = len(article_titles)
                next_article_id = self.seed_orphan_regions(
                    orphan_regions, article_assignment, next_article_id
                )

        # Detect columns for spatial reasoning
        columns = []
        for bbox in bboxes:
            if bbox.category != 'Picture':
                # Find or create column
                placed = False
                for col in columns:
                    if col and abs(bbox.center_x - col[0].center_x) < 100:
                        col.append(bbox)
                        placed = True
                        break
                if not placed:
                    columns.append([bbox])

        # Sort each column by Y
        for col in columns:
            col.sort(key=lambda b: b.y1)

        # Sort columns by X
        columns.sort(key=lambda col: col[0].center_x if col else 0)

        # PHASE 2: Directional Label Propagation (Reading Order)
        # Process boxes in reading order: column-by-column, top-to-bottom
        reading_order = []
        for column in columns:
            reading_order.extend(column)

        for iteration in range(max_iterations):
            updated_count = 0

            for box in reading_order:  # SPATIAL ORDER!
                if box.index in article_assignment:
                    continue

                # PRIORITY 1: Box directly above (SAME COLUMN ONLY!)
                above_box = self.find_box_directly_above(box, bboxes, article_assignment)

                if above_box:
                    # Verify same column using X-distance constraint
                    h_dist = abs(box.center_x - above_box.center_x)
                    h_threshold = thresholds['horizontal_gap']

                    if h_dist < h_threshold:
                        # Close horizontally - safe to inherit
                        article_assignment[box.index] = article_assignment[above_box.index]
                        updated_count += 1
                        continue
                    # else: different columns - don't inherit, try other methods

                # PRIORITY 2: Title affinity (if strong)
                if box.index in title_affinity:
                    best_title_idx, affinity_score = title_affinity[box.index]

                    if affinity_score > 0.60:  # Strong spatial + semantic affinity to title
                        for article_id, title in titles_by_article.items():
                            if title.index == best_title_idx:
                                # Check X-distance constraint (don't assign to far-away titles)
                                h_dist = abs(box.center_x - title.center_x)
                                h_threshold = thresholds['horizontal_gap']
                                if h_dist < h_threshold:  # Only assign if reasonably close horizontally
                                    article_assignment[box.index] = article_id
                                    updated_count += 1
                                    break
                        continue

                # PRIORITY 3: General voting
                votes = self.get_neighbor_votes(box, G, article_assignment,
                                               title_affinity, titles_by_article, parallel_groups, bboxes,
                                               thresholds['horizontal_gap'])

                if votes:
                    best_article = max(votes, key=votes.get)
                    best_score = votes[best_article]

                    if best_score >= 0.4:  # Lower threshold for voting
                        article_assignment[box.index] = best_article
                        updated_count += 1

            print(f"  Iteration {iteration + 1}: Assigned {updated_count} boxes")

            if updated_count == 0:
                break  # Converged

        # PHASE 3: Refine based on article bounds
        print("Refining assignments based on spatial bounds...")

        for refinement_iteration in range(3):
            updated_count = 0

            for box in bboxes:
                if box.index not in article_assignment:
                    continue

                current_article = article_assignment[box.index]

                # Get current article's spatial extent
                article_boxes = [b for b in bboxes
                               if article_assignment.get(b.index) == current_article]
                bounds = self.get_article_bounds(article_boxes)

                # Check if box fits in article bounds
                if not self.fits_in_bounds(box, bounds, tolerance=500):
                    # Box is spatially inconsistent - re-vote with title affinity and parallel blocking
                    votes = self.get_neighbor_votes(box, G, article_assignment,
                                                   title_affinity, titles_by_article, parallel_groups, bboxes,
                                                   thresholds['horizontal_gap'])

                    if votes:
                        best_article = max(votes, key=votes.get)
                        if best_article != current_article:
                            article_assignment[box.index] = best_article
                            updated_count += 1

            if updated_count > 0:
                print(f"  Refinement {refinement_iteration + 1}: Reassigned {updated_count} boxes")

            if updated_count == 0:
                break

        # PHASE 4: Handle unassigned boxes
        unassigned = [b for b in bboxes if b.index not in article_assignment]

        if unassigned:
            print(f"Assigning {len(unassigned)} orphan boxes...")

            for box in unassigned:
                # Find nearest assigned box
                min_dist = float('inf')
                nearest_article = None

                for assigned_box in bboxes:
                    if assigned_box.index in article_assignment:
                        dist = self.euclidean_distance(box, assigned_box)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_article = article_assignment[assigned_box.index]

                # Assign to nearest article if reasonably close
                # Increased from 300px to 500px for better orphan recovery
                if nearest_article is not None and min_dist < 500:
                    article_assignment[box.index] = nearest_article

        # Convert assignments to article groups
        articles_dict = defaultdict(list)
        for box in bboxes:
            if box.index in article_assignment:
                article_id = article_assignment[box.index]
                articles_dict[article_id].append(box)

        # Sort each article by reading order
        articles = []
        for article_boxes in articles_dict.values():
            if article_boxes:
                article_boxes.sort(key=lambda b: (b.y1, b.x1))
                articles.append(article_boxes)

        print(f"Message passing complete: {len(articles)} articles")

        # POST-PROCESSING: Split articles that span multiple far-apart columns
        # But preserve articles with subtitle relationships
        print("Post-processing: Checking for cross-column contamination...")
        articles = self.split_cross_column_articles(articles, subsection_map)
        print(f"After cross-column split: {len(articles)} articles")

        return articles, subsection_map

    def merge_titleless_fragments(self, articles: List[List[BBox]]) -> List[List[BBox]]:
        """
        Merge article fragments that lack proper titles and are spatially close

        After cross-column split, some titleless content gets fragmented into
        multiple small articles. This method merges fragments that:
        1. Have no proper title (Section-header/Title)
        2. Are horizontally close (< 600px apart)
        3. Are vertically overlapping (parallel columns)

        Args:
            articles: List of article bbox groups

        Returns:
            List of articles with titleless fragments merged
        """
        if len(articles) <= 1:
            return articles

        # Classify articles as titled or titleless
        titled = []
        titleless = []

        for article in articles:
            has_proper_title = any(b.category in ['Title', 'Section-header']
                                  for b in article)

            if has_proper_title:
                titled.append(article)
            else:
                titleless.append(article)

        if not titleless:
            return articles  # All articles have titles

        print(f"  Merging {len(titleless)} titleless fragment(s)...")

        # Merge titleless fragments that are spatially close
        merged_titleless = []
        used = set()

        for i, frag1 in enumerate(titleless):
            if i in used:
                continue

            # Start new merged group
            merged = frag1[:]
            used.add(i)

            # Get spatial bounds of frag1
            frag1_x_min = min(b.x1 for b in frag1)
            frag1_x_max = max(b.x2 for b in frag1)
            frag1_y_min = min(b.y1 for b in frag1)
            frag1_y_max = max(b.y2 for b in frag1)

            # Find nearby fragments to merge
            for j, frag2 in enumerate(titleless):
                if j in used or j <= i:
                    continue

                # Get spatial bounds of frag2
                frag2_x_min = min(b.x1 for b in frag2)
                frag2_x_max = max(b.x2 for b in frag2)
                frag2_y_min = min(b.y1 for b in frag2)
                frag2_y_max = max(b.y2 for b in frag2)

                # Compute horizontal distance (center-to-center)
                frag1_x_center = (frag1_x_min + frag1_x_max) / 2
                frag2_x_center = (frag2_x_min + frag2_x_max) / 2
                h_dist = abs(frag1_x_center - frag2_x_center)

                # Compute vertical overlap ratio
                y_overlap = min(frag1_y_max, frag2_y_max) - max(frag1_y_min, frag2_y_min)
                y_span = max(frag1_y_max, frag2_y_max) - min(frag1_y_min, frag2_y_min)
                v_overlap_ratio = y_overlap / y_span if y_span > 0 else 0

                # Merge if:
                # - Horizontally close (< 600px apart)
                # - Vertically overlapping (> 30% overlap, indicating parallel columns)
                if h_dist < 600 and v_overlap_ratio > 0.3:
                    print(f"    Merging fragments {i} and {j} (h_dist={h_dist:.0f}px, v_overlap={v_overlap_ratio:.2f})")
                    merged.extend(frag2)
                    used.add(j)

            # Sort merged group by reading order
            merged.sort(key=lambda b: (b.y1, b.x1))
            merged_titleless.append(merged)

        print(f"  Merged to {len(merged_titleless)} titleless article(s)")

        # Combine titled articles with merged titleless articles
        return titled + merged_titleless

    def split_cross_column_articles(self, articles: List[List[BBox]],
                                   subsection_map: Dict[int, int] = None,
                                   max_horizontal_span: float = 350) -> List[List[BBox]]:
        """
        Split articles that incorrectly span multiple far-apart columns

        If an article has boxes that are >350px apart horizontally,
        it's likely contaminated. Split into separate articles by X-clustering.

        EXCEPTION: Preserve articles with subtitle relationships (don't split).

        Args:
            articles: List of article bbox groups
            subsection_map: Map of subtitle indices to parent article indices
            max_horizontal_span: Maximum horizontal distance within one article (default: 350px)

        Returns:
            List of split articles with titleless fragments merged
        """
        if subsection_map is None:
            subsection_map = {}

        split_articles = []

        for art_idx, article in enumerate(articles):
            if len(article) < 2:
                split_articles.append(article)
                continue

            # Check if this article has subtitles (protect from splitting)
            has_subtitles = any(b.index in subsection_map.values() or b.index in subsection_map.keys()
                               for b in article if b.category in ['Title', 'Section-header'])

            # Check horizontal span
            x_min = min(b.x1 for b in article)
            x_max = max(b.x2 for b in article)
            h_span = x_max - x_min

            if h_span > max_horizontal_span:
                if has_subtitles:
                    print(f"  Article {art_idx}: H-span={h_span:.0f}px - has subtitles, preserving")
                    split_articles.append(article)
                    continue
                else:
                    print(f"  Article {art_idx}: H-span={h_span:.0f}px (>{max_horizontal_span}px) - checking for contamination...")

            if h_span <= max_horizontal_span:
                # Normal article - keep as is
                split_articles.append(article)
                continue

            # Wide article - check if it's contaminated (boxes in far-apart columns)
            # Cluster boxes by X-coordinate
            x_centers = [b.center_x for b in article]
            x_sorted_indices = sorted(range(len(article)), key=lambda i: x_centers[i])

            # Find gaps > max_horizontal_span
            clusters = [[article[x_sorted_indices[0]]]]

            for i in range(1, len(x_sorted_indices)):
                prev_x = x_centers[x_sorted_indices[i-1]]
                curr_x = x_centers[x_sorted_indices[i]]

                if abs(curr_x - prev_x) > max_horizontal_span:
                    # Large gap - start new cluster
                    clusters.append([article[x_sorted_indices[i]]])
                else:
                    # Add to current cluster
                    clusters[-1].append(article[x_sorted_indices[i]])

            if len(clusters) > 1:
                # Article was contaminated - split it
                print(f"  Split contaminated article: {len(article)} boxes → {len(clusters)} articles")
                for cluster in clusters:
                    # Sort cluster by reading order
                    cluster.sort(key=lambda b: (b.y1, b.x1))
                    split_articles.append(cluster)
            else:
                # Wide but not contaminated (e.g., wide title spanning multiple columns)
                split_articles.append(article)

        # NEW: Merge titleless fragments created by splitting
        # This prevents titleless content from being split into multiple small articles
        split_articles = self.merge_titleless_fragments(split_articles)

        return split_articles

    def extract_articles_reading_flow(self, bboxes: List[BBox], columns: List[List[BBox]]) -> List[List[BBox]]:
        """
        Extract articles by following reading flow order with strict boundaries

        Algorithm:
        1. Sort all boxes in reading order (column-by-column, top-to-bottom)
        2. Iterate sequentially
        3. Start new article on: Title/Section-header OR large gap from current article
        4. Add to current article if spatially close to last box in current article

        Returns:
            List of article bbox groups
        """
        # Flatten columns into reading order
        reading_order_boxes = []
        for column in columns:
            reading_order_boxes.extend(column)  # Already sorted top-to-bottom within column

        if not reading_order_boxes:
            return []

        articles = []
        current_article = []
        last_box_y = None

        for bbox in reading_order_boxes:
            start_new_article = False

            # Rule 1: Title/Section-header ALWAYS starts new article
            if bbox.category in self.title_categories:
                start_new_article = True

            # Rule 2: Large vertical gap from last box in current article
            elif current_article and last_box_y is not None:
                gap = bbox.y1 - last_box_y
                if gap > self.min_article_gap:
                    start_new_article = True

            # Start new article if needed
            if start_new_article and current_article:
                articles.append(current_article)
                current_article = []

            # Add box to current article
            current_article.append(bbox)
            last_box_y = bbox.y2

        # Add final article
        if current_article:
            articles.append(current_article)

        return articles

    def extract_articles(self, ocr_file_path: Path) -> List[Article]:
        """
        Extract all articles from a single OCR file

        Args:
            ocr_file_path: Path to OCR JSON file

        Returns:
            List of extracted articles
        """
        # Load bboxes
        bboxes = self.load_ocr_file(ocr_file_path)
        if not bboxes:
            return []

        # Filter metadata
        content_boxes = self.filter_metadata_boxes(bboxes)

        # Filter masthead (newspaper name, date, etc. in top 15%)
        content_boxes = self.filter_masthead(content_boxes)

        # Detect columns
        columns = self.column_detector.detect_columns(content_boxes)

        # Page ID from file path (e.g., "1974/04/24/page_001")
        page_id = str(ocr_file_path.relative_to(Path('ocr_output'))).replace('_ocr.json', '')

        # Clear embedding cache for this page (free memory)
        self.embedding_cache = {}

        # NEW: Graph-based approach with semantic embeddings
        print(f"Extracting articles using spatial + semantic similarity...")

        # Build article graph (computes embeddings internally)
        G, embeddings = self.build_article_graph(content_boxes, columns)

        # Extract articles using message passing label propagation
        titles_ocr = [b for b in content_boxes if b.category in self.title_categories]

        # NEW: Detect title-like Text boxes (misclassified by OCR)
        print("Detecting title-like Text boxes (OCR misclassifications)...")
        title_like = self.detect_title_like_text(content_boxes, titles_ocr)

        if title_like:
            print(f"Detected {len(title_like)} title-like Text box(es)")
            # Promote them to Section-headers
            promoted = self.promote_to_titles(content_boxes, title_like)

            # Update content_boxes: replace Text boxes with promoted versions
            content_boxes_updated = []
            promoted_indices = {p.index for p in promoted}

            for bbox in content_boxes:
                if bbox.index in promoted_indices:
                    # Use promoted version
                    promoted_box = next(p for p in promoted if p.index == bbox.index)
                    content_boxes_updated.append(promoted_box)
                else:
                    content_boxes_updated.append(bbox)

            content_boxes = content_boxes_updated

            # Combine OCR titles with promoted titles
            titles = titles_ocr + promoted
        else:
            print("No title-like Text boxes detected")
            titles = titles_ocr

        article_bbox_groups, subsection_map = self.extract_articles_message_passing(content_boxes, G, titles, embeddings)

        print(f"Found {len(article_bbox_groups)} articles (from {len(titles)} titles)")

        # Build Article objects
        all_articles = []
        for article_num, bbox_group in enumerate(article_bbox_groups):
            if bbox_group:  # Skip empty articles
                article = self.build_article_object(
                    bbox_group,
                    page_id,
                    article_num
                )
                all_articles.append(article)

        # Filter out empty image-only articles
        all_articles = self.filter_empty_articles(all_articles)

        return all_articles


def main():
    """Test article extraction on sample files"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Extract articles from OCR-processed newspaper pages')
    parser.add_argument('ocr_file', type=Path, nargs='?', help='OCR JSON file to process')
    parser.add_argument('--articles-output-dir', type=Path, default=Path('articles_output'),
                        help='Output directory for extracted articles (default: articles_output)')
    parser.add_argument('--ocr-base-dir', type=Path, default=Path('ocr_output'),
                        help='Base directory for OCR files (default: ocr_output)')

    args = parser.parse_args()

    if args.ocr_file:
        ocr_file = args.ocr_file
    else:
        # Default: test on first available file
        ocr_files = list(args.ocr_base_dir.rglob('*_ocr.json'))
        if not ocr_files:
            print(f"No OCR files found in {args.ocr_base_dir}/")
            sys.exit(1)
        ocr_file = ocr_files[0]

    print(f"Extracting articles from: {ocr_file}")
    print("=" * 80)

    extractor = ArticleExtractor()
    articles = extractor.extract_articles(ocr_file)

    print(f"\nFound {len(articles)} articles\n")

    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"ID: {article.article_id}")
        print(f"Title: {article.title[:100]}...")
        print(f"Paragraphs: {len(article.bboxes)}")
        print(f"Images: {len(article.images)}")
        if article.continuation_to:
            print(f"Continues to: {article.continuation_to}")
        if article.continuation_from:
            print(f"Continued from: {article.continuation_from}")
        print(f"Text preview: {article.text[:200]}...")

    # Calculate output path - mirror OCR directory structure
    try:
        relative_path = ocr_file.parent.relative_to(args.ocr_base_dir)
        output_dir = args.articles_output_dir / relative_path
    except ValueError:
        # OCR file not in expected base directory, use same directory
        print(f"Warning: {ocr_file} not in {args.ocr_base_dir}, saving to same directory")
        output_dir = ocr_file.parent

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    output_file = output_dir / (ocr_file.stem.replace('_ocr', '_articles') + '.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([a.to_dict() for a in articles], f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved articles to: {output_file}")


if __name__ == '__main__':
    main()
