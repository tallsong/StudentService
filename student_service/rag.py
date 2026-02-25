from __future__ import annotations

import io
import pickle
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_BASE_URL = "https://xu-university.com/"


@dataclass
class Chunk:
    text: str
    source_url: str
    title: str


class HashingVectorizerEmbeddings(Embeddings):
    def __init__(self, n_features: int = 2048) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        matrix_any = cast(Any, matrix)
        dense = np.asarray(matrix_any.toarray(), dtype=np.float32)
        return cast(list[list[float]], dense.tolist())

    def embed_query(self, text: str) -> list[float]:
        embedded = self.embed_documents([text])
        return embedded[0] if embedded else []


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


class WebsiteRAG:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 15.0,
        index_path: str | None = None,
        collected_txt_path: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        
        # Set default index path relative to this module
        if index_path is None:
            module_dir = Path(__file__).parent
            self.index_path = module_dir / ".knowledge_base_index.pkl"
        else:
            self.index_path = Path(index_path)

        self.vector_db_path = self.index_path.with_suffix(".chroma")
        self.collection_name = "xu_university_kb"

        if collected_txt_path is None:
            self.collected_txt_path = self.index_path.with_suffix(".txt")
        else:
            self.collected_txt_path = Path(collected_txt_path)
        
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._last_indexed_at = 0.0
        self._chunks: list[Chunk] = []
        self._embeddings = HashingVectorizerEmbeddings(n_features=2048)
        self._vectorstore: Chroma | None = None
        
        # Auto-load index from disk if available
        self._load_index()

    def query(self, question: str, top_k: int = 5) -> dict[str, object]:
        if not question.strip():
            return {
                "status": "error",
                "message": "Question cannot be empty.",
                "sources": [],
            }

        self._ensure_index()

        if not self._chunks or self._vectorstore is None:
            return {
                "status": "error",
                "message": "No indexed website content available.",
                "sources": [],
            }

        vectorstore = self._vectorstore

        try:
            results = vectorstore.similarity_search_with_score(
                question,
                k=max(top_k * 20, 100),
            )
        except Exception as e:
            return {
                "status": "error",
                "message": f"Vector database query failed: {e}",
                "sources": [],
            }

        question_lower = question.lower()
        fee_intent = any(keyword in question_lower for keyword in ("fee", "fees", "tuition", "cost", "costs", "price", "pricing"))

        def chunk_bonus(document_text: str) -> float:
            if not fee_intent:
                return 0.0
            lowered = document_text.lower()
            bonus = 0.0
            if any(term in lowered for term in ("fee", "fees", "tuition", "cost", "costs", "per month", "per year")):
                bonus += 0.08
            if "€" in document_text or "eur" in lowered or "euro" in lowered:
                bonus += 0.18
            if re.search(r"\b\d{3,}(?:[\.,]\d+)?\b", document_text):
                bonus += 0.05
            return bonus

        best_by_source: dict[str, tuple[float, dict[str, object]]] = {}
        for doc, distance in results:
            doc_any = cast(Any, doc)
            metadata_raw = doc_any.metadata
            metadata = {str(key): cast(object, value) for key, value in dict(metadata_raw).items()}
            document = doc.page_content
            source_url = str(metadata.get("source_url", "")).strip()
            if not source_url:
                continue

            score = self._distance_to_score(distance)
            selection_score = score + chunk_bonus(str(document))
            candidate: dict[str, object] = {
                "score": round(score, 4),
                "source_url": source_url,
                "title": str(metadata.get("title", "")),
                "source_type": "pdf" if self._is_pdf_url(source_url) else "webpage",
                "content": str(document),
            }

            existing = best_by_source.get(source_url)
            if existing is None or selection_score > existing[0]:
                best_by_source[source_url] = (selection_score, candidate)

        candidates = [item[1] for item in sorted(best_by_source.values(), key=lambda pair: pair[0], reverse=True)]

        if not candidates:
            return {
                "status": "no_match",
                "message": "No relevant information found from the website index.",
                "sources": [],
            }

        # Re-rank all candidates first, then keep top_k.
        candidates = self._augment_candidates_with_url_matches(candidates, question)
        hits = self._rerank_by_context(candidates, question)[:top_k]
        for hit in hits:
            source_url = str(hit.get("source_url", ""))
            default_content = str(hit.get("content", ""))
            hit["content"] = self._best_chunk_for_source(source_url, question, default_content)

        return {
            "status": "success",
            "question": question,
            "sources": hits,
            "guidance": (
                "Use the returned sources to answer the user. "
                "Cite source_url values and avoid unsupported claims."
            ),
        }

    def _rerank_by_context(self, hits: list[dict[str, object]], question: str) -> list[dict[str, object]]:
        """
        Smart context-aware re-ranking based on question intent.
        This is NOT hard-coded rule logic—just reasonable heuristics about content structure.
        E.g., if asking about "MSc", prefer /master-programs/ pages; if asking "bachelor", prefer /bachelor/.
        """
        question_lower = question.lower()
        question_normalized = re.sub(r"[^a-z0-9]+", " ", question_lower).strip()
        
        # Detect program level intent from question keywords
        keywords_master = {
            "msc",
            "m sc",
            "m.sc",
            "m.sc.",
            "master",
            "m.a",
            "mba",
            "postgraduate",
            "graduate",
        }
        keywords_bachelor = {"bachelor", "b.sc", "undergraduate", "b.a"}
        
        # Detect specific program name
        keywords_industry = {"industry 4.0", "industry4.0", "industry 4", "i4.0"}
        keywords_data_science = {"data science", "datascience"}
        keywords_entrepreneurship = {"entrepreneurship"}
        keywords_business = {"digital business", "digitalbusiness"}
        
        has_master_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_master)
        has_bachelor_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_bachelor)
        has_industry_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_industry)
        has_data_science_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_data_science)
        has_entrepreneurship_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_entrepreneurship)
        has_business_intent = any(kw in question_lower or kw in question_normalized for kw in keywords_business)

        def _score_value(value: object) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            return 0.0
        
        def relevance_boost(hit: dict[str, object]) -> float:
            source_url = str(hit.get("source_url", "")).lower()
            base_score = _score_value(hit.get("score", 0.0))
            adjustment = 0.0
            
            # Smart program-level matching
            if has_master_intent:
                if "/master-programs/" in source_url:
                    adjustment += 0.15
                elif "/bachelor/" in source_url:
                    adjustment -= 0.10
                else:
                    adjustment -= 0.06
            
            if has_bachelor_intent:
                if "/bachelor/" in source_url:
                    adjustment += 0.15
                elif "/master-programs/" in source_url:
                    adjustment -= 0.10
            
            # Specific program matching (higher priority than level)
            if has_industry_intent:
                if "industry" in source_url and ("4" in source_url or "4-0" in source_url):
                    adjustment += 0.20
                else:
                    adjustment -= 0.08
            if has_data_science_intent:
                if "data-science" in source_url or "datascience" in source_url:
                    adjustment += 0.20
            if has_entrepreneurship_intent:
                if "entrepreneurship" in source_url:
                    adjustment += 0.20
            if has_business_intent:
                if "business" in source_url or "digital" in source_url:
                    adjustment += 0.15
            
            # De-prioritize generic/legal pages
            if any(section in source_url for section in ["imprint", "data-policy", "terms", "privacy", "cookie", "careers"]):
                adjustment -= 0.15
            
            # De-prioritize fragments/anchors if better full pages are available
            if "#" in source_url and any("#" not in str(alt.get("source_url", "")) for alt in hits):
                adjustment -= 0.05
            
            return base_score + adjustment
        
        # Re-sort by adjusted scores
        reranked = sorted(hits, key=relevance_boost, reverse=True)
        return reranked

    def _augment_candidates_with_url_matches(
        self,
        candidates: list[dict[str, object]],
        question: str,
    ) -> list[dict[str, object]]:
        """Fallback: add URL-term matched candidates when vector recall misses obvious program pages."""
        tokens = re.findall(r"[a-z0-9]+", question.lower())
        stopwords = {
            "how",
            "much",
            "is",
            "the",
            "of",
            "a",
            "an",
            "program",
            "fee",
            "fees",
            "tuition",
            "cost",
            "costs",
            "for",
        }
        terms = [t for t in tokens if t not in stopwords and (len(t) >= 3 or t.isdigit())]
        if not terms:
            return candidates

        existing_urls = {str(item.get("source_url", "")) for item in candidates}
        by_url: dict[str, Chunk] = {}
        for chunk in self._chunks:
            if chunk.source_url not in by_url:
                by_url[chunk.source_url] = chunk

        augmented = list(candidates)
        for source_url, chunk in by_url.items():
            if source_url in existing_urls:
                continue

            lowered = source_url.lower()
            matched = sum(1 for term in terms if term in lowered)
            if matched == 0:
                continue

            ratio = matched / max(len(terms), 1)
            if ratio < 0.34:
                continue

            pseudo_score = 0.18 + (0.22 * ratio)
            augmented.append(
                {
                    "score": round(pseudo_score, 4),
                    "source_url": source_url,
                    "title": chunk.title,
                    "source_type": "pdf" if self._is_pdf_url(source_url) else "webpage",
                    "content": chunk.text,
                }
            )

        return augmented

    def _best_chunk_for_source(self, source_url: str, question: str, fallback: str) -> str:
        """Pick the most answerable chunk for a source URL and question intent."""
        if not source_url:
            return fallback

        source_chunks = [chunk.text for chunk in self._chunks if chunk.source_url == source_url]
        if not source_chunks:
            return fallback

        question_lower = question.lower()
        fee_intent = any(keyword in question_lower for keyword in ("fee", "fees", "tuition", "cost", "costs", "price", "pricing"))
        bachelor_list_intent = (
            "bachelor" in question_lower
            and any(term in question_lower for term in ("program", "offered", "available", "which", "what"))
        )
        query_tokens = [token for token in re.findall(r"[a-z0-9]+", question_lower) if len(token) >= 3]

        best_text = fallback
        best_score = -1.0
        best_index = -1
        for index, text in enumerate(source_chunks):
            lowered = text.lower()
            score = 0.0

            if query_tokens:
                score += sum(1.0 for token in query_tokens if token in lowered) / len(query_tokens)

            if bachelor_list_intent:
                if any(term in lowered for term in ("bachelor", "program", "on campus", "years |", "skills for tomorrow")):
                    score += 0.8
                # Promote chunks likely containing the full program list section.
                score += min(lowered.count(" on campus"), 5) * 0.25

            if fee_intent:
                if any(term in lowered for term in ("fee", "fees", "tuition", "cost", "costs", "per month", "per year")):
                    score += 2.0
                if "€" in text or "eur" in lowered or "euro" in lowered:
                    score += 2.5
                if re.search(r"\b\d{3,}(?:[\.,]\d+)?\b", text):
                    score += 1.0

            if score > best_score:
                best_score = score
                best_text = text
                best_index = index

        if bachelor_list_intent and "/bachelor/" in source_url and best_index >= 0:
            start = max(0, best_index - 1)
            end = min(len(source_chunks), best_index + 2)
            merged = " ".join(source_chunks[start:end]).strip()
            if merged:
                return self._clean_bachelor_program_excerpt(merged)

        return best_text

    def _clean_bachelor_program_excerpt(self, text: str) -> str:
        """Trim navigation/menu noise and keep bachelor program list section only."""
        compact = " ".join(text.split())
        lowered = compact.lower()

        start_markers = [
            "bachelor programs through learning by doing",
            "home | bachelor bachelor programs",
        ]
        start = -1
        for marker in start_markers:
            idx = lowered.find(marker)
            if idx != -1:
                start = idx
                break

        if start != -1:
            compact = compact[start:]
            lowered = compact.lower()

        end_markers = [
            "start your journey today",
            "partial scholarships",
            "study with us",
        ]
        end_positions = [lowered.find(marker) for marker in end_markers if lowered.find(marker) != -1]
        if end_positions:
            compact = compact[: min(end_positions)]

        return compact.strip()

    def _distance_to_score(self, distance: Any) -> float:
        try:
            dist = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if dist < 0:
            dist = 0.0
        return 1.0 / (1.0 + dist)

    def _ensure_index(self) -> None:
        """Ensure index is loaded. Does NOT auto-rebuild - must call rebuild_index() manually."""
        with self._lock:
            needs_load = self._vectorstore is None or not self._chunks
            if needs_load:
                if not self._load_index():
                    raise RuntimeError(
                        f"Knowledge base index not found at {self.index_path}. "
                        "Run 'python -m student_service.rebuild_kb' to build it."
                    )

    def _build_index_from_cache(self) -> bool:
        """Load chunks from cached .txt file if available. Returns True if successful."""
        if not self.collected_txt_path.exists():
            return False
        
        try:
            with open(self.collected_txt_path, "r") as f:
                content = f.read()
            
            chunks: list[Chunk] = []
            
            # Parse format with "--- Chunk NNN ---" markers
            import re
            pattern = r"--- Chunk \d+ ---\nURL: (.+?)\nTitle: (.+?)\nContent:\n(.+?)(?=--- Chunk|\Z)"
            
            for match in re.finditer(pattern, content, re.DOTALL):
                source_url = match.group(1).strip()
                title = match.group(2).strip()
                text = match.group(3).strip()
                
                if text and source_url:
                    chunks.append(Chunk(text=text, source_url=source_url, title=title))
            
            if chunks:
                self._chunks = chunks
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to load chunks from cache: {e}")
            return False

    def _build_index_locked(self) -> None:
        pages = self._collect_pages()
        chunks: list[Chunk] = []

        for url in pages:
            try:
                response = self._session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException:
                continue

            if self._is_pdf_response(url, response):
                title = self._pdf_title_from_url(url)
                body = self._extract_pdf_text(response.content)
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                title = _normalize_whitespace(soup.title.get_text(" ")) if soup.title else ""
                body = _normalize_whitespace(soup.get_text(" "))

            if not body:
                continue

            for piece in _split_text(body):
                chunk_text = _normalize_whitespace(f"{title} {piece}" if title else piece)
                chunks.append(Chunk(text=chunk_text, source_url=url, title=title))

        self._chunks = chunks
        if not chunks:
            self._vectorstore = None
            self._last_indexed_at = time.time()
            return

        self._init_vectorstore(reset=True)
        self._upsert_chunks(self._chunks)

        self._last_indexed_at = time.time()
        
        # Save index to disk
        self._save_index()

    def _init_vectorstore(self, reset: bool = False) -> None:
        """Initialize LangChain Chroma vector store with local embeddings."""
        vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.vector_db_path),
            embedding_function=self._embeddings,
        )

        if reset:
            try:
                vectorstore.delete_collection()
            except Exception:
                pass
            vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.vector_db_path),
                embedding_function=self._embeddings,
            )

        self._vectorstore = vectorstore

    def _upsert_chunks(self, chunks: list[Chunk]) -> None:
        if self._vectorstore is None:
            raise RuntimeError("Vector store is not initialized")

        batch_size = 200
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunk_batch = chunks[start:end]
            ids = [f"chunk-{i}" for i in range(start, min(end, len(chunks)))]
            documents = [
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source_url": chunk.source_url,
                        "title": chunk.title,
                        "source_type": "pdf" if self._is_pdf_url(chunk.source_url) else "webpage",
                    },
                )
                for chunk in chunk_batch
            ]
            self._vectorstore.add_documents(documents=documents, ids=ids)

    def _init_chroma_from_chunks(self) -> None:
        """Initialize vector store from pre-loaded self._chunks."""
        if not self._chunks:
            self._vectorstore = None
            return

        self._init_vectorstore(reset=True)
        self._upsert_chunks(self._chunks)

    def _collect_pages(self) -> list[str]:
        discovered = set(self._from_sitemap())
        discovered.update(self._from_homepage_links())

        # Multi-level page discovery
        discovered.update(self._discover_all_links(discovered, max_depth=3))

        # Comprehensive PDF discovery
        discovered.update(self._discover_pdf_links(discovered))

        urls = sorted(discovered, key=self._url_priority_key)
        return urls

    def _url_priority_key(self, url: str) -> tuple[int, str]:
        lowered = url.lower()

        if "/master-programs/" in lowered:
            return (0, lowered)
        if "/master/" in lowered:
            return (1, lowered)
        if "/bachelor/" in lowered:
            return (2, lowered)
        if "/wp-content/uploads/" in lowered and lowered.endswith(".pdf"):
            return (3, lowered)
        if any(segment in lowered for segment in ("/about", "/contact", "/application")):
            return (6, lowered)
        return (4, lowered)

    def _from_sitemap(self) -> Iterable[str]:
        candidates: list[str] = [
            urljoin(self.base_url, "/sitemap.xml"),
            urljoin(self.base_url, "/sitemap_index.xml"),
        ]

        visited: set[str] = set()
        while candidates:
            sitemap_url = candidates.pop(0)
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)

            try:
                response = self._session.get(sitemap_url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException:
                continue

            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                continue

            namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
            is_sitemap_index = root.tag.endswith("sitemapindex")

            if is_sitemap_index:
                for loc in root.findall(f".//{namespace}loc"):
                    if not loc.text:
                        continue
                    nested = loc.text.strip()
                    if nested.endswith(".xml"):
                        candidates.append(nested)
                continue

            for loc in root.findall(f".//{namespace}loc"):
                if not loc.text:
                    continue
                candidate = loc.text.strip()
                if self._is_allowed_url(candidate):
                    yield candidate

    def _from_homepage_links(self) -> Iterable[str]:
        try:
            response = self._session.get(self.base_url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links: set[str] = set()
        for a in soup.find_all("a", href=True):
            href_value = a.get("href", "")
            href = href_value.strip() if isinstance(href_value, str) else ""
            if not href:
                continue
            full_url = urljoin(self.base_url, href)
            if self._is_allowed_url(full_url):
                links.add(full_url)
        return links

    def _discover_all_links(self, seed_urls: Iterable[str], max_depth: int = 3) -> set[str]:
        """Recursively discover all links from seed pages up to max_depth."""
        discovered: set[str] = set()
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(url, 0) for url in seed_urls if not self._is_pdf_url(url)]

        while queue:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            try:
                response = self._session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href_value = a.get("href", "")
                href = href_value.strip() if isinstance(href_value, str) else ""
                if not href:
                    continue
                full_url = urljoin(self.base_url, href)
                if self._is_allowed_url(full_url) and full_url not in visited:
                    discovered.add(full_url)
                    if not self._is_pdf_url(full_url) and depth < max_depth:
                        queue.append((full_url, depth + 1))

        return discovered

    def _discover_pdf_links(self, urls: Iterable[str], max_scan_pages: int = 100) -> set[str]:
        discovered: set[str] = set()
        html_urls = [url for url in urls if not self._is_pdf_url(url)]

        for url in html_urls[:max_scan_pages]:
            try:
                response = self._session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href_value = a.get("href", "")
                href = href_value.strip() if isinstance(href_value, str) else ""
                if not href:
                    continue
                full_url = urljoin(self.base_url, href)
                if self._is_allowed_url(full_url) and self._is_pdf_url(full_url):
                    discovered.add(full_url)
        return discovered

    def _is_allowed_url(self, value: str) -> bool:
        parsed_base = urlparse(self.base_url)
        parsed = urlparse(value)
        if not parsed.scheme.startswith("http"):
            return False
        if parsed.netloc != parsed_base.netloc:
            return False

        lowered = value.lower()
        if lowered.endswith(".xml"):
            return False
        if any(lowered.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
            return False
        return True

    def _is_pdf_response(self, url: str, response: requests.Response) -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        return self._is_pdf_url(url) or "application/pdf" in content_type

    def _is_pdf_url(self, url: str) -> bool:
        return url.lower().endswith(".pdf")

    def _extract_pdf_text(self, content: bytes) -> str:
        if not content.lstrip().startswith(b"%PDF"):
            return ""

        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception:
            return ""

        parts: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                parts.append(text)

        return _normalize_whitespace("\n".join(parts))

    def _pdf_title_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        filename = parsed.path.rsplit("/", 1)[-1]
        if filename.lower().endswith(".pdf"):
            filename = filename[:-4]
        title = filename.replace("-", " ").replace("_", " ")
        return _normalize_whitespace(title)

    def _save_index(self) -> bool:
        """Save index metadata to disk (vector data is persisted in ChromaDB path)."""
        try:
            index_data: dict[str, Any] = {
                "chunks": self._chunks,
                "last_indexed_at": self._last_indexed_at,
                "base_url": self.base_url,
                "vector_db_path": str(self.vector_db_path),
                "collection_name": self.collection_name,
                "version": "3.0",
                "backend": "langchain_chroma",
            }
            
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.index_path, "wb") as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
        except Exception as e:
            print(f"Warning: Failed to save index to {self.index_path}: {e}")
            return False

    def _save_collected_txt(self) -> bool:
        """Save all collected/retrieved chunks to a readable text file."""
        try:
            self.collected_txt_path.parent.mkdir(parents=True, exist_ok=True)

            unique_urls = sorted({chunk.source_url for chunk in self._chunks})

            lines: list[str] = [
                f"Knowledge base collected content export",
                f"Base URL: {self.base_url}",
                f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._last_indexed_at))}",
                f"Total URLs: {len(unique_urls)}",
                f"Total chunks: {len(self._chunks)}",
                "",
                "=== COLLECTED URLS ===",
            ]

            lines.extend(unique_urls)
            lines.append("")
            lines.append("=== COLLECTED CHUNKS ===")
            lines.append("")

            for idx, chunk in enumerate(self._chunks, start=1):
                lines.append(f"--- Chunk {idx} ---")
                lines.append(f"URL: {chunk.source_url}")
                lines.append(f"Title: {chunk.title}")
                lines.append("Content:")
                lines.append(chunk.text)
                lines.append("")

            with open(self.collected_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True
        except Exception as e:
            print(f"Warning: Failed to save collected txt to {self.collected_txt_path}: {e}")
            return False

    def _load_index(self) -> bool:
        """Load the index from disk. Returns True if successful."""
        if not self.index_path.exists():
            return False
        
        try:
            with open(self.index_path, "rb") as f:
                index_data = pickle.load(f)
            
            # Validate version and base URL
            if index_data.get("base_url") != self.base_url:
                print(f"Warning: Index built for different URL ({index_data.get('base_url')}), ignoring.")
                return False
            
            self._chunks = index_data["chunks"]
            self._last_indexed_at = index_data["last_indexed_at"]

            stored_db_path = index_data.get("vector_db_path")
            if isinstance(stored_db_path, str) and stored_db_path:
                self.vector_db_path = Path(stored_db_path)

            stored_collection_name = index_data.get("collection_name")
            if isinstance(stored_collection_name, str) and stored_collection_name:
                self.collection_name = stored_collection_name

            if not self.vector_db_path.exists():
                return False

            self._init_vectorstore(reset=False)
            if self._vectorstore is None:
                return False

            collection_any = cast(Any, self._vectorstore)._collection
            if collection_any is None or int(collection_any.count()) == 0:
                return False
            
            return True
        except Exception as e:
            print(f"Warning: Failed to load index from {self.index_path}: {e}")
            return False

    def rebuild_index(self, verbose: bool = True) -> dict[str, Any]:
        """
        Manually rebuild the entire knowledge base index from scratch.
        
        This should be run by developers when:
        - The website content has been updated
        - The indexing logic has changed
        - The index file is corrupted or missing
        
        Returns a dict with build statistics.
        """
        if verbose:
            print(f"Rebuilding knowledge base for {self.base_url}...")
        
        start_time = time.time()
        
        with self._lock:
            self._build_index_locked()

        txt_saved = self._save_collected_txt()
        
        elapsed = time.time() - start_time
        
        stats: dict[str, Any] = {
            "total_chunks": len(self._chunks),
            "total_urls": len(set(c.source_url for c in self._chunks)),
            "pdf_urls": len(set(c.source_url for c in self._chunks if c.source_url.endswith(".pdf"))),
            "html_urls": len(set(c.source_url for c in self._chunks if not c.source_url.endswith(".pdf"))),
            "elapsed_seconds": round(elapsed, 1),
            "index_path": str(self.index_path),
            "saved": self.index_path.exists(),
            "vector_db_path": str(self.vector_db_path),
            "collected_txt_path": str(self.collected_txt_path),
            "collected_txt_saved": txt_saved,
        }
        
        if verbose:
            print(f"✓ Indexed {stats['total_urls']} URLs ({stats['html_urls']} HTML + {stats['pdf_urls']} PDF)")
            print(f"✓ Created {stats['total_chunks']} text chunks")
            print(f"✓ Completed in {stats['elapsed_seconds']}s")
            print(f"✓ Saved to {stats['index_path']}")
            print(f"✓ Vector DB persisted at {stats['vector_db_path']}")
            if stats["collected_txt_saved"]:
                print(f"✓ Exported collected content to {stats['collected_txt_path']}")
        
        return stats

    def rebuild_index_from_cache(self, verbose: bool = True) -> dict[str, Any]:
        """
        Quick rebuild from existing cached .txt file (no web crawling).
        Used when network is unstable or for testing vector DB initialization.
        """
        if verbose:
            print("Rebuilding knowledge base from cache...")
        
        start_time = time.time()
        
        with self._lock:
            if not self._build_index_from_cache():
                raise RuntimeError(f"No cached index found at {self.collected_txt_path}")
            
            self._init_chroma_from_chunks()
            self._save_index()
        
        elapsed = time.time() - start_time
        
        stats: dict[str, Any] = {
            "total_chunks": len(self._chunks),
            "total_urls": len(set(c.source_url for c in self._chunks)),
            "pdf_urls": len(set(c.source_url for c in self._chunks if c.source_url.endswith(".pdf"))),
            "html_urls": len(set(c.source_url for c in self._chunks if not c.source_url.endswith(".pdf"))),
            "elapsed_seconds": round(elapsed, 1),
            "index_path": str(self.index_path),
            "saved": self.index_path.exists(),
            "vector_db_path": str(self.vector_db_path),
            "from_cache": True,
        }
        
        if verbose:
            print(f"✓ Loaded {stats['total_chunks']} chunks from cache")
            print(f"✓ Computed embeddings and initialized Chroma")
            print(f"✓ Completed in {stats['elapsed_seconds']}s")
            print(f"✓ Vector DB persisted at {stats['vector_db_path']}")
        
        return stats
