import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import config


class VectorStore:
    def __init__(
        self,
        knowledge_base_dir: Path = config.KNOWLEDGE_BASE_DIR,
        resolved_cases_path: Path = config.RESOLVED_CASES_PATH,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: str = config.DEVICE,
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.resolved_cases_path = resolved_cases_path
        self.device = device

        print(f"[VectorStore] Loading embedding model '{model_name}' on device '{device}'...")
        kwargs = {}
        if config.HF_TOKEN:
            kwargs["token"] = config.HF_TOKEN
        self.model = SentenceTransformer(model_name, device=device, **kwargs)


        self.passages: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

        self._build_index()

    def _chunk_markdown(self, filename: str, content: str) -> List[Dict[str, Any]]:
        """Splits markdown file content into logical header sections."""
        passages = []
        lines = content.splitlines()

        # Parse YAML frontmatter if present
        doc_id = filename
        title = filename
        frontmatter_end = -1
        if content.startswith("---"):
            try:
                second_dash = content.find("---", 3)
                if second_dash != -1:
                    fm_text = content[3:second_dash]
                    for fm_line in fm_text.splitlines():
                        if fm_line.startswith("document_id:"):
                            doc_id = fm_line.split(":", 1)[1].strip()
                        elif fm_line.startswith("title:"):
                            title = fm_line.split(":", 1)[1].strip()
                    frontmatter_end = second_dash + 3
            except Exception:
                pass

        body = content[frontmatter_end:].strip() if frontmatter_end != -1 else content

        # Split by level 1 & 2 headers
        sections = []
        current_header = title
        current_buffer = []

        for line in body.splitlines():
            if line.startswith("# ") or line.startswith("## "):
                if current_buffer:
                    sections.append((current_header, "\n".join(current_buffer).strip()))
                    current_buffer = []
                current_header = line.lstrip("#").strip()
            else:
                current_buffer.append(line)

        if current_buffer:
            sections.append((current_header, "\n".join(current_buffer).strip()))

        # If splitting resulted in empty sections, fall back to full body
        if not sections:
            sections = [(title, body)]

        for section_header, section_text in sections:
            if not section_text.strip():
                continue
            full_passage = f"Document: {title} ({doc_id})\nSection: {section_header}\nContent:\n{section_text}"
            passages.append({
                "source_id": doc_id,
                "file_name": filename,
                "title": title,
                "section": section_header,
                "source_type": "documentation",
                "passage": full_passage,
                "superseded": False,
            })

        return passages

    def _build_index(self):
        """Loads knowledge base markdown files and resolved cases into memory."""
        self.passages.clear()

        # 1. Load Markdown Files from Knowledge Base
        if self.knowledge_base_dir.exists():
            md_files = sorted(list(self.knowledge_base_dir.glob("*.md")))
            for filepath in md_files:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = self._chunk_markdown(filepath.name, content)
                self.passages.extend(chunks)

        # 2. Load Resolved Cases
        if self.resolved_cases_path.exists():
            with open(self.resolved_cases_path, "r", encoding="utf-8") as f:
                cases_data = json.load(f)

            for case in cases_data.get("cases", []):
                case_id = case.get("case_id", "UNKNOWN_CASE")
                title = case.get("title", "")
                status = case.get("status", "resolved")
                is_superseded = (status == "superseded")
                symptoms = "\n- ".join(case.get("symptoms", []))
                resolution = "\n- ".join(case.get("resolution", []))
                important_limit = case.get("important_limit", "")
                superseded_reason = case.get("superseded_reason", "")

                passage_lines = [
                    f"Resolved Case: {case_id} - {title}",
                    f"Status: {status}",
                    f"Symptoms:\n- {symptoms}" if symptoms else "",
                    f"Resolution:\n- {resolution}" if resolution else "",
                ]
                if important_limit:
                    passage_lines.append(f"Important Limit: {important_limit}")
                if superseded_reason:
                    passage_lines.append(f"Superseded Reason: {superseded_reason}")

                full_passage = "\n".join([line for line in passage_lines if line])

                self.passages.append({
                    "source_id": case_id,
                    "file_name": "resolved_cases.json",
                    "title": title,
                    "section": "Resolved Case Summary",
                    "source_type": "resolved_case",
                    "passage": full_passage,
                    "superseded": is_superseded,
                })

        print(f"[VectorStore] Total passages loaded for indexing: {len(self.passages)}")

        # 3. Compute Embeddings
        if self.passages:
            texts = [p["passage"] for p in self.passages]
            encoded = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            self.embeddings = F.normalize(encoded, p=2, dim=1)
        else:
            self.embeddings = None

    def search(self, query: str, top_k: int = 5, include_superseded: bool = False) -> List[Dict[str, Any]]:
        """Performs cosine similarity search against index."""
        if not self.passages or self.embeddings is None:
            return []

        query_vec = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        query_norm = F.normalize(query_vec, p=2, dim=0)

        # Compute cosine similarity
        similarities = torch.mv(self.embeddings, query_norm)
        scores, indices = torch.topk(similarities, k=min(top_k * 2, len(self.passages)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            item = dict(self.passages[idx])
            item["score"] = round(float(score), 4)

            if not include_superseded and item["superseded"]:
                continue

            results.append(item)
            if len(results) >= top_k:
                break

        return results


# Module-level store instance and helper function
_global_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    global _global_store
    if _global_store is None:
        _global_store = VectorStore()
    return _global_store

def retrieve_context(query: str, top_k: int = 5, include_superseded: bool = False) -> List[Dict[str, Any]]:
    """Helper function to retrieve context for a query."""
    store = get_vector_store()
    return store.search(query=query, top_k=top_k, include_superseded=include_superseded)
