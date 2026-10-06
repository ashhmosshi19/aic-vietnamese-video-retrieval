#!/usr/bin/env python3
"""Baseline suite for Vietnamese news-video query-to-keyframe retrieval.

The default path is intentionally dependency-light and safe for small smoke
tests: it builds a keyframe manifest, attaches available text evidence, runs
BM25 over ASR/OCR/caption/metadata channels, fuses scores, and writes metrics
and hubness artifacts. Model-backed visual and dense retrieval are explicit
opt-in paths so the script remains runnable in a fresh project-local env.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = REPO_ROOT / "runs"
DEFAULT_KEYFRAMES_ROOT = REPO_ROOT / "data" / "keyframes_1fps"
DEFAULT_ASR_JSON = REPO_ROOT / "data" / "asr.json"
DEFAULT_OCR_JSON = REPO_ROOT / "data" / "ocr.json"
DEFAULT_SEGMENTS_CSV = REPO_ROOT / "data" / "segments.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CHANNELS = ("visual", "asr", "ocr", "caption", "metadata", "merged")


@dataclass
class KeyframeDoc:
    doc_id: str
    video_id: str
    frame_id: str
    image_path: str
    timestamp_sec: float | None
    segment_id: str | None = None
    segment_start_sec: float | None = None
    segment_end_sec: float | None = None
    asr_text: str = ""
    ocr_text: str = ""
    caption_text: str = ""
    metadata_text: str = ""

    @property
    def merged_text(self) -> str:
        return " ".join(
            text
            for text in (
                self.caption_text,
                self.asr_text,
                self.ocr_text,
                self.metadata_text,
            )
            if text
        )


@dataclass
class QuerySpec:
    query_id: str
    text: str
    relevant_video_ids: set[str]
    relevant_segment_ids: set[str]
    relevant_frame_ids: set[str]
    relevant_doc_ids: set[str]
    intervals: list[dict[str, Any]]


class SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.avgdl = 0.0
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self._initialize(corpus_tokens)

    def _initialize(self, corpus_tokens: list[list[str]]) -> None:
        df: Counter[str] = Counter()
        total_len = 0
        for tokens in corpus_tokens:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            df.update(freqs.keys())
        self.avgdl = total_len / self.corpus_size if self.corpus_size else 0.0
        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str]) -> list[float]:
        if not self.corpus_size or not query_tokens or self.avgdl <= 0:
            return [0.0] * self.corpus_size
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                dl = self.doc_lens[idx]
                norm = 1.0 - self.b + self.b * (dl / self.avgdl)
                scores[idx] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes-root", type=Path, default=DEFAULT_KEYFRAMES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--collection-limit", type=int, default=1)
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--queries-jsonl", type=Path, default=None)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--asr-json", type=Path, default=DEFAULT_ASR_JSON)
    parser.add_argument("--asr-csv", type=Path, default=None)
    parser.add_argument("--ocr-json", type=Path, default=DEFAULT_OCR_JSON)
    parser.add_argument("--captions-json", type=Path, default=None)
    parser.add_argument("--metadata-jsonl", type=Path, default=None)
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hubness-k", type=int, default=10)
    parser.add_argument("--weights", default="visual=0.50,asr=0.25,ocr=0.15,caption=0.10")
    parser.add_argument("--run-visual-model", action="store_true")
    parser.add_argument("--visual-model-name", default="google/siglip2-base-patch16-384")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--allow-placeholder-visual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Output path must stay inside repo: {resolved}") from exc
    return resolved


def default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return EXPERIMENT_ROOT / "runs" / stamp


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: Any) -> list[str]:
    return normalize_text(text).split()


def parse_weights(raw: str) -> dict[str, float]:
    weights = {name: 0.0 for name in CHANNELS}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        name = name.strip().lower()
        if name not in weights:
            raise ValueError(f"Unknown score channel in --weights: {name}")
        weights[name] = float(value)
    return weights


def infer_video_id(path: Path) -> str:
    if re.fullmatch(r"K\d+_V\d+", path.parent.name):
        return path.parent.name
    match = re.search(r"(K\d+_V\d+)", str(path))
    return match.group(1) if match else path.parent.name


def infer_timestamp(path: Path) -> float | None:
    stem = path.stem
    if stem.isdigit():
        return max(0.0, float(int(stem) - 1))
    match = re.search(r"frame_(\d+)", stem)
    if match:
        raw = int(match.group(1))
        return raw / 1000.0 if raw > 10000 else float(raw)
    return None


def collect_keyframes(root: Path, collection_limit: int, video_limit: int, frames_per_video: int) -> list[KeyframeDoc]:
    if not root.exists():
        raise FileNotFoundError(f"Keyframes root not found: {root}")
    collection_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if collection_limit > 0:
        collection_dirs = collection_dirs[:collection_limit]

    docs: list[KeyframeDoc] = []
    for collection_dir in collection_dirs:
        video_dirs = [p for p in sorted(collection_dir.iterdir()) if p.is_dir()]
        if video_limit > 0:
            video_dirs = video_dirs[:video_limit]
        for video_dir in video_dirs:
            frames = [
                p
                for p in sorted(video_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
            if frames_per_video > 0:
                frames = frames[:frames_per_video]
            for frame_path in frames:
                video_id = infer_video_id(frame_path)
                frame_id = frame_path.stem
                doc_id = f"{video_id}:{frame_id}"
                docs.append(
                    KeyframeDoc(
                        doc_id=doc_id,
                        video_id=video_id,
                        frame_id=frame_id,
                        image_path=str(frame_path),
                        timestamp_sec=infer_timestamp(frame_path),
                        metadata_text=f"{video_id} keyframe {frame_id}",
                    )
                )
    return docs


def load_segments(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ("start_sec", "end_sec", "start_time", "end_time", "timestamp"):
                if key in parsed and parsed[key] not in ("", None):
                    try:
                        parsed[key] = float(parsed[key])
                    except ValueError:
                        pass
            rows.append(parsed)
    return rows


def attach_segments(docs: list[KeyframeDoc], segments: list[dict[str, Any]]) -> None:
    if not segments:
        return
    for doc in docs:
        if doc.timestamp_sec is None:
            continue
        for row in segments:
            start = row.get("start_sec")
            if start is None:
                start = row.get("start_time")
            end = row.get("end_sec")
            if end is None:
                end = row.get("end_time")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if float(start) <= doc.timestamp_sec <= float(end):
                doc.segment_id = str(row.get("scene_id") or row.get("segment_id") or row.get("shot_id") or "")
                doc.segment_start_sec = float(start)
                doc.segment_end_sec = float(end)
                break


def text_for_time(items: list[dict[str, Any]], timestamp_sec: float | None, window_sec: float = 5.0) -> str:
    if timestamp_sec is None:
        return ""
    parts = []
    for item in items:
        start = item.get("start_time", item.get("start", item.get("start_sec")))
        end = item.get("end_time", item.get("end", item.get("end_sec")))
        text = item.get("text") or item.get("word") or item.get("transcript") or ""
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f >= timestamp_sec - window_sec and start_f <= timestamp_sec + window_sec and text:
            parts.append(str(text))
    return " ".join(parts)


def load_asr_items(json_path: Path | None, csv_path: Path | None) -> list[dict[str, Any]]:
    if json_path and json_path.exists():
        data = read_json(json_path)
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return [dict(item) for item in data["segments"]]
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
    if csv_path and csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    return []


def load_ocr_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    data = read_json(path)
    rows = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    mapping: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("ocr_text") or row.get("text") or ""
        keys = [
            row.get("image_path"),
            row.get("keyframe_id"),
            Path(str(row.get("image_path", ""))).name if row.get("image_path") else None,
        ]
        for key in keys:
            if key:
                mapping[str(key)] = str(text)
    return mapping


def load_text_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        mapping = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            key = row.get("doc_id") or row.get("image_path") or row.get("keyframe_id") or row.get("frame_id")
            text = row.get("caption") or row.get("caption_text") or row.get("text") or row.get("metadata_text")
            if key and text:
                mapping[str(key)] = str(text)
        return mapping
    return {}


def lookup_text(mapping: dict[str, str], doc: KeyframeDoc) -> str:
    for key in (doc.doc_id, doc.image_path, Path(doc.image_path).name, doc.frame_id):
        if key in mapping:
            return mapping[key]
    return ""


def attach_evidence(
    docs: list[KeyframeDoc],
    asr_items: list[dict[str, Any]],
    ocr_map: dict[str, str],
    caption_map: dict[str, str],
    metadata_map: dict[str, str],
) -> None:
    for doc in docs:
        doc.asr_text = text_for_time(asr_items, doc.timestamp_sec)
        doc.ocr_text = lookup_text(ocr_map, doc)
        doc.caption_text = lookup_text(caption_map, doc)
        doc.metadata_text = " ".join(part for part in (doc.metadata_text, lookup_text(metadata_map, doc)) if part)


def load_queries(args: argparse.Namespace) -> list[QuerySpec]:
    raw_queries: list[dict[str, Any]] = []
    if args.queries_jsonl and args.queries_jsonl.exists():
        with args.queries_jsonl.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_queries.append(json.loads(line))
    for idx, text in enumerate(args.query, start=1):
        raw_queries.append({"query_id": f"cli_{idx}", "text": text})
    if not raw_queries:
        raw_queries.append(
            {
                "query_id": "smoke_1",
                "text": "bản tin thời sự Việt Nam người dẫn chương trình",
            }
        )

    labels_by_id: dict[str, dict[str, Any]] = {}
    if args.labels_jsonl and args.labels_jsonl.exists():
        with args.labels_jsonl.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = str(row.get("query_id") or row.get("id") or "")
                if qid:
                    labels_by_id[qid] = row

    queries = []
    for idx, row in enumerate(raw_queries, start=1):
        qid = str(row.get("query_id") or row.get("id") or f"q{idx}")
        label = labels_by_id.get(qid, row)
        gt = label.get("ground_truth", label)
        intervals = gt.get("acceptable_intervals", gt.get("intervals", [])) if isinstance(gt, dict) else []
        video_ids = set(str(x) for x in gt.get("video_ids", []) if x) if isinstance(gt, dict) else set()
        segment_ids = set(str(x) for x in gt.get("segment_ids", []) if x) if isinstance(gt, dict) else set()
        frame_ids = set(str(x) for x in gt.get("frame_ids", []) if x) if isinstance(gt, dict) else set()
        doc_ids = set(str(x) for x in gt.get("doc_ids", []) if x) if isinstance(gt, dict) else set()
        for interval in intervals if isinstance(intervals, list) else []:
            if isinstance(interval, dict) and interval.get("video_id"):
                video_ids.add(str(interval["video_id"]))
        queries.append(
            QuerySpec(
                query_id=qid,
                text=str(row.get("text") or row.get("query") or row.get("query_text") or ""),
                relevant_video_ids=video_ids,
                relevant_segment_ids=segment_ids,
                relevant_frame_ids=frame_ids,
                relevant_doc_ids=doc_ids,
                intervals=intervals if isinstance(intervals, list) else [],
            )
        )
    return queries


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def build_bm25_scores(docs: list[KeyframeDoc], query: str, field: str) -> list[float]:
    texts = [getattr(doc, field) for doc in docs]
    corpus = [tokenize(text) for text in texts]
    return SimpleBM25(corpus).score(tokenize(query))


def placeholder_visual_scores(docs: list[KeyframeDoc], query: str) -> list[float]:
    """Deterministic placeholder used only when explicitly requested.

    This is not a semantic visual baseline. It exists to exercise the output
    contract when image-text dependencies are unavailable.
    """

    scores = []
    for doc in docs:
        digest = hashlib.sha256(f"{query}\t{doc.image_path}".encode("utf-8")).digest()
        scores.append(int.from_bytes(digest[:8], "big") / float(2**64 - 1))
    return scores


def try_visual_model_scores(
    docs: list[KeyframeDoc],
    query: str,
    model_name: str,
    device: str,
) -> tuple[list[float], str]:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import AutoModel, AutoProcessor  # type: ignore
    except Exception as exc:
        return [0.0] * len(docs), f"missing_dependency: {exc}"

    try:
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        model.to(device)
        model.eval()
        images = [Image.open(doc.image_path).convert("RGB") for doc in docs]
        with torch.no_grad():
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
        image_emb = image_features.cpu().float().numpy()
        text_emb = text_features.cpu().float().numpy()
        image_emb = image_emb / np.maximum(np.linalg.norm(image_emb, axis=1, keepdims=True), 1e-8)
        text_emb = text_emb / np.maximum(np.linalg.norm(text_emb, axis=1, keepdims=True), 1e-8)
        return (image_emb @ text_emb.T).reshape(-1).astype(float).tolist(), "ok"
    except Exception as exc:
        return [0.0] * len(docs), f"model_error: {exc}"


def relevant_at_level(query: QuerySpec, doc: KeyframeDoc, level: str) -> bool:
    if level == "video":
        return bool(query.relevant_video_ids and doc.video_id in query.relevant_video_ids)
    if level == "segment":
        return bool(query.relevant_segment_ids and doc.segment_id in query.relevant_segment_ids)
    if level == "frame":
        return bool(
            (query.relevant_doc_ids and doc.doc_id in query.relevant_doc_ids)
            or (query.relevant_frame_ids and doc.frame_id in query.relevant_frame_ids)
        )
    return False


def interval_iou(doc: KeyframeDoc, intervals: list[dict[str, Any]]) -> float:
    if doc.timestamp_sec is None:
        return 0.0
    pred_start = doc.segment_start_sec if doc.segment_start_sec is not None else doc.timestamp_sec
    pred_end = doc.segment_end_sec if doc.segment_end_sec is not None else doc.timestamp_sec + 1.0
    best = 0.0
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        if interval.get("video_id") and str(interval["video_id"]) != doc.video_id:
            continue
        start = interval.get("start_sec", interval.get("start_time"))
        end = interval.get("end_sec", interval.get("end_time"))
        if start is None and interval.get("start_ms") is not None:
            start = float(interval["start_ms"]) / 1000.0
        if end is None and interval.get("end_ms") is not None:
            end = float(interval["end_ms"]) / 1000.0
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        intersection = max(0.0, min(pred_end, end_f) - max(pred_start, start_f))
        union = (pred_end - pred_start) + (end_f - start_f) - intersection
        best = max(best, intersection / union if union > 0 else 0.0)
    return best


def rank_metrics(queries: list[QuerySpec], rankings: dict[str, list[KeyframeDoc]], level: str) -> dict[str, Any]:
    valid = [q for q in queries if any(relevant_at_level(q, doc, level) for doc in rankings[q.query_id])]
    if not valid:
        return {"available": False, "num_queries": 0}
    result: dict[str, Any] = {"available": True, "num_queries": len(valid)}
    ranks = []
    for q in valid:
        ranking = rankings[q.query_id]
        first_rank = None
        gains = []
        for idx, doc in enumerate(ranking, start=1):
            hit = relevant_at_level(q, doc, level)
            gains.append(1 if hit else 0)
            if hit and first_rank is None:
                first_rank = idx
        if first_rank is not None:
            ranks.append(first_rank)
        for cutoff in (1, 5, 10):
            key = f"R@{cutoff}"
            result[key] = result.get(key, 0.0) + (1.0 if any(gains[:cutoff]) else 0.0)
        result["MRR@10"] = result.get("MRR@10", 0.0) + (
            1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
        )
        dcg = sum(gain / math.log2(pos + 2) for pos, gain in enumerate(gains[:10]))
        ideal_hits = min(sum(gains), 10)
        idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
        result["NDCG@10"] = result.get("NDCG@10", 0.0) + (dcg / idcg if idcg else 0.0)
    for key in ("R@1", "R@5", "R@10", "MRR@10", "NDCG@10"):
        result[key] = result.get(key, 0.0) / len(valid)
    result["median_rank"] = statistics.median(ranks) if ranks else None
    return result


def evaluate(queries: list[QuerySpec], rankings: dict[str, list[KeyframeDoc]]) -> dict[str, Any]:
    metrics = {
        "video_level": rank_metrics(queries, rankings, "video"),
        "segment_level": rank_metrics(queries, rankings, "segment"),
        "exact_frame_level": rank_metrics(queries, rankings, "frame"),
    }
    interval_queries = [q for q in queries if q.intervals]
    if interval_queries:
        ious = [interval_iou(rankings[q.query_id][0], q.intervals) for q in interval_queries if rankings[q.query_id]]
        metrics["temporal_iou_top1"] = {
            "available": True,
            "num_queries": len(ious),
            "mean": sum(ious) / len(ious) if ious else 0.0,
        }
    else:
        metrics["temporal_iou_top1"] = {"available": False, "num_queries": 0}
    return metrics


def run(args: argparse.Namespace) -> int:
    output_dir = ensure_repo_local(args.output_dir or default_run_dir())
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights)
    docs = collect_keyframes(
        root=args.keyframes_root,
        collection_limit=args.collection_limit,
        video_limit=args.video_limit,
        frames_per_video=args.frames_per_video,
    )
    segments = load_segments(args.segments_csv)
    attach_segments(docs, segments)
    attach_evidence(
        docs=docs,
        asr_items=load_asr_items(args.asr_json, args.asr_csv),
        ocr_map=load_ocr_map(args.ocr_json),
        caption_map=load_text_map(args.captions_json),
        metadata_map=load_text_map(args.metadata_jsonl),
    )
    queries = load_queries(args)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "keyframes_root": str(args.keyframes_root),
        "collection_limit": args.collection_limit,
        "video_limit": args.video_limit,
        "frames_per_video": args.frames_per_video,
        "top_k": args.top_k,
        "hubness_k": args.hubness_k,
        "weights": weights,
        "model_names": {
            "visual": args.visual_model_name if args.run_visual_model else None,
            "bm25": "SimpleBM25",
            "dense_text": None,
        },
        "evidence_sources": {
            "asr_json": str(args.asr_json) if args.asr_json else None,
            "asr_csv": str(args.asr_csv) if args.asr_csv else None,
            "ocr_json": str(args.ocr_json) if args.ocr_json else None,
            "captions_json": str(args.captions_json) if args.captions_json else None,
            "metadata_jsonl": str(args.metadata_jsonl) if args.metadata_jsonl else None,
            "segments_csv": str(args.segments_csv) if args.segments_csv else None,
        },
        "dataset_subset": {
            "num_keyframes": len(docs),
            "num_queries": len(queries),
            "videos": sorted({doc.video_id for doc in docs}),
        },
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "manifest.json", [asdict(doc) | {"merged_text": doc.merged_text} for doc in docs])
    write_json(output_dir / "queries.json", [asdict(q) | {
        "relevant_video_ids": sorted(q.relevant_video_ids),
        "relevant_segment_ids": sorted(q.relevant_segment_ids),
        "relevant_frame_ids": sorted(q.relevant_frame_ids),
        "relevant_doc_ids": sorted(q.relevant_doc_ids),
    } for q in queries])

    all_top_rows: list[dict[str, Any]] = []
    rankings_for_eval: dict[str, list[KeyframeDoc]] = {}
    hubness: Counter[str] = Counter()
    model_status: dict[str, str] = {}

    for query in queries:
        channel_scores: dict[str, list[float]] = {
            "asr": build_bm25_scores(docs, query.text, "asr_text"),
            "ocr": build_bm25_scores(docs, query.text, "ocr_text"),
            "caption": build_bm25_scores(docs, query.text, "caption_text"),
            "metadata": build_bm25_scores(docs, query.text, "metadata_text"),
            "merged": build_bm25_scores(docs, query.text, "merged_text"),
        }
        if args.run_visual_model:
            visual_scores, status = try_visual_model_scores(docs, query.text, args.visual_model_name, args.device)
            model_status[f"visual:{query.query_id}"] = status
        elif args.allow_placeholder_visual:
            visual_scores = placeholder_visual_scores(docs, query.text)
            model_status[f"visual:{query.query_id}"] = "placeholder_not_semantic"
        else:
            visual_scores = [0.0] * len(docs)
            model_status[f"visual:{query.query_id}"] = "not_run"
        channel_scores["visual"] = visual_scores

        normalized = {name: min_max_normalize(scores) for name, scores in channel_scores.items()}
        fused = []
        for idx, doc in enumerate(docs):
            score = sum(weights.get(name, 0.0) * normalized[name][idx] for name in CHANNELS)
            fused.append((score, idx, doc))
        fused.sort(key=lambda item: item[0], reverse=True)
        rankings_for_eval[query.query_id] = [doc for _, _, doc in fused]
        for _, _, doc in fused[: args.hubness_k]:
            hubness[doc.doc_id] += 1
        for rank, (score, idx, doc) in enumerate(fused[: args.top_k], start=1):
            all_top_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.text,
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "video_id": doc.video_id,
                    "frame_id": doc.frame_id,
                    "segment_id": doc.segment_id or "",
                    "timestamp_sec": doc.timestamp_sec if doc.timestamp_sec is not None else "",
                    "image_path": doc.image_path,
                    "final_score": score,
                    "visual_score": normalized["visual"][idx],
                    "asr_score": normalized["asr"][idx],
                    "ocr_score": normalized["ocr"][idx],
                    "caption_score": normalized["caption"][idx],
                    "metadata_score": normalized["metadata"][idx],
                    "merged_score": normalized["merged"][idx],
                    "asr_text": doc.asr_text[:500],
                    "ocr_text": doc.ocr_text[:500],
                    "caption_text": doc.caption_text[:500],
                    "metadata_text": doc.metadata_text[:500],
                }
            )

    metrics = evaluate(queries, rankings_for_eval)
    metrics["labels_loaded"] = any(
        q.relevant_video_ids or q.relevant_segment_ids or q.relevant_frame_ids or q.relevant_doc_ids or q.intervals
        for q in queries
    )
    metrics["model_status"] = model_status
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "topk_examples.json", all_top_rows)
    write_csv(
        output_dir / "topk_examples.csv",
        all_top_rows,
        [
            "query_id",
            "query",
            "rank",
            "doc_id",
            "video_id",
            "frame_id",
            "segment_id",
            "timestamp_sec",
            "image_path",
            "final_score",
            "visual_score",
            "asr_score",
            "ocr_score",
            "caption_score",
            "metadata_score",
            "merged_score",
            "asr_text",
            "ocr_text",
            "caption_text",
            "metadata_text",
        ],
    )
    hub_rows = [
        {"doc_id": doc_id, "topk_frequency": count}
        for doc_id, count in sorted(hubness.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_json(output_dir / "hubness_counts.json", hub_rows)
    write_csv(output_dir / "hubness_counts.csv", hub_rows, ["doc_id", "topk_frequency"])
    write_json(output_dir / "setup_instructions.json", setup_instructions())

    print(f"Wrote baseline run to {output_dir}")
    print(f"Keyframes: {len(docs)} | Queries: {len(queries)} | Labels loaded: {metrics['labels_loaded']}")
    return 0


def setup_instructions() -> dict[str, Any]:
    return {
        "current_default": "stdlib BM25/evidence/hybrid/hubness runner; no global installs required",
        "optional_project_local_dependencies": [
            "uv add numpy pillow torch transformers",
            "uv add sentence-transformers faiss-cpu",
        ],
        "visual_models_to_try_when_available": [
            "openai/clip-vit-base-patch32",
            "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            "google/siglip2-base-patch16-384",
            "jinaai/jina-clip-v2",
        ],
        "text_dense_models_to_try_when_available": [
            "bkai-foundation-models/vietnamese-bi-encoder",
            "BAAI/bge-m3",
        ],
        "notes": [
            "Use project-local uv dependencies only.",
            "The runner uses local_files_only=True for visual models to avoid implicit downloads.",
            "Do not run cuda unless GPU use is approved by the team.",
        ],
    }


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
