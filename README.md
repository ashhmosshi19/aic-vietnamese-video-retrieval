# Vietnamese News-Video Retrieval Baseline

A dependency-light baseline for Vietnamese text-query to keyframe retrieval. It scores textual evidence attached to video frames and evaluates video-, segment-, frame-, and temporal-IoU relevance when labels are available.

## Implemented baseline

- BM25 over ASR, OCR, captions, metadata, and merged evidence;
- weighted score fusion;
- hubness analysis for repeated top-k frames;
- video, segment, exact-frame, and temporal-IoU metrics;
- optional visual-model hooks that are explicit rather than silently downloading models.

The repository contains only the sanitized runner. Video, keyframe, ASR, OCR, captions, and labels must be supplied from an authorized local dataset.

## Smoke test

```bash
python run_baseline.py --help
python run_baseline.py \
  --keyframes-root ./data/keyframes_1fps \
  --asr-json ./data/asr.json \
  --ocr-json ./data/ocr.json \
  --collection-limit 1 \
  --video-limit 1 \
  --frames-per-video 20 \
  --query "người dẫn chương trình thời sự"
```

Run outputs are written under `runs/`, which is ignored by Git.
