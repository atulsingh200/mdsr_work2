"""
Scrape all Adobe Experience Platform tutorial video transcripts from
https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/overview

Strategy:
  1. Download TOC.md from GitHub (authoritative list of all tutorial pages)
  2. Convert each /help/platform/... path → EL page URL
  3. Fetch each EL page HTML, extract adobe.tv video ID
  4. Fetch https://video.tv.adobe.com/vc/<ID>/eng.json → captions array
  5. Join caption content into a single transcript string
  6. Output: transcripts.json  (list of dicts with url, title, video_id, transcript)
"""

import re
import time
import json
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TOC_URL = "https://raw.githubusercontent.com/AdobeDocs/platform-learn.en/main/help/platform/TOC.md"
EL_BASE  = "https://experienceleague.adobe.com"
OUT_FILE = Path(__file__).parent / "aep_transcripts.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Regex to find adobe.tv video IDs in page HTML
# Matches: video.tv.adobe.com/v/12345  or  video.tv.adobe.com/vc/12345
VIDEO_ID_RE = re.compile(r'video\.tv\.adobe\.com/v[ci]/(\d+)', re.IGNORECASE)
# Also matches data-video-src or src attributes
VIDEO_SRC_RE = re.compile(r'"(?:https?://)?video\.tv\.adobe\.com/v[ci]/(\d+)', re.IGNORECASE)


def fetch(url: str, retries: int = 3, delay: float = 2.0) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None  # page doesn't exist, skip silently
            log.warning(f"HTTP {r.status_code} for {url}")
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def toc_path_to_el_url(toc_path: str) -> str:
    """
    /help/platform/foo/bar.md  →  https://experienceleague.adobe.com/en/docs/platform-learn/tutorials/foo/bar
    """
    path = toc_path.strip()
    path = re.sub(r'\.md$', '', path)          # remove .md
    path = re.sub(r'^/help/platform', '', path) # strip /help/platform prefix
    return f"{EL_BASE}/en/docs/platform-learn/tutorials{path}"


def extract_tutorial_paths(toc_md: str) -> list[dict]:
    """
    Parse TOC.md and extract all internal /help/platform/... links with their titles.
    Skips external URLs (experienceleague.adobe.com/en/docs/... absolute links and target="_blank").
    """
    entries = []
    # Match: [Title](/help/platform/foo/bar.md)  (internal links only)
    for m in re.finditer(r'\[([^\]]+)\]\((/help/platform/[^\)\s]+\.md)\)', toc_md):
        title = m.group(1).strip()
        path  = m.group(2).strip()
        url   = toc_path_to_el_url(path)
        entries.append({"title": title, "toc_path": path, "url": url})
    return entries


def extract_video_id(html: str) -> str | None:
    """Extract the first adobe.tv video ID from page HTML."""
    for pattern in (VIDEO_ID_RE, VIDEO_SRC_RE):
        m = pattern.search(html)
        if m:
            return m.group(1)
    return None


def fetch_transcript(video_id: str) -> str | None:
    """Fetch caption JSON and join into a single transcript string."""
    caption_url = f"https://video.tv.adobe.com/vc/{video_id}/eng.json"
    r = fetch(caption_url)
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    captions = data.get("captions", [])
    if not captions:
        return None
    # Sort by startTime to ensure correct ordering
    captions.sort(key=lambda c: c.get("startTime", 0))
    return " ".join(c["content"].strip() for c in captions if c.get("content", "").strip())


def scrape_page(entry: dict) -> dict | None:
    """Fetch one tutorial page, extract video ID and transcript."""
    url = entry["url"]
    r = fetch(url)
    if r is None:
        log.info(f"  SKIP (no response): {url}")
        return None

    video_id = extract_video_id(r.text)
    if not video_id:
        log.info(f"  SKIP (no video): {url}")
        return None

    transcript = fetch_transcript(video_id)
    if not transcript:
        log.info(f"  SKIP (no transcript): {url} (video {video_id})")
        return None

    log.info(f"  OK  video={video_id}  words={len(transcript.split())}  {url}")
    return {
        "title":      entry["title"],
        "url":        url,
        "toc_path":   entry["toc_path"],
        "video_id":   video_id,
        "transcript": transcript,
    }


def main():
    # ── Step 1: get TOC ──────────────────────────────────────────────────
    log.info("Fetching TOC from GitHub...")
    r = fetch(TOC_URL)
    if r is None:
        raise RuntimeError("Could not fetch TOC.md")
    toc_md = r.text

    # ── Step 2: parse all tutorial paths ────────────────────────────────
    entries = extract_tutorial_paths(toc_md)
    log.info(f"Found {len(entries)} tutorial pages in TOC")

    # ── Step 3: scrape each page ─────────────────────────────────────────
    results = []
    for i, entry in enumerate(entries, 1):
        log.info(f"[{i}/{len(entries)}] {entry['title']}")
        record = scrape_page(entry)
        if record:
            results.append(record)
        time.sleep(0.4)  # polite crawl delay

    # ── Step 4: save ─────────────────────────────────────────────────────
    OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info(f"\nDone. {len(results)}/{len(entries)} transcripts saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
