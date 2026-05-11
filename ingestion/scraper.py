"""
WebMD Scraper — A-Z Health Topics (Improved)
Run: python -m ingestion.scraper
"""
import os
import time
import re
import hashlib
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

from config import RAW_DATA_DIR, MAX_PAGES, WEBMD_BASE_URL
from utils.logger import get_logger
from utils.helpers import ensure_dirs, write_txt, read_txt, save_json, load_json

log = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MedRAGBot/1.0)"
}

A_Z_URL = WEBMD_BASE_URL + "/a-to-z-guides/health-topics"
REQUEST_TIMEOUT = 15
MANIFEST_PATH = os.path.join(RAW_DATA_DIR, "manifest.json")

ALLOWED_PATH_HINTS = (
    "/default.htm",
    "/default.asp",
    "/article",
    "/conditions/",
    "/drugs/",
    "/symptoms/",
    "/diseases/",
    "/health/",
    "/heart-disease/",
    "/mental-health/",
    "/pain-management/",
    "/skin-problems-and-treatments/",
    "/cancer/",
    "/add-adhd/",
    "/allergies/",
    "/diabetes/",
    "/arthritis/",
    "/breast-cancer/",
    "/hiv-aids/",
    "/hypertension-high-blood-pressure/",
    "/lung/",
    "/migraines-headaches/",
)

DISEASE_BLOCKED_HINTS = (
    "/drugs/",
    "/pill-identification/",
    "/news",
    "/videos",
    "/video/",
    "/quiz",
    "/slides",
    "/photo/",
)

BLOCKED_PATH_HINTS = (
    "/search/",
    "/video/",
    "/quiz",
    "/slides",
    "/photo/",
    "/tool/",
    "/story/",
    "/sponsored/",
    "/static/",
    "/share.aspx",
    "/my-library",
    "/api/",
    "/mm/",
    "/kapi/",
)



# ── Extract clean structured content ─────────────────────────────────────────
def _extract_content(soup):
    content = []

    main_nodes = soup.select("main, article, .article-body, .content, .content-body")
    scope = main_nodes if main_nodes else [soup]

    for node in scope:
        for tag in node.find_all(["h1", "h2", "h3", "p", "li"]):
            text = tag.get_text(" ", strip=True)

            if not text or len(text) < 40:
                continue

            content.append(text)

    # remove duplicates
    content = list(dict.fromkeys(content))

    full_text = "\n".join(content)

    # normalize excessive newlines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


def _normalize_url(url, base_url=A_Z_URL):
    absolute = urljoin(base_url, url)
    absolute, _ = urldefrag(absolute)
    parsed = urlparse(absolute)
    return parsed._replace(query="").geturl()


def _is_webmd_content_url(url):
    parsed = urlparse(url)

    if parsed.netloc not in {"www.webmd.com", "webmd.com"}:
        return False

    path = parsed.path.lower()

    if any(blocked in path for blocked in BLOCKED_PATH_HINTS):
        return False

    return any(hint in path for hint in ALLOWED_PATH_HINTS)


def _extract_links(soup, base_url):
    links = []

    for anchor in soup.find_all("a", href=True):
        url = _normalize_url(anchor["href"], base_url)

        if _is_webmd_content_url(url):
            links.append(url)

    return list(dict.fromkeys(links))


def _is_disease_topic_url(url):
    parsed = urlparse(url)
    path = parsed.path.lower()

    if any(blocked in path for blocked in DISEASE_BLOCKED_HINTS):
        return False

    # Keep condition/reference pages while excluding obvious non-disease sections.
    return any(hint in path for hint in (
        "/a-to-z-guides/",
        "/conditions/",
        "/diseases/",
        "/health/",
        "/heart-disease/",
        "/mental-health/",
        "/pain-management/",
        "/skin-problems-and-treatments/",
        "/cancer/",
        "/diabetes/",
        "/arthritis/",
        "/hiv-aids/",
        "/hypertension-high-blood-pressure/",
        "/lung/",
        "/migraines-headaches/",
    ))


def _content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_manifest_records():
    data = load_json(MANIFEST_PATH)
    if not data:
        return []
    records = data.get("records", [])
    return records if isinstance(records, list) else []


def _save_manifest_records(records):
    save_json({"records": records}, MANIFEST_PATH)


def _existing_page_files():
    return [f for f in os.listdir(RAW_DATA_DIR) if re.fullmatch(r"page_\d+\.txt", f)]


def _next_page_index(existing_files):
    if not existing_files:
        return 0
    return max(int(re.search(r"page_(\d+)\.txt", f).group(1)) for f in existing_files) + 1


# ── Step 1: Collect links ─────────────────────────────────────────────────────
def get_topic_links():
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    all_links = []

    try:
        # fetch the main A-Z page first
        r = requests.get(A_Z_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            all_links.extend(_extract_links(soup, A_Z_URL))

        # then fetch per-letter pages (pg=a..z)
        for pg in letters:
            try:
                page_url = f"{A_Z_URL}?pg={pg}"
                r = requests.get(page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

                if r.status_code != 200:
                    log.debug(f"Letter page {pg} returned {r.status_code}")
                    continue

                soup = BeautifulSoup(r.text, "html.parser")
                links = _extract_links(soup, page_url)
                all_links.extend(links)
                time.sleep(0.2)

            except Exception:
                log.debug(f"Failed to fetch letter page: {pg}")

        # deduplicate while preserving order and keep disease/topic endpoints only
        seen = set()
        dedup = []
        for l in all_links:
            if l not in seen and _is_disease_topic_url(l):
                seen.add(l)
                dedup.append(l)

        log.info(f"Collected {len(dedup)} seed links from A-Z and letter pages")
        return dedup

    except Exception as e:
        log.error(f"A-Z topic fetch failed: {e}")
        return []
    
# ── Step 2: Scrape pages ─────────────────────────────────────────────────────
def scrape(max_pages: int | None = MAX_PAGES, reset: bool = False):
    ensure_dirs(RAW_DATA_DIR)

    if reset:
        for fname in _existing_page_files():
            os.remove(os.path.join(RAW_DATA_DIR, fname))
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
        log.info("Reset enabled: removed previous scraped pages and manifest")

    log.info("Collecting A-Z topic links...")
    urls = get_topic_links()

    if not urls:
        log.error("No URLs found. Scraping aborted.")
        return

    if max_pages is not None:
        urls = urls[:max_pages]

    existing_files = _existing_page_files()
    next_index = _next_page_index(existing_files)

    records = _load_manifest_records()
    known_urls = {r.get("url") for r in records if r.get("url")}
    known_hashes = {r.get("sha256") for r in records if r.get("sha256")}

    saved = len(existing_files)
    fetched = 0
    skipped_url = 0
    skipped_hash = 0
    failed = 0

    log.info(f"Processing {len(urls)} A-Z disease/topic URLs with dedup enabled")

    for url in urls:
        if url in known_urls:
            skipped_url += 1
            continue

        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if r.status_code != 200:
                continue

            content_type = r.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # remove junk tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                tag.decompose()

            text = _extract_content(soup)

            # quality filter + content dedup
            if not text or len(text) <= 300:
                continue

            text_hash = _content_hash(text)
            if text_hash in known_hashes:
                skipped_hash += 1
                continue

            fname = f"page_{next_index:05d}.txt"
            write_txt(os.path.join(RAW_DATA_DIR, fname), text)
            records.append({"url": url, "file": fname, "sha256": text_hash})

            known_urls.add(url)
            known_hashes.add(text_hash)
            saved += 1
            next_index += 1

            if saved % 25 == 0:
                _save_manifest_records(records)

            fetched += 1

            if fetched % 50 == 0:
                log.info(
                    f"Progress: fetched {fetched}, saved {saved}, skipped_url {skipped_url}, skipped_hash {skipped_hash}"
                )

            time.sleep(0.25)

        except Exception as e:
            log.warning(f"Failed {url}: {e}")
            failed += 1

    _save_manifest_records(records)
    log.info(
        f"Done: saved {saved} unique pages, skipped_url {skipped_url}, skipped_hash {skipped_hash}, failed {failed}"
    )
    log.info(f"Data saved → {RAW_DATA_DIR}")
    


if __name__ == "__main__":
    scrape()