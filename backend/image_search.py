"""
Visual reference image search for the MJCF generation pipeline.

Search flow:
  1. Generate 2 queries per description (schematic + CAD)
  2. Fetch up to 3 URLs per query from Bing or Google
  3. Probe each URL: download with 2-second timeout, check >= 200x200
  4. Return valid image bytes; cache by description key for reuse
"""

import asyncio
import logging
import os
from io import BytesIO
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BING_API_KEY = os.getenv("BING_SEARCH_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

MIN_DIM = 200        # pixels
PROBE_TIMEOUT = 2.0  # seconds
IMAGES_PER_QUERY = 3

# In-memory cache: normalized description key -> list of valid image bytes
_cache: dict[str, list[bytes]] = {}


def _cache_key(description: str) -> str:
    return description.strip().lower()[:60]


def make_search_queries(description: str) -> list[str]:
    return [
        f"{description} engineering schematic labeled",
        f"{description} mechanism CAD diagram",
    ]


async def _search_bing(query: str, n: int = IMAGES_PER_QUERY) -> list[str]:
    if not BING_API_KEY:
        return []
    url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {**_HEADERS, "Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": n, "safeSearch": "Moderate"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            return [img["contentUrl"] for img in data.get("value", [])[:n]]
    except Exception as e:
        logger.warning(f"[image_search] Bing search failed for '{query}': {e}")
        return []


async def _search_google(query: str, n: int = IMAGES_PER_QUERY) -> list[str]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": n,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return [item["link"] for item in data.get("items", [])[:n]]
    except Exception as e:
        logger.warning(f"[image_search] Google search failed for '{query}': {e}")
        return []


async def _search_ddg(query: str, n: int = IMAGES_PER_QUERY) -> list[str]:
    """DuckDuckGo image search — no API key required. Used when no paid key is configured."""
    import re
    headers = {**_HEADERS, "Referer": "https://duckduckgo.com/"}
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            r = await client.get("https://duckduckgo.com/", params={"q": query}, headers=_HEADERS)
            m = re.search(r'vqd=["\']([\d-]+)["\';]', r.text) or re.search(r'vqd=([\d-]+)', r.text)
            if not m:
                return []
            vqd = m.group(1)
            r2 = await client.get(
                "https://duckduckgo.com/i.js",
                params={"l": "us-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
                headers=headers,
            )
            data = r2.json()
            return [item["image"] for item in data.get("results", [])[:n]]
    except Exception as e:
        logger.warning(f"[image_search] DDG search failed for '{query}': {e}")
        return []


async def _search_images(query: str) -> list[str]:
    """Try Bing → Google → DuckDuckGo (no key required)."""
    urls = await _search_bing(query)
    if not urls:
        urls = await _search_google(query)
    if not urls:
        urls = await _search_ddg(query)
    return urls


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


async def _probe_url(url: str) -> Optional[bytes]:
    """Download URL with a 2-second timeout. Return bytes if image >= 200x200, else None."""
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()
            data = resp.content

        from PIL import Image
        img = Image.open(BytesIO(data))
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            logger.debug(f"[image_search] Skipped {url} — too small ({w}x{h})")
            return None
        return data
    except Exception as e:
        logger.debug(f"[image_search] Skipped {url}: {e}")
        return None


async def fetch_reference_images(description: str) -> list[bytes]:
    """
    Search and probe reference images for a description.
    Returns a list of valid image bytes (in memory only, never written to disk).
    Results are cached by description keyword.
    """
    key = _cache_key(description)
    if key in _cache:
        logger.info(f"[image_search] Cache hit for '{key}' ({len(_cache[key])} images)")
        return _cache[key]

    queries = make_search_queries(description)
    logger.info(f"[image_search] Searching with queries: {queries}")

    # Fetch URLs for both queries concurrently
    url_lists = await asyncio.gather(*[_search_images(q) for q in queries])
    all_urls: list[str] = [url for urls in url_lists for url in urls]
    logger.info(f"[image_search] Got {len(all_urls)} candidate URLs")

    # Probe all URLs concurrently
    probe_results = await asyncio.gather(*[_probe_url(url) for url in all_urls])
    valid = [b for b in probe_results if b is not None]

    _cache[key] = valid
    logger.info(f"[image_search] {len(valid)}/{len(all_urls)} images passed size/load checks")
    return valid
