import requests
import re
import urllib.parse
import json
import os
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PLATFORM_META = {
    "Amazon":   {"color": "#FF9900", "bg": "#FFF8E7", "domain": "amazon.in"},
    "Flipkart": {"color": "#2874F0", "bg": "#EEF4FF", "domain": "flipkart.com"},
    "Myntra":   {"color": "#FF3F6C", "bg": "#FFF0F3", "domain": "myntra.com"},
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"

PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)")


def clean_price(text: str) -> str | None:
    if not text:
        return None
    m = PRICE_RE.search(str(text))
    if m:
        raw = m.group(1).replace(",", "")
        try:
            val = int(float(raw))
            if 100 < val < 10_000_000:
                return f"₹{val:,}"
        except ValueError:
            pass
    return None


# ─── DuckDuckGo HTML search ───────────────────────────────────────────────────

def ddg_search(query: str, num: int = 10) -> list[dict]:
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": "", "kl": "in-en"},
            headers=HEADERS,
            timeout=12,
        )
        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for item in soup.select(".result")[:num]:
            title_el   = item.select_one(".result__title")
            snippet_el = item.select_one(".result__snippet")
            url_el     = item.select_one(".result__url")
            link_el    = item.select_one(".result__title a")
            title   = title_el.get_text(" ", strip=True)   if title_el   else ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            url_txt = url_el.get_text(strip=True)          if url_el     else ""
            link    = link_el.get("href", "")              if link_el    else ""
            # Decode DDG redirect
            if link and "duckduckgo.com" in link:
                try:
                    qs     = urllib.parse.urlparse(link).query
                    params = urllib.parse.parse_qs(qs)
                    link   = urllib.parse.unquote(params.get("uddg", [link])[0])
                except Exception:
                    pass
            results.append({"title": title, "snippet": snippet, "url": url_txt, "link": link})
        return results
    except Exception:
        return []


def get_platform_link(results: list[dict], domain: str) -> str | None:
    for r in results:
        if domain in r.get("url", "") or domain in r.get("link", ""):
            link = r["link"]
            if not link.startswith("http"):
                link = "https://" + r["url"]
            return link
    return None


# ─── HuggingFace model for real-time prices ───────────────────────────────────

def hf_get_prices(query: str) -> dict:
    """
    Ask the Qwen model for current prices on Amazon, Flipkart, Myntra (India).
    Returns {'amazon': '₹X,XXX', 'flipkart': ..., 'myntra': ...}
    Any unknown price is omitted.
    """
    if not HF_TOKEN:
        return {}
    try:
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a real-time Indian e-commerce price assistant. "
                        "You know current approximate prices on Amazon India, Flipkart, and Myntra. "
                        "Reply with ONLY three lines in this exact format:\n"
                        "Amazon: ₹PRICE\n"
                        "Flipkart: ₹PRICE\n"
                        "Myntra: ₹PRICE\n"
                        "If a product is not sold on a platform, write N/A. "
                        "No extra text. No explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Current price of '{query}' on Amazon India, Flipkart, Myntra?",
                },
            ],
            model=MODEL_ID,
            max_tokens=80,
            temperature=0.05,
        )
        text = resp.choices[0].message.content.strip()
        result = {}
        for line in text.splitlines():
            line_lower = line.lower()
            price = clean_price(line)
            if not price:
                continue
            if "amazon" in line_lower:
                result["amazon"] = price
            elif "flipkart" in line_lower:
                result["flipkart"] = price
            elif "myntra" in line_lower:
                result["myntra"] = price
        return result
    except Exception:
        return {}


def normalize_query(raw: str) -> str:
    """Ask HF model to produce a clean product search query."""
    if not HF_TOKEN or not raw.strip():
        return raw.strip()
    try:
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product search query cleaner. "
                        "Output ONLY a short, clean product name (max 8 words) suitable for searching on "
                        "Amazon India, Flipkart, and Myntra. No explanation, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": f"Clean this product name: {raw}"},
            ],
            model=MODEL_ID,
            max_tokens=25,
            temperature=0.05,
        )
        cleaned = resp.choices[0].message.content.strip().strip('"').strip("'")
        if cleaned and 3 < len(cleaned) < 100:
            return cleaned
        return raw.strip()
    except Exception:
        return raw.strip()


# ─── Product image via DDG ────────────────────────────────────────────────────

def get_product_image(query: str, ddg_results: list[dict]) -> str | None:
    """Try to get a product image URL from Amazon or other CDN links found in DDG."""
    # Look for amazon CDN images from DDG results
    for r in ddg_results:
        link = r.get("link", "")
        if "amazon.in" in link or "amazon.com" in link:
            try:
                resp = requests.get(link, headers=HEADERS, timeout=8)
                soup = BeautifulSoup(resp.text, "lxml")
                for sel in ["#landingImage", "#imgBlkFront", ".a-dynamic-image"]:
                    img = soup.select_one(sel)
                    if img:
                        src = img.get("src", "")
                        if src and src.startswith("http"):
                            return src
                        data = img.get("data-a-dynamic-image", "")
                        if data:
                            try:
                                d = json.loads(data)
                                return max(d.keys(), key=lambda u: d[u][0] * d[u][1])
                            except Exception:
                                pass
            except Exception:
                pass

    # Fallback: search for image in any page among top results
    for r in ddg_results[:5]:
        link = r.get("link", "")
        if not link or "duckduckgo" in link:
            continue
        try:
            resp = requests.get(link, headers=HEADERS, timeout=6)
            soup = BeautifulSoup(resp.text, "lxml")
            # Look for og:image
            og = soup.select_one("meta[property='og:image']")
            if og and og.get("content", "").startswith("http"):
                return og["content"]
        except Exception:
            pass
    return None


# ─── Main aggregator ─────────────────────────────────────────────────────────

def get_all_prices(query: str) -> tuple:
    import concurrent.futures

    # Fire DDG search and HF price call in parallel
    def _ddg():
        return ddg_search(f"{query} buy online india amazon flipkart myntra price", num=12)

    def _hf():
        return hf_get_prices(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_ddg = ex.submit(_ddg)
        f_hf  = ex.submit(_hf)
        ddg_results = f_ddg.result()
        hf_prices   = f_hf.result()

    # Fetch product image (can run while we build the result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        f_img = ex.submit(get_product_image, query, ddg_results)
        image = f_img.result()

    def build_result(platform: str) -> dict:
        meta  = PLATFORM_META[platform]
        domain = meta["domain"]
        price = hf_prices.get(platform.lower())
        link  = get_platform_link(ddg_results, domain)
        # Try to find a title from DDG results
        title = None
        for r in ddg_results:
            if domain in r.get("url", "") or domain in r.get("link", ""):
                title = r["title"]
                break
        return {
            "platform": platform,
            **meta,
            "price": price,
            "title": title,
            "link": link,
            "image": None,
        }

    amazon   = build_result("Amazon")
    flipkart = build_result("Flipkart")
    myntra   = build_result("Myntra")

    return amazon, flipkart, myntra, image
