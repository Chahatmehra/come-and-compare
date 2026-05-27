import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import os
import urllib.parse
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)")
ASIN_RE  = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")

DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_client():
    token = os.environ.get("HF_TOKEN", "")
    return InferenceClient(token=token) if token else InferenceClient()


def clean_price(text: str):
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


def clean_amazon_link(raw_link: str) -> str:
    """Extract ASIN and return a clean, working Amazon.in product URL."""
    if not raw_link:
        return None
    m = ASIN_RE.search(raw_link)
    if m:
        return f"https://www.amazon.in/dp/{m.group(1)}"
    # If no ASIN, keep only the base path (strip all query params/tracking)
    try:
        parsed = urllib.parse.urlparse(raw_link)
        if "amazon" in parsed.netloc:
            clean = parsed._replace(query="", fragment="").geturl()
            return clean
    except Exception:
        pass
    return raw_link


def clean_flipkart_link(raw_link: str) -> str:
    """Keep only essential Flipkart URL params, strip tracking."""
    if not raw_link:
        return None
    try:
        parsed = urllib.parse.urlparse(raw_link)
        qs = urllib.parse.parse_qs(parsed.query)
        kept = {}
        for k in ("pid", "lid", "marketplace"):
            if k in qs:
                kept[k] = qs[k]
        new_q = urllib.parse.urlencode(kept, doseq=True)
        return parsed._replace(query=new_q, fragment="").geturl()
    except Exception:
        return raw_link


def ddg_search(query: str, num: int = 12):
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": "", "kl": "in-en"},
            headers=DDG_HEADERS,
            timeout=15,
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


def normalize_query(raw: str) -> str:
    try:
        client = get_client()
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
    except Exception as e:
        print(f"[normalize_query] {e}")
    return raw.strip()


def hf_get_prices(query: str) -> dict:
    try:
        client = get_client()
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
            price = clean_price(line)
            if not price:
                continue
            ll = line.lower()
            if "amazon" in ll:
                result["amazon"] = price
            elif "flipkart" in ll:
                result["flipkart"] = price
            elif "myntra" in ll:
                result["myntra"] = price
        return result
    except Exception as e:
        print(f"[hf_get_prices] {e}")
        return {}


def hf_ai_analysis(query: str, amazon: dict, flipkart: dict, myntra: dict) -> str:
    lines = []
    for r in [amazon, flipkart, myntra]:
        p = r.get("price") or "N/A"
        lines.append(f"- {r['platform']}: {p}")
    scraped_str = "\n".join(lines)
    try:
        client = get_client()
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart Indian price comparison assistant called 'Come & Compare'.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Product: {query}\n\nPrices:\n{scraped_str}\n\n"
                        "Reply in this exact format:\n"
                        "🏆 BEST DEAL: [platform] at [price]\n\n"
                        "📊 PRICE RANKING:\n1. [platform] — [price]\n2. ...\n\n"
                        "💡 BUYING ADVICE:\n[2-3 line recommendation]\n\n"
                        "⚠️ NOTES:\n[any warnings about unavailable prices]"
                    ),
                },
            ],
            model=MODEL_ID,
            max_tokens=350,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI analysis unavailable: {str(e)}"


def get_platform_link(results, domain: str, platform: str):
    """Return a clean, working link for the platform."""
    for r in results:
        url  = r.get("url", "")
        link = r.get("link", "")
        if domain in url or domain in link:
            raw = link if link.startswith("http") else ("https://" + url if url else None)
            if not raw:
                continue
            if platform == "Amazon.in":
                cleaned = clean_amazon_link(raw)
                if cleaned:
                    return cleaned
            elif platform == "Flipkart":
                cleaned = clean_flipkart_link(raw)
                if cleaned:
                    return cleaned
            else:
                return raw
    return None


def get_platform_title(results, domain: str):
    for r in results:
        if domain in r.get("url", "") or domain in r.get("link", ""):
            return r.get("title", "")
    return ""


def get_product_image(query: str, ddg_results: list):
    import json
    for r in ddg_results:
        link = r.get("link", "")
        if "amazon.in" in link or "amazon.com" in link:
            try:
                resp = requests.get(link, headers=DDG_HEADERS, timeout=8)
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
    for r in ddg_results[:5]:
        link = r.get("link", "")
        if not link or "duckduckgo" in link:
            continue
        try:
            resp = requests.get(link, headers=DDG_HEADERS, timeout=6)
            soup = BeautifulSoup(resp.text, "lxml")
            og = soup.select_one("meta[property='og:image']")
            if og and og.get("content", "").startswith("http"):
                return og["content"]
        except Exception:
            pass
    return None


def compare_prices(product_name, product_details, selected_platforms, progress=gr.Progress()):
    if not product_name or not product_name.strip():
        return (
            "<p style='color:#c62828;text-align:center;padding:20px;font-size:15px'>⚠️ Please enter a product name.</p>",
            "❌ No product entered.",
            "",
        )

    query = product_name.strip()
    if product_details and product_details.strip():
        query = f"{query} {product_details.strip()}"

    progress(0.05, desc="🤖 Normalizing query with Qwen 7B...")
    normalized = normalize_query(query)

    progress(0.2, desc="🔍 Searching DuckDuckGo for product links...")
    ddg_results = ddg_search(f"{normalized} buy online india amazon flipkart myntra price", num=12)

    progress(0.5, desc="💰 Fetching prices via Qwen 7B...")
    hf_prices = hf_get_prices(normalized)

    progress(0.7, desc="🖼️ Finding product image...")
    image_url = get_product_image(normalized, ddg_results)

    progress(0.85, desc="🤖 Running AI analysis...")

    enc = urllib.parse.quote_plus(normalized)
    PLATFORMS = [
        {"platform": "Amazon.in", "domain": "amazon.in",    "color": "#FF9900", "bg": "#FFF8EE",
         "search": f"https://www.amazon.in/s?k={enc}", "price_key": "amazon"},
        {"platform": "Flipkart",   "domain": "flipkart.com", "color": "#2874F0", "bg": "#EEF4FF",
         "search": f"https://www.flipkart.com/search?q={enc}", "price_key": "flipkart"},
        {"platform": "Myntra",     "domain": "myntra.com",   "color": "#FF3F6C", "bg": "#FFF0F4",
         "search": f"https://www.myntra.com/{enc}", "price_key": "myntra"},
    ]

    active_keys = {p.lower(): p for p in (selected_platforms or [])}
    results = []
    for p in PLATFORMS:
        if active_keys and not any(k in p["platform"].lower() for k in active_keys):
            continue
        link  = get_platform_link(ddg_results, p["domain"], p["platform"]) or p["search"]
        title = get_platform_title(ddg_results, p["domain"])
        price = hf_prices.get(p["price_key"])
        results.append({**p, "price": price, "title": title, "link": link})

    ai_out = hf_ai_analysis(normalized, *results[:3]) if len(results) >= 3 else "Need all 3 platforms for AI analysis."

    progress(1.0, desc="✅ Done!")

    table_html = _build_cards(results, image_url, normalized)
    links_html = _build_links(normalized, results)
    return table_html, ai_out, links_html


def _find_best(results):
    found = [r for r in results if r.get("price")]
    if not found:
        return ""
    def val(r):
        return int(r["price"].replace("₹","").replace(",","").strip())
    try:
        return min(found, key=val)["platform"]
    except Exception:
        return ""


def _build_cards(results, image_url, query):
    best = _find_best(results)

    img_html = ""
    if image_url:
        img_html = (
            f'<div style="text-align:center;margin-bottom:24px">'
            f'<img src="{image_url}" style="max-height:220px;max-width:300px;'
            f'border-radius:16px;object-fit:contain;background:#fff;'
            f'padding:12px;box-shadow:0 4px 20px rgba(0,0,0,.10)" /></div>'
        )

    cards = ""
    for r in results:
        color  = r["color"]
        bg     = r["bg"]
        price  = r.get("price")
        title  = (r.get("title") or "")[:72]
        link   = r.get("link", r["search"])
        is_best = best and r["platform"] == best and price

        border = f"3px solid {color}" if is_best else f"2px solid {color}33"
        shadow = f"0 8px 28px {color}30" if is_best else "0 4px 16px rgba(0,0,0,.08)"
        trophy = '<div style="position:absolute;top:-12px;left:50%;transform:translateX(-50%);background:#FFD700;color:#333;border-radius:20px;padding:3px 14px;font-size:11px;font-weight:700;white-space:nowrap">🏆 BEST DEAL</div>' if is_best else ""

        price_html = (
            f'<div style="font-size:2rem;font-weight:800;color:{color};margin:10px 0 6px;letter-spacing:-0.5px">{price}</div>'
            if price else
            '<div style="font-size:1rem;color:#aaa;font-weight:500;margin:10px 0 6px">Not Available</div>'
        )
        title_html = f'<div style="font-size:11px;color:#666;margin-bottom:12px;line-height:1.4;min-height:28px">{title}</div>' if title else '<div style="min-height:28px"></div>'
        btn_html = (
            f'<a href="{link}" target="_blank" style="display:inline-block;background:{color};color:#fff;'
            f'text-decoration:none;border-radius:50px;padding:8px 20px;font-size:13px;font-weight:600;'
            f'margin-top:4px">View on {r["platform"]} →</a>'
        ) if price else ""

        cards += f'''
        <div style="position:relative;background:{bg};border:{border};border-radius:20px;
            padding:24px 18px 20px;text-align:center;flex:1;min-width:180px;max-width:240px;
            box-shadow:{shadow};transition:transform .2s">
            {trophy}
            <div style="font-size:28px;margin-bottom:6px">{"🛒" if "Amazon" in r["platform"] else "🛍️" if "Flipkart" in r["platform"] else "👗"}</div>
            <div style="font-size:16px;font-weight:700;color:{color}">{r["platform"]}</div>
            {price_html}
            {title_html}
            {btn_html}
        </div>'''

    cards_row = f'<div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin:8px 0">{cards}</div>'

    has_price = any(r.get("price") for r in results)
    no_token_warn = "" if has_price else (
        '<div style="background:#FFF3CD;border:1px solid #FFC107;border-radius:12px;'
        'padding:14px 18px;margin-bottom:18px;color:#856404;font-size:13px;text-align:center">'
        '⚠️ No prices found — make sure <b>HF_TOKEN</b> is set in Space Secrets '
        '(Settings → Variables and secrets)</div>'
    )

    heading = (
        f'<div style="text-align:center;margin-bottom:18px">'
        f'<span style="background:#E3F2FD;color:#1565C0;border-radius:20px;'
        f'padding:6px 18px;font-size:13px;font-weight:600">📦 Results for: {query}</span></div>'
    )

    return f"{no_token_warn}{heading}{img_html}{cards_row}"


def _build_links(query, results):
    q = urllib.parse.quote_plus(query)
    chips = "".join(
        f'<a href="{r["search"]}" target="_blank" style="display:inline-block;'
        f'background:#fff;border:1.5px solid {r["color"]};color:{r["color"]};'
        f'border-radius:20px;padding:6px 16px;font-size:13px;font-weight:600;'
        f'text-decoration:none;margin:4px">{r["platform"]}</a>'
        for r in results
    )
    chips += (
        f'<a href="https://www.google.com/search?q={q}&tbm=shop" target="_blank" '
        f'style="display:inline-block;background:#fff;border:1.5px solid #34A853;color:#34A853;'
        f'border-radius:20px;padding:6px 16px;font-size:13px;font-weight:600;'
        f'text-decoration:none;margin:4px">🌐 Google Shopping</a>'
    )
    return f'<div style="padding:14px 0 6px"><p style="color:#555;margin-bottom:10px;font-size:13px">🔗 Search directly on each platform:</p>{chips}</div>'


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #E0F7FA 0%, #E8F5E9 40%, #E3F2FD 100%) !important;
    min-height: 100vh;
}

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 36px 24px 20px;
    background: linear-gradient(135deg, #ffffff 0%, #F0FFFE 100%);
    border-radius: 0 0 28px 28px;
    box-shadow: 0 4px 24px rgba(0,150,136,.12);
    margin-bottom: 20px;
}

.app-title {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    background: linear-gradient(90deg, #FF9900 0%, #00ACC1 50%, #43A047 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.app-subtitle { font-size: .95rem; color: #546E7A; margin-top: 8px; font-weight: 500; }

.app-badges {
    display: flex; gap: 8px; justify-content: center; margin-top: 14px; flex-wrap: wrap;
}
.badge {
    background: linear-gradient(135deg, #E0F7FA, #E8F5E9);
    border: 1px solid #B2DFDB;
    border-radius: 20px; padding: 5px 14px;
    font-size: .75rem; color: #00695C; font-weight: 600;
}

/* Input panel */
label, .label-wrap { color: #263238 !important; font-weight: 600 !important; font-size: .9rem !important; }

textarea, input[type=text] {
    background: #ffffff !important;
    border: 2px solid #B2DFDB !important;
    color: #263238 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    box-shadow: 0 2px 8px rgba(0,150,136,.06) !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #00ACC1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,172,193,.15) !important;
}

/* Compare button */
.compare-btn {
    background: linear-gradient(135deg, #00ACC1, #00897B) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(0,172,193,.35) !important;
    letter-spacing: .3px !important;
}
.compare-btn:hover { filter: brightness(1.08) !important; }

/* Tabs */
.tab-nav button { color: #546E7A !important; font-weight: 600 !important; }
.tab-nav button.selected { color: #00ACC1 !important; border-bottom-color: #00ACC1 !important; }

/* AI output box */
textarea[readonly] {
    background: #F1FFFE !important;
    border: 2px solid #B2EBF2 !important;
    color: #263238 !important;
    line-height: 1.7 !important;
}

/* Checkbox */
.wrap-inner {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 2px solid #B2DFDB !important;
}

/* Footer */
.app-footer {
    text-align: center; padding: 20px; color: #78909C;
    font-size: .8rem; margin-top: 10px;
    border-top: 1px solid #B2DFDB;
}

footer { display: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #E0F7FA; }
::-webkit-scrollbar-thumb { background: #80CBC4; border-radius: 3px; }
"""

HEADER_HTML = """
<div class="app-header">
    <h1 class="app-title">Come &amp; Compare 🛒</h1>
    <p class="app-subtitle">AI-powered real-time price comparison across India's top e-commerce platforms</p>
    <div class="app-badges">
        <span class="badge">🤖 Qwen2.5-7B</span>
        <span class="badge">⚡ Under 32B Parameters</span>
        <span class="badge">🇮🇳 Amazon · Flipkart · Myntra</span>
        <span class="badge">🏆 HF Small Models Hackathon</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
    Built for the HuggingFace Build Small Hackathon 2025 &nbsp;·&nbsp;
    Model: Qwen/Qwen2.5-7B-Instruct (&lt;32B) &nbsp;·&nbsp;
    Search: DuckDuckGo HTML
</div>
"""

with gr.Blocks(css=CSS, title="Come & Compare — Price Comparison AI") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Search Product")
            product_name = gr.Textbox(
                label="Product Name",
                placeholder='e.g. "Nike Air Force 1 White" or "Samsung Galaxy S24 128GB"',
                lines=1,
            )
            product_details = gr.Textbox(
                label="Additional Details (optional)",
                placeholder="e.g. size, color, model number...",
                lines=2,
            )
            platform_select = gr.CheckboxGroup(
                choices=["Amazon.in", "Flipkart", "Myntra"],
                value=["Amazon.in", "Flipkart", "Myntra"],
                label="Platforms to Search",
            )
            compare_btn = gr.Button("🔍 Compare Prices Now", elem_classes=["compare-btn"])
            gr.Markdown("**💡 Tip:** Include brand + model for best results.")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📊 Results"):
                    results_html = gr.HTML()
                    links_html   = gr.HTML()
                with gr.TabItem("🤖 AI Analysis"):
                    ai_output = gr.Textbox(
                        label="AI Recommendation (Qwen2.5-7B)",
                        lines=15,
                        interactive=False,
                    )

    gr.HTML(FOOTER_HTML)

    gr.Examples(
        examples=[
            ["iPhone 15 128GB",      "Apple, Black"],
            ["Nike Air Force 1",     "White, Size 9 UK"],
            ["Samsung 55 inch 4K TV","Smart TV"],
            ["boAt Airdopes 141",    ""],
            ["OnePlus Nord CE 4",    "8GB RAM 128GB"],
        ],
        inputs=[product_name, product_details],
        label="🌟 Try these examples",
    )

    compare_btn.click(
        fn=compare_prices,
        inputs=[product_name, product_details, platform_select],
        outputs=[results_html, ai_output, links_html],
    )

if __name__ == "__main__":
    demo.launch(share=False)
