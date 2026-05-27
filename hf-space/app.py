import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import os
import urllib.parse
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)")

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
]

DDG_HEADERS = {
    "User-Agent": UA_LIST[0],
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


def get_platform_link(results, domain: str):
    for r in results:
        if domain in r.get("url", "") or domain in r.get("link", ""):
            link = r["link"]
            if not link.startswith("http"):
                link = "https://" + r["url"]
            return link
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
            "<p style='color:#e53935;text-align:center;padding:20px'>⚠️ Please enter a product name.</p>",
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

    PLATFORMS = [
        {"platform": "Amazon.in", "domain": "amazon.in",    "color": "#FF9900", "url_base": "https://www.amazon.in",    "search": f"https://www.amazon.in/s?k={normalized.replace(' ','+')}",    "price_key": "amazon"},
        {"platform": "Flipkart",   "domain": "flipkart.com", "color": "#2874F0", "url_base": "https://www.flipkart.com", "search": f"https://www.flipkart.com/search?q={normalized.replace(' ','+')}","price_key": "flipkart"},
        {"platform": "Myntra",     "domain": "myntra.com",   "color": "#FF3F6C", "url_base": "https://www.myntra.com",   "search": f"https://www.myntra.com/{normalized.replace(' ','+')}",          "price_key": "myntra"},
    ]

    active_keys = {p.lower(): p for p in (selected_platforms or [])}
    results = []
    for p in PLATFORMS:
        if active_keys and not any(k in p["platform"].lower() for k in active_keys):
            continue
        link  = get_platform_link(ddg_results, p["domain"]) or p["search"]
        title = get_platform_title(ddg_results, p["domain"])
        price = hf_prices.get(p["price_key"])
        results.append({**p, "price": price, "title": title, "link": link})

    ai_out = hf_ai_analysis(normalized, *results[:3]) if len(results) >= 3 else "Need all 3 platforms for AI analysis."

    progress(1.0, desc="✅ Done!")

    table_html = _build_table(results, image_url, normalized)
    links_html = _build_links(normalized, results)
    return table_html, ai_out, links_html


def _build_table(results, image_url, query):
    rows = ""
    for r in results:
        price = r.get("price") or "Not Available"
        color = r["color"]
        p_color = color if r.get("price") else "#999"
        title = (r.get("title") or "")[:70]
        link  = r.get("link", "#")
        view  = f'<a href="{link}" target="_blank" class="view-btn" style="border-color:{color};color:{color}">View →</a>' if link != "#" else ""
        rows += f"""<tr>
            <td><span class="platform-badge" style="border-color:{color};color:{color}">{r['platform']}</span></td>
            <td style="color:{p_color};font-weight:700;font-size:1.1em">{price}</td>
            <td class="product-title">{title}</td>
            <td>{view}</td>
        </tr>"""

    img_html = ""
    if image_url:
        img_html = f'<div style="text-align:center;margin-bottom:20px"><img src="{image_url}" style="max-height:200px;max-width:280px;border-radius:12px;object-fit:contain;background:#fff;padding:8px" /></div>'

    has_price = any(r.get("price") for r in results)
    no_token_warn = "" if has_price else (
        "<div style='background:rgba(255,165,0,.1);border:1px solid rgba(255,165,0,.4);border-radius:10px;"
        "padding:12px 16px;margin-bottom:14px;color:#FFD700;font-size:13px'>"
        "⚠️ No prices found — make sure <b>HF_TOKEN</b> is set in Space Secrets "
        "(Settings → Variables and secrets → New secret → <code>HF_TOKEN</code>)</div>"
    )

    return f"""
{no_token_warn}
{img_html}
<div class="results-container">
    <h3 class="results-title">📦 Price Comparison — <em style="color:#aaa;font-weight:400">{query}</em></h3>
    <table class="price-table">
        <thead><tr><th>Platform</th><th>Price</th><th>Product Found</th><th>Link</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""


def _build_links(query, results):
    q = query.replace(" ", "+")
    chips = "".join(
        f'<a href="{r["search"]}" target="_blank" class="link-chip">{r["platform"]}</a>'
        for r in results
    )
    chips += f'<a href="https://www.google.com/search?q={q}&tbm=shop" target="_blank" class="link-chip">🌐 Google Shopping</a>'
    return f'<div class="links-wrapper"><p style="color:#aaa;margin-bottom:8px">🔗 Open directly:</p>{chips}</div>'


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg: #0a0a0f; --surface: #13131a; --surface2: #1c1c28;
    --accent: #ff6b35; --accent2: #00e5ff; --gold: #ffd700;
    --text: #f0f0f5; --muted: #888; --border: rgba(255,255,255,0.08); --radius: 16px;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.app-header {
    text-align: center; padding: 40px 20px 20px;
    background: linear-gradient(135deg, #0d0d18 0%, #1a0a1e 50%, #0d1320 100%);
    border-bottom: 1px solid var(--border);
}
.app-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800; letter-spacing: -1px; margin: 0;
    background: linear-gradient(135deg, #ff6b35 0%, #ffd700 40%, #00e5ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-subtitle { font-size: 1rem; color: var(--muted); margin-top: 8px; }
.app-badges { display: flex; gap: 8px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
.badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 4px 12px; font-size: .75rem; color: var(--muted); }
label, .label-wrap { color: var(--text) !important; font-size: .9rem !important; }
textarea, input[type=text] {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,.15) !important;
}
.compare-btn {
    background: linear-gradient(135deg,#ff6b35,#ff4d1a) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important; font-size: 1.05rem !important;
    font-weight: 700 !important; padding: 14px 28px !important;
    cursor: pointer !important; width: 100% !important; text-transform: uppercase !important;
}
.results-container { padding: 8px 0; }
.results-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; color: var(--accent2); margin-bottom: 16px; }
.price-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.price-table thead th {
    background: var(--surface2); color: var(--muted); padding: 10px 14px;
    text-align: left; font-weight: 500; font-size: .8rem; text-transform: uppercase;
    letter-spacing: .5px; border-bottom: 1px solid var(--border);
}
.price-table tbody tr { border-bottom: 1px solid var(--border); transition: background .15s; }
.price-table tbody tr:hover { background: var(--surface2); }
.price-table td { padding: 12px 14px; color: var(--text); vertical-align: middle; }
.platform-badge { border: 1px solid; border-radius: 8px; padding: 4px 10px; font-size: .85rem; white-space: nowrap; }
.product-title { color: var(--muted); font-size: .82rem; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.view-btn { text-decoration: none !important; border: 1px solid; border-radius: 6px; padding: 4px 10px; font-size: .85rem; white-space: nowrap; transition: all .15s; }
.view-btn:hover { opacity: .75; }
.links-wrapper { padding: 16px 0 8px; }
.link-chip {
    display: inline-block; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: 6px 14px; color: var(--text); text-decoration: none;
    font-size: .82rem; margin: 4px; transition: all .15s;
}
.link-chip:hover { border-color: var(--accent); color: var(--accent); }
.app-footer { text-align: center; padding: 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

HEADER_HTML = """
<div class="app-header">
    <h1 class="app-title">Come &amp; Compare</h1>
    <p class="app-subtitle">AI-powered real-time price comparison across Indian e-commerce platforms</p>
    <div class="app-badges">
        <span class="badge">🤖 Qwen2.5-7B</span>
        <span class="badge">⚡ ≤32B Parameters</span>
        <span class="badge">🇮🇳 Amazon · Flipkart · Myntra</span>
        <span class="badge">🏆 HF Small Models Hackathon</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
    Built for the HuggingFace Build Small Hackathon 2025 &nbsp;|&nbsp;
    Model: Qwen/Qwen2.5-7B-Instruct (&lt;32B) &nbsp;|&nbsp;
    Search: DuckDuckGo HTML (no API key needed)
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
            compare_btn = gr.Button("⚡ Compare Prices Now", elem_classes=["compare-btn"])
            gr.Markdown("""---\n**💡 Tips:** Be specific — include brand + model. Add size/color for clothing.""")

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
            ["iPhone 15 128GB",           "Apple, Black"],
            ["Nike Air Force 1",           "White, Size 9 UK"],
            ["Samsung 55 inch 4K TV",      "Smart TV"],
            ["boAt Airdopes 141",          ""],
            ["OnePlus Nord CE 4",          "8GB RAM 128GB"],
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
