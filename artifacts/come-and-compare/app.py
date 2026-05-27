import gradio as gr
import requests
import os
from io import BytesIO
from PIL import Image
from scraper import get_all_prices, normalize_query

PLATFORM_COLORS = {
    "Amazon":   "#FF9900",
    "Flipkart": "#2874F0",
    "Myntra":   "#FF3F6C",
}

PLATFORM_ICONS = {
    "Amazon":   "🛒",
    "Flipkart": "🛍️",
    "Myntra":   "👗",
}


def load_image_from_url(url: str):
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None


def build_price_card(data: dict) -> str:
    platform = data.get("platform", "")
    price    = data.get("price")
    title    = data.get("title", "") or ""
    link     = data.get("link") or "#"
    color    = PLATFORM_COLORS.get(platform, "#444")
    icon     = PLATFORM_ICONS.get(platform, "🏷️")

    if price:
        price_html = f'<div class="price-value" style="color:{color}">{price}</div>'
        title_html = (
            f'<div class="product-title">{(title[:65] + "…") if len(title) > 65 else title}</div>'
            if title else ""
        )
        link_html = (
            f'<a href="{link}" target="_blank" class="view-btn" style="background:{color}">View on {platform}</a>'
            if link != "#" else ""
        )
        card_cls = "price-card found"
    else:
        price_html = '<div class="price-na">Not Available</div>'
        title_html = ""
        link_html  = ""
        card_cls   = "price-card not-found"

    return f"""
<div class="{card_cls}" style="border-top:4px solid {color};">
  <div class="platform-header">
    <span class="platform-icon">{icon}</span>
    <span class="platform-name" style="color:{color}">{platform}</span>
  </div>
  {title_html}
  {price_html}
  {link_html}
</div>"""


def find_best_deal(results: list) -> str:
    found = [r for r in results if r.get("price")]
    if not found:
        return ""
    def price_val(r):
        return int(r["price"].replace("₹","").replace(",","").strip())
    try:
        return min(found, key=price_val)["platform"]
    except Exception:
        return ""


def compare_prices(product_name: str):
    if not product_name or not product_name.strip():
        return (
            None,
            "<p style='color:#e53935;text-align:center;font-size:16px;padding:20px'>⚠️ Please enter a product name.</p>",
            "",
        )

    # Normalize query via HF model
    normalized = normalize_query(product_name.strip())
    status_html = (
        f"<p style='color:rgba(255,255,255,0.65);font-size:13px;text-align:center;margin:4px 0 12px'>"
        f"🔍 Searching for: <em><strong style='color:white'>{normalized}</strong></em>"
        f" &nbsp;·&nbsp; Powered by Qwen 7B</p>"
    )

    # Fetch all prices (HF model + DDG links)
    amazon, flipkart, myntra, image_url = get_all_prices(normalized)

    # Load product image
    product_image = load_image_from_url(image_url) if image_url else None

    # Build price cards
    results = [amazon, flipkart, myntra]
    best_platform = find_best_deal(results)

    cards = ""
    for r in results:
        card = build_price_card(r)
        if best_platform and r.get("platform") == best_platform and r.get("price"):
            card = card.replace('class="price-card found"', 'class="price-card found best-deal"')
        cards += card

    best_html = ""
    if best_platform:
        bc = PLATFORM_COLORS.get(best_platform, "#444")
        best_html = (
            f'<div style="text-align:center;margin-top:18px;">'
            f'<span class="best-banner">🏆 Best Deal on '
            f'<strong style="color:{bc}">{best_platform}</strong>!</span>'
            f'</div>'
        )

    cards_html = f'<div class="cards-row">{cards}</div>{best_html}'

    return product_image, status_html, cards_html


# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

/* ── Header ── */
.app-header { text-align: center; padding: 36px 20px 8px; }
.app-logo {
    font-size: 52px; font-weight: 800; margin: 0; line-height: 1;
    background: linear-gradient(90deg, #FF9900 0%, #FF3F6C 50%, #2874F0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -1.5px;
}
.app-sub { color: rgba(255,255,255,0.65); font-size: 15px; margin: 10px 0 0; font-weight: 400; }
.plat-pills { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:14px 0 4px; }
.pill {
    border-radius: 50px; padding: 5px 16px; font-size: 12px; font-weight: 600;
    border: 1.5px solid; background: transparent;
}

/* ── Search ── */
#search-input textarea, #search-input input {
    background: rgba(255,255,255,0.07) !important;
    border: 2px solid rgba(255,255,255,0.18) !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-size: 16px !important;
    padding: 13px 18px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .2s, box-shadow .2s;
}
#search-input textarea:focus, #search-input input:focus {
    border-color: #FF3F6C !important;
    box-shadow: 0 0 0 3px rgba(255,63,108,.2) !important;
}
#search-input label { color: rgba(255,255,255,.75) !important; font-weight: 600 !important; font-size: 13px !important; }

#compare-btn {
    background: linear-gradient(135deg, #FF3F6C, #FF9900) !important;
    border: none !important; border-radius: 14px !important;
    font-size: 15px !important; font-weight: 700 !important;
    color: #fff !important; padding: 13px 28px !important;
    box-shadow: 0 4px 20px rgba(255,63,108,.35) !important;
    transition: transform .2s, box-shadow .2s !important;
    width: 100% !important; margin-top: 10px !important;
}
#compare-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(255,63,108,.5) !important; }

/* ── Image panel ── */
#product-image { background: transparent !important; }
#product-image .image-container, #product-image .svelte-1ysrlvx {
    background: rgba(255,255,255,.06) !important;
    border: 2px dashed rgba(255,255,255,.18) !important;
    border-radius: 20px !important;
}
#product-image img { border-radius: 16px; object-fit: contain; max-height: 300px; }

/* ── Price Cards ── */
.cards-row {
    display: flex; gap: 18px; justify-content: center;
    flex-wrap: wrap; padding: 10px 0;
}
.price-card {
    background: #fff; border-radius: 20px;
    padding: 24px 20px; width: 210px; min-width: 180px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.22);
    transition: transform .25s, box-shadow .25s;
    position: relative;
}
.price-card:hover { transform: translateY(-6px); box-shadow: 0 16px 40px rgba(0,0,0,.3); }
.price-card.best-deal { box-shadow: 0 8px 32px rgba(0,0,0,.25), 0 0 0 3px gold; }
.price-card.not-found { opacity: .55; }

.platform-header { display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:12px; }
.platform-icon { font-size: 22px; }
.platform-name { font-size: 17px; font-weight: 700; }
.product-title { font-size: 11px; color: #777; margin-bottom: 8px; line-height: 1.45; min-height: 28px; }
.price-value { font-size: 32px; font-weight: 800; margin: 6px 0 14px; letter-spacing: -.5px; }
.price-na { font-size: 15px; color: #bbb; font-weight: 500; margin: 10px 0 14px; }
.view-btn {
    display: inline-block; color: #fff !important; text-decoration: none !important;
    border-radius: 50px; padding: 7px 16px; font-size: 12px; font-weight: 600;
    transition: opacity .2s;
}
.view-btn:hover { opacity: .82; }

/* ── Best deal banner ── */
.best-banner {
    display: inline-block;
    background: rgba(255,215,0,.12);
    border: 1.5px solid rgba(255,215,0,.4);
    border-radius: 50px; padding: 10px 24px;
    color: #FFD700; font-size: 14px; font-weight: 600;
}

/* ── Examples ── */
.gr-examples label { color: rgba(255,255,255,.6) !important; font-size: 12px !important; }
.gr-examples .gr-button { background: rgba(255,255,255,.08) !important; color: rgba(255,255,255,.8) !important;
    border: 1px solid rgba(255,255,255,.15) !important; border-radius: 8px !important; font-size: 12px !important; }

footer, .built-with { display: none !important; }
"""

HEADER_HTML = """
<div class="app-header">
  <h1 class="app-logo">Come &amp; Compare</h1>
  <p class="app-sub">Real-time price comparison · Powered by a small language model (Qwen 7B &le;32B)</p>
  <div class="plat-pills">
    <span class="pill" style="color:#FF9900;border-color:#FF9900;">🛒 Amazon</span>
    <span class="pill" style="color:#2874F0;border-color:#2874F0;">🛍️ Flipkart</span>
    <span class="pill" style="color:#FF3F6C;border-color:#FF3F6C;">👗 Myntra</span>
  </div>
</div>
"""

EXAMPLES = [
    ["iPhone 15 128GB"],
    ["Samsung Galaxy S24"],
    ["Sony WH-1000XM5 headphones"],
    ["Nike Air Max running shoes"],
    ["boAt Airdopes 141"],
    ["Lakme face powder"],
]

# ─── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Come & Compare — Price Comparison") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            search_input = gr.Textbox(
                label="Enter Product Name",
                placeholder="e.g. iPhone 15, Nike Air Max, boAt Airdopes...",
                elem_id="search-input",
                lines=1,
            )
            compare_btn = gr.Button("🔍 Compare Prices Now", elem_id="compare-btn", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[search_input], label="Quick Examples")

    status_out = gr.HTML()

    with gr.Row():
        product_image = gr.Image(
            label="Product Image",
            elem_id="product-image",
            show_label=False,
            height=280,
        )

    cards_out = gr.HTML()

    compare_btn.click(
        fn=compare_prices,
        inputs=[search_input],
        outputs=[product_image, status_out, cards_out],
    )
    search_input.submit(
        fn=compare_prices,
        inputs=[search_input],
        outputs=[product_image, status_out, cards_out],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CSS,
        footer_links=[],
        quiet=True,
    )
