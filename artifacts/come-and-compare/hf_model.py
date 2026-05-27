import os
import re
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def normalize_query(raw_query: str) -> str:
    """Use HuggingFace model to produce a clean, searchable product query."""
    if not HF_TOKEN or not raw_query.strip():
        return raw_query.strip()
    try:
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product search assistant for Indian e-commerce. "
                        "Given a product name, output ONLY a short, clean search query "
                        "(max 8 words) suitable for Amazon India, Flipkart, and Myntra. "
                        "No explanations, no punctuation at the end."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Normalize this product for search: {raw_query}",
                },
            ],
            model=MODEL_ID,
            max_tokens=25,
            temperature=0.1,
        )
        cleaned = resp.choices[0].message.content.strip().strip('"').strip("'")
        if cleaned and 3 < len(cleaned) < 100:
            return cleaned
        return raw_query.strip()
    except Exception:
        return raw_query.strip()


def fill_missing_prices(query: str, amazon_price: str | None, flipkart_price: str | None, myntra_price: str | None) -> dict:
    """
    Use the HF model to fill in prices that scraping couldn't find.
    Returns dict with keys 'amazon', 'flipkart', 'myntra' for any missing prices.
    """
    missing = []
    if not amazon_price:
        missing.append("Amazon India")
    if not flipkart_price:
        missing.append("Flipkart")
    if not myntra_price:
        missing.append("Myntra")

    if not missing or not HF_TOKEN:
        return {}

    try:
        client = InferenceClient(token=HF_TOKEN)
        platforms_str = ", ".join(missing)
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a real-time Indian e-commerce price assistant. "
                        "Provide the current approximate selling price of products on Indian platforms. "
                        "Give ONLY prices in the format: Platform: ₹XXXX. One per line. "
                        "Use your knowledge of current Indian market prices. "
                        "If you truly don't know the price, write Platform: N/A"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"What is the current price of '{query}' on these Indian platforms: {platforms_str}? "
                        f"Answer with just the prices."
                    ),
                },
            ],
            model=MODEL_ID,
            max_tokens=120,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        result = {}
        price_re = re.compile(r"(?:₹|Rs\.?)\s*([\d,]+)")

        for line in text.splitlines():
            line_lower = line.lower()
            price_m = price_re.search(line)
            if not price_m:
                continue
            raw_val = price_m.group(1).replace(",", "")
            try:
                val = int(float(raw_val))
                if val < 50 or val > 10_000_000:
                    continue
                price_str = f"₹{val:,}"
            except ValueError:
                continue

            if "amazon" in line_lower and not amazon_price:
                result["amazon"] = price_str
            elif "flipkart" in line_lower and not flipkart_price:
                result["flipkart"] = price_str
            elif "myntra" in line_lower and not myntra_price:
                result["myntra"] = price_str

        return result
    except Exception:
        return {}
