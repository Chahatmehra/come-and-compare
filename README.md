# Come & Compare 🛒

Real-time product price comparison across Amazon India, Flipkart, and Myntra.

Built for the **HuggingFace Small Models Hackathon** — uses Qwen/Qwen2.5-7B-Instruct (7B, under 32B limit).

## Live Demo

Try it on HuggingFace Spaces: https://huggingface.co/spaces/PHOENIXREBORNAGAIN/hackathon

## How it works

1. Enter any product name (e.g. iPhone 15, Nike Air Max, boAt Airdopes 141)
2. Qwen 7B normalizes the query via HuggingFace Inference API
3. DuckDuckGo HTML search finds real product page links (no API key needed)
4. Qwen 7B estimates current prices on each platform
5. Best deal is highlighted, each card links to the actual product page
6. AI Analysis tab gives a full buying recommendation

## Project Structure

```
artifacts/come-and-compare/   # Local Gradio app (development)
  app.py                      # Gradio UI, layout, CSS, event handlers
  scraper.py                  # DDG search + Qwen price fetching + image retrieval
  hf_model.py                 # HuggingFace Inference API helper

hf-space/                     # HuggingFace Space (deployed)
  app.py                      # Full app for HF Spaces
  requirements.txt
```

## Tech Stack

- UI: Gradio 5
- AI Model: Qwen/Qwen2.5-7B-Instruct (7B parameters, under 32B hackathon limit)
- Price data: Qwen 7B via HuggingFace Inference API
- Product links: DuckDuckGo HTML search (no API key needed)
- Hosting: HuggingFace Spaces (free tier)

## Setup (Local)

```bash
pip install gradio requests beautifulsoup4 huggingface_hub lxml
export HF_TOKEN=your_hf_token_here
python artifacts/come-and-compare/app.py
```

## Setup (HuggingFace Space)

1. Upload hf-space/ contents to a new Gradio Space
2. Add HF_TOKEN in Space Settings → Variables and secrets
3. The app starts automatically — no other config needed

## Why no direct scraping?

Amazon, Flipkart, and Myntra all block cloud server IPs with bot protection.
Solution: DuckDuckGo HTML search for product links + Qwen 7B for price estimates.

## Hackathon

- Event: HuggingFace Build Small Hackathon 2025
- Constraint: Models must be under 32B parameters
- Model: Qwen/Qwen2.5-7B-Instruct (7 billion parameters)

## License

MIT
