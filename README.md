# GPIO Extractor — Board Schematic Analyzer

A Flask web application that extracts and describes all GPIO signals from board schematic PDFs using AI (Intel ExpertGPT / Azure OpenAI GPT-4o Vision).

## Features

- **Upload any schematic PDF** — drag-and-drop or file picker
- **AI-powered extraction** — uses GPT-4o Vision to read image-based PDFs (Altium, OrCAD, KiCad exports)
- **Text + Vision modes** — auto-detects whether the PDF is text-based or image-based
- **Password-protected PDF support** — prompts for password if required
- **HTML report output** — downloadable, searchable table of all GPIOs with signal name, direction, voltage, and description
- **Regex fallback** — works without an API key for basic extraction

## Requirements

- Python 3.12+  
- Intel internal network / VPN access (for ExpertGPT backend)
- [`uv`](https://github.com/astral-sh/uv) package manager (or pip)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Abhisrivastav/GPIO-Extractor.git
cd GPIO-Extractor

# 2. Create virtual environment and install dependencies
uv venv .venv
.venv\Scripts\activate          # Windows
uv pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and add your EXPERGPT_API_KEY

# 4. Run
start.bat                        # Windows double-click launcher
# or
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Description |
|---|---|
| `EXPERGPT_BASE_URL` | `https://expertgpt.intel.com/v1` |
| `EXPERGPT_API_KEY` | Your `pak_...` ExpertGPT key |

## How It Works

1. User uploads a schematic PDF (optionally provides password)
2. App extracts text with PyMuPDF — if average text < 80 chars/page, switches to **Vision mode**
3. In Vision mode, each page is rendered to PNG at 150 DPI and sent to GPT-4o Vision
4. AI returns structured JSON: GPIO name, direction, voltage level, description
5. Results rendered as a searchable HTML table with a download button

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Flask 3.1 |
| PDF text extraction | PyMuPDF (fitz), pdfplumber |
| AI backend | OpenAI SDK → Intel ExpertGPT |
| HTTP client | httpx |
| Environment config | python-dotenv |
