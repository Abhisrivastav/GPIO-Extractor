"""
GPIO Extractor - Flask application
Accepts a board schematic PDF and returns an HTML report of all GPIO signals.
Uses Azure OpenAI GPT-4o + GPT-4o Vision for intelligent extraction.

Extraction strategy:
  1. Try text extraction from PDF (works for text-based PDFs)
  2. If text is sparse (<80 chars/page avg), switch to VISION mode:
     - Render each page to a PNG image
     - Send images to GPT-4o Vision
     - AI reads the schematic visually — handles image-based PDFs (Altium, OrCAD, KiCad exports)
"""

import os
import re
import json
import base64
import tempfile
import io
from pathlib import Path

import fitz                       # PyMuPDF  (pip install pymupdf)
import pdfplumber                 # table extractor
import httpx
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from openai import AzureOpenAI

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Network / SSL bootstrap ────────────────────────────────────────────────
# Private endpoint is the exclusive access method.
# When AZURE_PRIVATE_ENDPOINT_IP is set we bypass both DNS and the corporate
# internet proxy by connecting directly to the private IP while keeping the
# correct TLS SNI / Host header so the server accepts the connection.

_private_ip   = os.getenv("AZURE_PRIVATE_ENDPOINT_IP", "").strip()
_raw_proxy     = os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", ""))
_azure_host   = (os.getenv("AZURE_OPENAI_ENDPOINT", "")
                 .replace("https://","").replace("http://","")
                 .split("/")[0])

# Corporate SSL cert bundle (includes Intel TLS-inspection CA)
_cert_bundle = os.getenv("SSL_CERT_FILE", "")
if not _cert_bundle:
    _local_cert = os.path.join(os.path.dirname(__file__), "intel_certs.pem")
    if os.path.exists(_local_cert):
        _cert_bundle = _local_cert
if _cert_bundle and os.path.exists(_cert_bundle):
    os.environ["SSL_CERT_FILE"]      = _cert_bundle
    os.environ["REQUESTS_CA_BUNDLE"] = _cert_bundle
    print(f"[SSL]  Cert bundle: {os.path.basename(_cert_bundle)}")
else:
    _cert_bundle = True


def _make_http_client() -> httpx.Client:
    """
    Build an httpx.Client configured for private endpoint access.

    Strategy (in priority order):
    1. AZURE_PRIVATE_ENDPOINT_IP is set → use a custom transport that rewrites
       the target URL to the private IP while keeping the correct Host header.
       This completely bypasses DNS and the internet proxy.
    2. No private IP configured → fall back to internet proxy if available.
    """
    cert = (_cert_bundle
            if isinstance(_cert_bundle, str) and os.path.exists(_cert_bundle)
            else True)

    if _private_ip and _azure_host:
        # Build a transport with a pre-resolved address map so httpx connects
        # to the private IP without touching DNS or the proxy
        print(f"[Net]  Private endpoint: {_azure_host} → {_private_ip} (proxy bypassed)")
        return httpx.Client(
            verify=cert,
            timeout=60,
            # Pin the hostname → private IP at the transport level
            # Works with httpx >= 0.24  (the 'local_address' / 'uds' trick won't work;
            # we use the `base_url` override in the AzureOpenAI client instead)
        )

    # No private IP — use internet proxy if configured
    proxy = _raw_proxy or None
    if proxy:
        print(f"[Net]  Internet proxy: {proxy}")
    return httpx.Client(proxy=proxy, verify=cert, timeout=60)


# ── Apply DNS override at socket level ─────────────────────────────────────
# Patches socket.getaddrinfo so ALL HTTP libraries (httpx, requests, urllib)
# resolve the Azure hostname to the private IP — completely bypassing
# the corporate internet proxy and public DNS. No hosts file / admin needed.
if _private_ip and _azure_host:
    import socket as _socket
    _orig_getaddrinfo = _socket.getaddrinfo

    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if isinstance(host, str) and (host == _azure_host or
                                       host.endswith(".openai.azure.com") or
                                       host.endswith(".privatelink.openai.azure.com")):
            print(f"[DNS]  {host}:{port} → {_private_ip} (private endpoint)")
            return _orig_getaddrinfo(_private_ip, port, *args, **kwargs)
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    _socket.getaddrinfo = _patched_getaddrinfo

    # Also clear proxy env vars so httpx doesn't route to the internet proxy
    for _ev in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(_ev, None)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB max upload

# Azure OpenAI credentials (from portal.azure.com)
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT",   "")
AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY",        "")
AZURE_OPENAI_API_VERSION= os.getenv("AZURE_OPENAI_API_VERSION","2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Also support ExperGPT pak_ keys as fallback
EXPERGPT_BASE_URL = os.getenv("EXPERGPT_BASE_URL", "")
EXPERGPT_API_KEY  = os.getenv("EXPERGPT_API_KEY",  "")

# Determine which backend to use: Azure takes priority, then ExperGPT
USE_AZURE    = bool(AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT)
USE_EXPERGPT = bool(not USE_AZURE and EXPERGPT_API_KEY and EXPERGPT_API_KEY.startswith("pak_"))
USE_AI       = USE_AZURE or USE_EXPERGPT

# Display name + model key
if USE_AZURE:
    AI_LABEL = f"Azure OpenAI GPT-4o (deployment: {AZURE_OPENAI_DEPLOYMENT})"
    AI_MODEL  = AZURE_OPENAI_DEPLOYMENT
elif USE_EXPERGPT:
    AI_LABEL = "Intel ExperGPT (GPT-4o)"
    AI_MODEL  = "gpt-4o"
else:
    AI_LABEL = "No AI — Regex fallback"
    AI_MODEL  = ""

# How many chars per page before we consider text extraction "sufficient"
TEXT_THRESHOLD_PER_PAGE = 80
# DPI for rendering schematic pages to images (higher = better but slower)
RENDER_DPI = 150
# Max pages to process in vision mode (cost control)
MAX_VISION_PAGES = 30


def get_ai_client():
    """Return a configured AI client for the active backend."""
    if USE_AZURE:
        http_client = _make_http_client()   # uses corporate proxy for Azure
        return AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            http_client=http_client,
        )
    from openai import OpenAI
    # ExpertGPT is an internal Intel address (10.184.69.166) — no proxy needed.
    # trust_env=False prevents httpx from picking up HTTPS_PROXY from the env
    # (which was set by load_dotenv() and would route internal traffic through
    #  the corporate internet proxy, causing a 401/403).
    cert = (_cert_bundle
            if isinstance(_cert_bundle, str) and os.path.exists(_cert_bundle)
            else True)
    expergpt_client = httpx.Client(verify=cert, timeout=60, trust_env=False)
    print(f"[Net]  ExpertGPT: direct (no proxy, trust_env=False) — {EXPERGPT_BASE_URL}")
    return OpenAI(
        api_key=EXPERGPT_API_KEY,
        base_url=EXPERGPT_BASE_URL,
        http_client=expergpt_client,
    )


# ── PDF Password Detection ─────────────────────────────────────────────────
def check_pdf_password(pdf_path: str) -> dict:
    """
    Check if a PDF is encrypted/password-protected.
    Returns {"encrypted": bool, "needs_password": bool}
    """
    try:
        doc = fitz.open(pdf_path)
        is_encrypted = doc.is_encrypted
        needs_password = False
        if is_encrypted:
            # Try to authenticate with empty password (owner-only restriction)
            auth_result = doc.authenticate("")
            # auth_result == 0 means needs a real password
            # auth_result != 0 means empty password worked (restriction-only, no password needed)
            needs_password = (auth_result == 0)
        doc.close()
        return {"encrypted": is_encrypted, "needs_password": needs_password}
    except Exception as e:
        return {"encrypted": False, "needs_password": False, "error": str(e)}


# ── PDF Open Helper ────────────────────────────────────────────────────────
def open_pdf(pdf_path: str, password: str = "") -> fitz.Document:
    """Open a PDF document, authenticating with password if needed."""
    doc = fitz.open(pdf_path)
    if doc.is_encrypted:
        result = doc.authenticate(password)
        if result == 0:
            doc.close()
            raise ValueError("Incorrect PDF password. Please enter the correct password.")
    return doc


# ── PDF Text Extraction ─────────────────────────────────────────────────────
def extract_text_pymupdf(pdf_path: str, password: str = "") -> tuple[str, int]:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns (combined_text, total_char_count).
    Tries multiple extraction methods to maximise text recovered.
    """
    doc = open_pdf(pdf_path, password)
    pages_text = []
    total_chars = 0
    for i, page in enumerate(doc):
        # Method 1: standard text
        text = page.get_text("text")
        if len(text.strip()) < 20:
            # Method 2: blocks (preserves layout better)
            blocks = page.get_text("blocks")
            text = "\n".join(b[4] for b in blocks if isinstance(b[4], str))
        if len(text.strip()) < 20:
            # Method 3: rawdict (catches more Unicode-mapped fonts)
            raw = page.get_text("rawdict")
            parts = []
            for block in raw.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            parts.append(span.get("text", ""))
            text = " ".join(parts)
        total_chars += len(text.strip())
        if text.strip():
            pages_text.append(f"--- PAGE {i+1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages_text), total_chars


def extract_tables_pdfplumber(pdf_path: str, password: str = "") -> str:
    """Extract tables from a PDF using pdfplumber — good for schematic net tables."""
    rows = []
    try:
        with pdfplumber.open(pdf_path, password=password if password else None) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in table:
                        clean = [str(c).strip() if c else "" for c in row]
                        rows.append(" | ".join(clean))
    except Exception:
        pass
    return "\n".join(rows)


def is_image_based_pdf(total_chars: int, page_count: int) -> bool:
    """Return True if the PDF has too little text — i.e. pages are images/vector graphics."""
    if page_count == 0:
        return True
    avg = total_chars / page_count
    return avg < TEXT_THRESHOLD_PER_PAGE


# ── Page → Base64 PNG (for Vision API) ─────────────────────────────────────
def pdf_pages_to_b64_images(pdf_path: str, password: str = "", dpi: int = RENDER_DPI,
                             max_pages: int = MAX_VISION_PAGES) -> list[dict]:
    """
    Render each PDF page to a base64-encoded PNG.
    Returns list of {"page": n, "b64": "..."} dicts.
    """
    doc = open_pdf(pdf_path, password)
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # scale factor from 72 DPI base
    num_pages = min(doc.page_count, max_pages)
    for i in range(num_pages):
        page = doc[i]
        pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pixmap.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        images.append({"page": i + 1, "b64": b64})
        pixmap = None  # free memory
    doc.close()
    return images


# ── AI Prompts ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEXT = """You are an expert board designer and firmware engineer specialising in Intel SoC schematics.
Extract ALL GPIO signal information from the schematic text provided.

For every GPIO found return a JSON object with these exact keys:
  gpio_num      – GPIO pad name (e.g. GPP_A0, GPIO_B2, GPD5)
  net_name      – Net / signal name connected to this pad (e.g. SPI_CLK, UART0_TX)
  direction     – Input | Output | Bidirectional | Unknown
  voltage       – 1.8V | 3.3V | 5V | Unknown
  function      – What this signal does on the board (e.g. "SPI Clock to BIOS Flash")
  connected_to  – Component or connector this signal goes to (e.g. "U12 - TPM", "J3 Pin 4")
  pull          – Pull-up 10K | Pull-down 100K | None | Unknown

Return ONLY a JSON array, no markdown fences, no explanation.
If a field is not determinable, use "Unknown".
Extract EVERY GPIO reference — do not skip any.
"""

SYSTEM_PROMPT_VISION = """You are an expert board designer and firmware engineer.
You are looking at a page from an Intel SoC board schematic diagram.

Identify and extract EVERY GPIO signal visible on this schematic page.
Look for:
- Intel pad names: GPP_A0..GPP_G23, GPD0..GPD11, GPIOA0..GPIOB7, GPIO0..GPIO127
- Net labels connected to GPIO pads
- Direction arrows on schematic symbols
- Pull-up / pull-down resistors associated with GPIO pins
- Component labels indicating what the GPIO connects to
- Voltage annotations (VCC3_3, VDDIO_1V8, etc.)

For each GPIO found return:
{
  "gpio_num":    "GPP_A0",
  "net_name":    "SPI_CS#",
  "direction":   "Output",
  "voltage":     "3.3V",
  "function":    "SPI Chip Select to BIOS Flash U5",
  "connected_to":"U5 Pin 1 - 256Mb SPI NOR Flash",
  "pull":        "Pull-up 10K to VCC3_3"
}

Return ONLY a valid JSON array. No markdown, no text outside the array.
If a page has no GPIOs, return [] (empty array).
"""


def _parse_ai_response(raw: str, context: str = "") -> list:
    """Parse a JSON array from LLM output, with multiple fallback strategies."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Find first [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except Exception:
            pass

    # Extract individual {...} objects
    objects = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
    if objects:
        gpios = []
        for obj in objects:
            try:
                gpios.append(json.loads(obj))
            except Exception:
                pass
        if gpios:
            return gpios

    print(f"[AI parse failed{' ' + context if context else ''}] raw: {raw[:200]}")
    return []


def _deduplicate(gpios: list) -> list:
    """Remove duplicate GPIOs by gpio_num+net_name key."""
    seen = set()
    unique = []
    for g in gpios:
        key = (str(g.get("gpio_num", "")).upper(), str(g.get("net_name", "")).upper())
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


# ── Text-Based AI Extraction ────────────────────────────────────────────────
def extract_gpios_ai_text(text: str) -> tuple[list, list]:
    """
    Use ExperGPT (GPT-4o) text mode to extract GPIOs from schematic text.
    Returns (gpio_list, error_list).
    """
    client = get_ai_client()
    max_chars = 90_000
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    all_gpios, errors = [], []

    for idx, chunk in enumerate(chunks):
        try:
            resp = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                    {"role": "user",   "content":
                     f"Board schematic text (chunk {idx+1}/{len(chunks)}):\n\n{chunk}"}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content
            gpios = _parse_ai_response(raw, f"text-chunk-{idx+1}")
            all_gpios.extend(gpios)
        except Exception as e:
            msg = f"Text chunk {idx+1}: {type(e).__name__}: {e}"
            print(f"[AI text error] {msg}")
            errors.append(msg)

    return _deduplicate(all_gpios), errors


# ── Vision-Based AI Extraction ──────────────────────────────────────────────
def extract_gpios_ai_vision(pdf_path: str, password: str = "") -> tuple[list, list]:
    """
    Use GPT-4o Vision to extract GPIOs from rendered schematic page images.
    This works for image-based PDFs (Altium, OrCAD, KiCad exports) where
    text extraction yields nothing.
    Returns (gpio_list, error_list).
    """
    client = get_ai_client()
    page_images = pdf_pages_to_b64_images(pdf_path, password)
    all_gpios, errors = [], []

    print(f"[Vision] Processing {len(page_images)} pages at {RENDER_DPI} DPI")

    # Process pages in batches of 2 (balance context vs cost)
    BATCH = 2
    for batch_start in range(0, len(page_images), BATCH):
        batch = page_images[batch_start:batch_start + BATCH]
        page_nums = [b["page"] for b in batch]
        print(f"[Vision] Sending pages {page_nums} to GPT-4o Vision")

        # Build vision message with multiple images
        content = [
            {"type": "text",
             "text": f"Schematic pages {page_nums}. Extract all GPIOs from all pages shown."}
        ]
        for img_data in batch:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_data['b64']}",
                    "detail": "high"
                }
            })

        try:
            resp = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_VISION},
                    {"role": "user",   "content": content}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content
            gpios = _parse_ai_response(raw, f"vision-pages-{page_nums}")
            print(f"[Vision] Pages {page_nums}: found {len(gpios)} GPIOs")
            all_gpios.extend(gpios)
        except Exception as e:
            msg = f"Vision pages {page_nums}: {type(e).__name__}: {e}"
            print(f"[AI vision error] {msg}")
            errors.append(msg)

    return _deduplicate(all_gpios), errors


# ── Regex-Based GPIO Extraction (fallback, no API key needed) ───────────────
# Patterns for common Intel-style GPIO naming conventions
GPIO_PATTERNS = [
    r"\bGPP_[A-Z]\d{1,2}\b",          # GPP_A0 .. GPP_G23
    r"\bGPIO_[A-Z]?\d{1,3}\b",        # GPIO_A0, GPIO0 .. GPIO127
    r"\bGPIO[A-Z]?\d{1,3}\b",         # GPIOA3
    r"\bGPD\d{1,2}\b",                # GPD0..GPD11 (deep sleep GPIOs)
    r"\bGPE\d{1,2}\b",                # GPE0..GPE7
    r"\bGPSM\d{1,2}\b",               # GPSM0..GPSM7
    r"\bGPCOM[0-5]_GPIO\d{1,3}\b",    # full qualified
    r"\bHDATA[AB]\b",                  # HDA GPIO pads
    r"\bJTAG_\w+\b",                   # JTAG GPIO multi-use
    r"\bSD\d_DATA[0-3]\b",             # SDIO data pads
    r"\bSATAGP\d\b",                   # SATA GP GPIO
]

DIRECTION_KEYWORDS = {
    "input":  ["IN", "INPUT", "RXD", "RX", "MISO", "SDI", "SDA", "PRSNT#", "IRQ", "INT#", "ALERT#", "INT"],
    "output": ["OUT", "OUTPUT", "TXD", "TX", "MOSI", "SDO", "CLK", "CS#", "RST#", "RESET#", "PERST#", "EN", "ENABLE"],
    "bidir":  ["GPIO", "BI", "BIDIR", "IO", "SDA", "SDIO"],
}

def guess_direction(net: str) -> str:
    net_up = net.upper()
    for d, kws in DIRECTION_KEYWORDS.items():
        if any(k in net_up for k in kws):
            if d == "input":  return "Input"
            if d == "output": return "Output"
            if d == "bidir":  return "Bidirectional"
    return "Unknown"

def guess_function(net: str) -> str:
    net_up = net.upper()
    mapping = {
        "SPI": "SPI Bus (BIOS Flash / TPM)",
        "UART": "UART Serial Debug",
        "I2C": "I2C Bus",
        "SMBUS": "SMBus / I2C",
        "PCIE": "PCIe Control Signal",
        "USB": "USB Control",
        "SD": "SD / eMMC Storage",
        "SATA": "SATA Storage",
        "JTAG": "JTAG Debug Interface",
        "PWM": "PWM Fan / LED Control",
        "HDA": "HD Audio",
        "GPIO": "General Purpose I/O",
        "INT": "Interrupt Signal",
        "RST": "Reset Control",
        "PERST": "PCIe Reset",
        "LED": "LED Control",
        "SMI": "System Management Interrupt",
        "WAKE": "Wake-on Event Signal",
        "CLKREQ": "PCIe Clock Request",
    }
    for key, desc in mapping.items():
        if key in net_up:
            return desc
    return "General Purpose I/O"

def extract_gpios_regex(text: str) -> list:
    """Pure regex GPIO extraction — works without any API key."""
    combined_pattern = "|".join(GPIO_PATTERNS)
    matches = re.findall(combined_pattern, text, re.IGNORECASE)

    # Try to grab the net name that appears near the GPIO reference
    gpios = []
    seen = set()
    lines = text.splitlines()

    for line in lines:
        gpio_hits = re.findall(combined_pattern, line, re.IGNORECASE)
        for gpio in gpio_hits:
            gpio_norm = gpio.upper().strip()
            if gpio_norm in seen:
                continue
            seen.add(gpio_norm)

            # Grab any CAPS_NET_NAME nearby
            nets = re.findall(r"\b[A-Z][A-Z0-9_#]{2,30}\b", line)
            nets = [n for n in nets if n != gpio_norm and not re.match(r"^(PAGE|NET|REF|VALUE|PART|TYPE|GPIO)$", n)]
            net_name = nets[0] if nets else "UNKNOWN_NET"

            gpios.append({
                "gpio_num":     gpio_norm,
                "net_name":     net_name,
                "direction":    guess_direction(net_name),
                "voltage":      "Unknown",
                "function":     guess_function(net_name),
                "connected_to": "See schematic",
                "pull":         "Unknown",
            })

    # Also scan for standalone GPIO mentions without full context
    all_hits = re.findall(combined_pattern, text, re.IGNORECASE)
    for gpio in all_hits:
        gpio_norm = gpio.upper().strip()
        if gpio_norm not in seen:
            seen.add(gpio_norm)
            gpios.append({
                "gpio_num":     gpio_norm,
                "net_name":     "UNKNOWN_NET",
                "direction":    "Unknown",
                "voltage":      "Unknown",
                "function":     "General Purpose I/O",
                "connected_to": "See schematic",
                "pull":         "Unknown",
            })

    return sorted(gpios, key=lambda x: x["gpio_num"])


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", use_ai=USE_AI, use_azure=USE_AZURE,
                           ai_label=AI_LABEL, deployment=AZURE_OPENAI_DEPLOYMENT)


@app.route("/check-password", methods=["POST"])
def check_password_route():
    """Check if an uploaded PDF is password-protected before full extraction."""
    if "schematic" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["schematic"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = check_pdf_password(tmp_path)
        return jsonify(result)
    finally:
        os.unlink(tmp_path)


@app.route("/debug-extract", methods=["POST"])
def debug_extract():
    """
    Debug endpoint — returns raw extracted text and metadata without calling AI.
    Use this to verify text extraction is working before enabling AI mode.
    """
    if "schematic" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["schematic"]
    pdf_password = request.form.get("pdf_password", "").strip()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        doc = open_pdf(tmp_path, pdf_password)
        pages = doc.page_count
        doc.close()

        text, total_chars = extract_text_pymupdf(tmp_path, pdf_password)
        avg_per_page = total_chars / max(pages, 1)
        is_image = is_image_based_pdf(total_chars, pages)

        return jsonify({
            "filename":        f.filename,
            "pages":           pages,
            "text_chars":      total_chars,
            "avg_chars_page":  round(avg_per_page, 1),
            "is_image_based":  is_image,
            "extraction_mode": "VISION" if is_image else "TEXT",
            "text_sample":     text[:2000] if text else "(no text extracted)",
        })
    except ValueError as e:
        return jsonify({"error": "wrong_password", "message": str(e)}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/extract", methods=["POST"])
def extract():
    if "schematic" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["schematic"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    pdf_password = request.form.get("pdf_password", "").strip()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        # 0. Password check
        pw_check = check_pdf_password(tmp_path)
        if pw_check.get("needs_password") and not pdf_password:
            return jsonify({"error": "password_required",
                            "message": "This PDF is password protected. Please enter the password."}), 403

        # 1. Open and get page count
        try:
            doc = open_pdf(tmp_path, pdf_password)
        except ValueError as e:
            return jsonify({"error": "wrong_password", "message": str(e)}), 403
        pages = doc.page_count
        doc.close()

        # 2. Try text extraction first
        try:
            text, total_chars = extract_text_pymupdf(tmp_path, pdf_password)
        except ValueError as e:
            return jsonify({"error": "wrong_password", "message": str(e)}), 403

        avg_per_page = total_chars / max(pages, 1)
        use_vision   = is_image_based_pdf(total_chars, pages)

        print(f"[Extract] '{f.filename}' | pages={pages} | chars={total_chars} "
              f"| avg/page={avg_per_page:.0f} | mode={'VISION' if use_vision else 'TEXT'} "
              f"| ai={USE_AI}")

        # 3. Extract GPIOs
        ai_errors = []
        extraction_mode = "unknown"

        if USE_AI:
            if use_vision:
                # PDF is image-based — use GPT-4o Vision on rendered page images
                extraction_mode = f"AI Vision — Intel ExperGPT GPT-4o (image-based PDF, {min(pages, MAX_VISION_PAGES)} pages rendered)"
                gpios, ai_errors = extract_gpios_ai_vision(tmp_path, pdf_password)
            else:
                # PDF has extractable text — use text mode (faster, cheaper)
                tables = extract_tables_pdfplumber(tmp_path, pdf_password)
                if tables.strip():
                    text += "\n\n--- EXTRACTED TABLES ---\n" + tables
                extraction_mode = "AI Text — Intel ExperGPT GPT-4o (text-based PDF)"
                gpios, ai_errors = extract_gpios_ai_text(text)

            # If AI text mode found nothing but PDF is borderline, retry with vision
            if not gpios and not use_vision and USE_AI and pages <= MAX_VISION_PAGES:
                print("[Extract] Text AI returned 0 GPIOs — retrying with Vision mode")
                extraction_mode += " → Vision fallback"
                gpios, vision_errors = extract_gpios_ai_vision(tmp_path, pdf_password)
                ai_errors.extend(vision_errors)
        else:
            # No API key — use regex pattern matching
            tables = extract_tables_pdfplumber(tmp_path, pdf_password)
            if tables.strip():
                text += "\n\n--- EXTRACTED TABLES ---\n" + tables
            gpios = extract_gpios_regex(text)
            extraction_mode = "Regex Pattern Matching (no ExperGPT API key)"
            if use_vision and not gpios:
                ai_errors.append(
                    "This PDF appears to be image-based — GPIOs could not be extracted without an AI key. "
                    "Please set EXPERGPT_API_KEY in .env to enable Vision mode."
                )

        return jsonify({
            "filename":        f.filename,
            "pages":           pages,
            "gpio_count":      len(gpios),
            "method":          extraction_mode,
            "gpios":           gpios,
            "text_chars":      total_chars,
            "avg_chars_page":  round(avg_per_page, 1),
            "pdf_mode":        "image-based" if use_vision else "text-based",
            "ai_errors":       ai_errors,
        })

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 65)
    print("  GPIO Extractor — Board Schematic Analyzer")
    print("=" * 65)
    if USE_AZURE:
        print(f"  AI Mode   : ENABLED — Azure OpenAI")
        print(f"  Endpoint  : {AZURE_OPENAI_ENDPOINT}")
        print(f"  Deployment: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"  Network   : {'Private Endpoint (' + _private_ip + ')' if _private_ip else 'Direct (no proxy)'}")
        print(f"  Vision    : Auto-enabled for image-based PDFs")
    elif USE_EXPERGPT:
        print(f"  AI Mode   : ENABLED — Intel ExperGPT (GPT-4o)")
        print(f"  Endpoint  : {EXPERGPT_BASE_URL}")
    else:
        print("  AI Mode   : DISABLED (regex fallback)")
        print("  Tip       : Set AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT in .env")
    print("  URL       : http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, port=5000)

