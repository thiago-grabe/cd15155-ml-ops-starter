# Production image for the FinBERT sentiment API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PYTHONPATH=/app

WORKDIR /app

# --- Corporate TLS-inspection trust (local builds only) ----------------------
# This directory is empty in git. On networks running a TLS-inspecting proxy
# (Zscaler here), dropping the proxy root CA into docker/certs/*.crt makes pip and
# huggingface_hub able to verify download.pytorch.org / huggingface.co, which are
# re-signed in transit. On GitHub Actions the directory is empty and this is a no-op.
# See docker/certs/README.md.
COPY docker/certs/ /usr/local/share/ca-certificates/
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# CPU-only torch, installed on its own and FIRST.
# --index-url REPLACES PyPI rather than adding to it, and the pytorch CPU index returns
# 404/403 for fastapi, pandas, transformers... so this cannot be merged with the next
# install. Keeping it separate also avoids pip resolving the default PyPI torch wheel,
# which is 888 MB and drags in ~15 nvidia-*/triton CUDA packages (~6 GB image).
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake and patch the model into the image.
# The upstream config.json stores id2label as floats (0.0/1.0/2.0), which newer
# transformers rejects; scripts/smoke_test.py performs the same fix for local runs.
# Baking matters most on ECS: a cold task would otherwise spend ~3 min pulling 440 MB
# and get killed by the health check before it ever answers.
ARG BAKE_MODEL=1
ARG HF_MODEL_ID=baptle/FinBERT_market_based
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "\
import json, os; \
from huggingface_hub import snapshot_download; \
p = snapshot_download('${HF_MODEL_ID}', allow_patterns=['*.json','*.txt','*.safetensors']); \
c = os.path.join(p, 'config.json'); \
d = json.load(open(c)); \
d['id2label'] = {'0':'negative','1':'neutral','2':'positive'}; \
d['label2id'] = {'negative':'0','neutral':'1','positive':'2'}; \
json.dump(d, open(c,'w'), indent=2); \
print('baked and patched', '${HF_MODEL_ID}')"; \
    fi

COPY app/ ./app/

EXPOSE 8000

# python:3.12-slim is Debian 13 (trixie) and ships NEITHER curl NOR wget at runtime,
# so the probe uses the standard library instead of adding an apt layer.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
