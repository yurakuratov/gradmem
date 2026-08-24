FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37

WORKDIR /home/jovyan/test-time-gd

# Install uv, then sync the locked environment (torch CUDA wheels included).
# `--frozen` honours uv.lock exactly; `--no-dev` skips the [dev] extra.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY . .

CMD ["bash"]
