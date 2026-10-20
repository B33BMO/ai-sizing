# Training/eval image: the same stack that runs on the RTX 5070 Ti box (torch 2.11 cu128, Blackwell-ready).
# Build context = repo root with data bundled by jobpack/build.sh (data is not in git).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir --break-system-packages transformers==5.19.0 peft==0.21.2 accelerate bitsandbytes==0.50.2 \
        tiktoken numpy rich huggingface_hub
WORKDIR /work
COPY memlm/ /work/memlm/
COPY jobpack/data/ /work/data/
# Spero tools come from the memlm-kb container on the same docker network; models cache on a volume
ENV KB_ADDR=memlm-kb:7787 HF_HOME=/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
WORKDIR /work/memlm
CMD ["bash"]
