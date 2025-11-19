# Existing lines from your Dockerfile...
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Use conda to manage main env, but install missing packages via pip pinned at the specific versions
RUN pip install --upgrade pip
RUN conda install -y ipykernel notebook gcc \
 && pip install bnlearn==0.12.0 pgmpy==0.1.25 gpytorch==1.14 \
 && pip install unsloth==2025.10.8 seaborn==0.13.2 bert-score==0.3.13 dotenv==0.9.9 \
 && conda clean -afy


# Install build-essential package which includes GCC and other tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user "dev" with home directory
RUN useradd -m -s /bin/bash dev \
    && apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to our dev user
USER dev
CMD ["bash"]
