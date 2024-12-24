FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/app \
  VIRTUAL_ENV=/app/.venv \
  PATH="/app/.venv/bin:/root/.local/bin:$PATH" \
  TZ=Asia/Seoul

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  tzdata \
  nano \
  zsh \
  htop \
  tree \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  # Set timezone
  && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
  && echo $TZ > /etc/timezone

# Install uv and create virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && . $HOME/.profile \
  && uv venv /app/.venv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages using uv
RUN . /app/.venv/bin/activate \
  && uv pip install -r requirements.txt

# Install ZSH configuration
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
  && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
  && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
  && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Copy ZSH configuration files
COPY ./zsh/zshrc /root/.zshrc
COPY ./zsh/p10k.zsh /root/.p10k.zsh
COPY ./zsh/aliases.zsh /root/.aliases.zsh

# Create project directories
RUN mkdir -p /app/data /app/src /app/runs /app/experiments

# Set default shell to ZSH
SHELL ["/bin/zsh", "-c"]

# Add volume for data
VOLUME ["/app/data"]