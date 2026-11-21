# Judgment-eval "server": a Debian box with simulated services (agent/judge). Build context: agent/judge
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps iproute2 lsof git psmisc less vim-tiny \
    && rm -rf /var/lib/apt/lists/* \
    && for u in alice bob deploy dev; do useradd -m -s /bin/bash $u; done \
    && useradd -r -s /usr/sbin/nologin postgres && useradd -r -s /usr/sbin/nologin backup-agent 2>/dev/null; \
    git config --system user.name deploy && git config --system user.email deploy@web01.local \
    && git config --system init.defaultBranch main && git config --system --add safe.directory '*' \
    && mkdir -p /usr/local/lib/opsd /etc/opsd/units /var/lib/opsd /etc/ssh
COPY simd lib.sh /usr/local/lib/opsd/
COPY bin/ /usr/local/bin/
RUN chmod 755 /usr/local/lib/opsd/simd /usr/local/bin/*
