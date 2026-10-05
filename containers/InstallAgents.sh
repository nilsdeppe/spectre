#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Install coding agents (Claude Code, Codex, Copilot) and the tools they use.
# Used by the `agents-mpich` and `agents-openmpi` stages of
# `containers/Dockerfile.buildenv`.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Bubblewrap is the sandbox backend of Codex on Linux
apt-get update -y
apt-get install -y --no-install-recommends bubblewrap

# Coding agents
curl -fsSL https://deb.nodesource.com/setup_24.x | bash -
apt-get install -y --no-install-recommends nodejs
apt-get -y clean
npm install -g @anthropic-ai/claude-code @openai/codex @github/copilot

# Pyright helps coding agents understand Python code better
python -m pip --no-cache-dir install pyright

# A wrapper that installs the clangd and pyright plugins for Claude Code before
# launching it
cat > /usr/local/bin/claude <<'EOF'
#!/bin/bash
if ! /usr/bin/claude plugin list 2>/dev/null | grep -q "clangd-lsp"; then
    /usr/bin/claude plugin marketplace update claude-plugins-official || true
    /usr/bin/claude plugin install clangd-lsp@claude-plugins-official || true
fi
if ! /usr/bin/claude plugin list 2>/dev/null | grep -q "pyright"; then
    /usr/bin/claude plugin marketplace update claude-plugins-official || true
    /usr/bin/claude plugin install pyright-lsp@claude-plugins-official || true
fi
exec env ENABLE_LSP_TOOL=1 /usr/bin/claude "$@"
EOF
chmod 755 /usr/local/bin/claude
