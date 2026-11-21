#!/bin/sh
# Add TorchCor Lighting to the application menu of the current user.
#
#   ./install.sh            run from the extracted release folder
#   ./install.sh --remove   take the menu entry away again
#
# Nothing is copied: the menu entry points at this folder, so keep the folder where
# it is (or run the script again after moving it). No root access needed.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$APPS/torchcor-lighting.desktop"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$ENTRY"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
    echo "Removed $ENTRY"
    exit 0
fi

if [ ! -x "$HERE/torchcor-lighting" ]; then
    echo "error: $HERE/torchcor-lighting not found; run this script from the extracted folder" >&2
    exit 1
fi

# Escape the folder for a sed replacement (|, & and \) and quote it for Exec=.
esc=$(printf '%s' "$HERE" | sed -e 's/[|&\\]/\\&/g')
mkdir -p "$APPS"
sed -e "s|^Exec=.*|Exec=\"$esc/torchcor-lighting\"|" \
    -e "s|^TryExec=.*|TryExec=$esc/torchcor-lighting|" \
    -e "s|^Icon=.*|Icon=$esc/torchcor-lighting.png|" \
    "$HERE/torchcor-lighting.desktop" > "$ENTRY"
chmod 644 "$ENTRY"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true

echo "Installed $ENTRY"
echo "TorchCor Lighting now appears in your application menu (log out and back in if it does not)."
