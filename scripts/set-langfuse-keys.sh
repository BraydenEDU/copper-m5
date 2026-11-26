#!/bin/sh
# Adds your Langfuse keys to .env.local without showing them on screen.
# Get them in Langfuse: your project > Settings > API Keys > Create new API keys.
cd "$(dirname "$0")/.." || exit 1
touch .env.local
printf "Langfuse Public Key (pk-lf-...): "
read -r PK
printf "Langfuse Secret Key (sk-lf-..., hidden): "
stty -echo; read -r SK; stty echo; echo
printf "Langfuse Host [https://us.cloud.langfuse.com]: "
read -r URL
URL=${URL:-https://us.cloud.langfuse.com}
# Drop any old Langfuse lines, then append the new ones.
grep -v '^LANGFUSE_' .env.local > .env.local.tmp || true
mv .env.local.tmp .env.local
[ -n "$(tail -c1 .env.local)" ] && echo >> .env.local
printf 'LANGFUSE_PUBLIC_KEY=%s\nLANGFUSE_SECRET_KEY=%s\nLANGFUSE_BASE_URL=%s\n' "$PK" "$SK" "$URL" >> .env.local
unset PK SK
echo "Saved your Langfuse keys to .env.local."
