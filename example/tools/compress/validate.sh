#!/usr/bin/env bash
# ASTC validation one-shot.
#
# Usage:
#   validate.sh <original_image> [astc_output_file]
#
# Defaults to ./astc_output.astc (what scene_tex_compress auto-saves). Decodes
# it via astcenc, compares PSNR/SSIM to the original, and writes the decoded
# RGBA PNG to <original_basename>_decoded.png next to the original.
#
# Requires: astc_validate (built from example/tools/compress/validate/) and
# astcenc on $PATH or at $ASTCENC_BIN.
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "usage: $0 <original_image> [astc_output_file]" >&2
	exit 1
fi

ORIGINAL="$1"
ASTC_FILE="${2:-astc_output.astc}"

if [[ ! -f "$ORIGINAL"  ]]; then echo "original not found: $ORIGINAL"  >&2; exit 1; fi
if [[ ! -f "$ASTC_FILE" ]]; then echo "astc not found: $ASTC_FILE"     >&2; exit 1; fi

# Locate astc_validate — prefer one built into the main project's build/.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VALIDATE_BIN=""
for candidate in \
	"$SCRIPT_DIR/../../../build/example/tools/compress/validate/astc_validate" \
	"$(command -v astc_validate || true)"; do
	if [[ -n "$candidate" && -x "$candidate" ]]; then
		VALIDATE_BIN="$candidate"
		break
	fi
done
if [[ -z "$VALIDATE_BIN" ]]; then
	echo "astc_validate not found — build the 'astc_validate' cmake target first" >&2
	exit 1
fi

# Locate astcenc (validate.sh also shells out to it for the final decoded PNG).
ASTCENC="${ASTCENC_BIN:-}"
if [[ -z "$ASTCENC" ]]; then
	for c in astcenc-avx2 astcenc-sse4.1 astcenc-sse2 astcenc \
		/home/koujaku/Apps/astcenc/astcenc-avx2 \
		/home/koujaku/Apps/astcenc/astcenc-sse4.1 \
		/home/koujaku/Apps/astcenc/astcenc-sse2; do
		if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then
			ASTCENC="$c"
			break
		fi
	done
fi
if [[ -z "$ASTCENC" ]]; then
	echo "astcenc not found — set ASTCENC_BIN or install to PATH" >&2
	exit 1
fi

# Decoded PNG lands next to the original: <stem>_decoded.png.
ORIG_DIR="$(dirname  "$ORIGINAL")"
ORIG_STEM="$(basename "$ORIGINAL")"
ORIG_STEM="${ORIG_STEM%.*}"
DECODED_PNG="$ORIG_DIR/${ORIG_STEM}_decoded.png"

echo "--- decoding $ASTC_FILE → $DECODED_PNG"
"$ASTCENC" -dl "$ASTC_FILE" "$DECODED_PNG" >/dev/null

echo
"$VALIDATE_BIN" "$ORIGINAL" "$ASTC_FILE" --reference medium --astcenc "$ASTCENC"
echo "decoded image: $DECODED_PNG"
