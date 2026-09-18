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
	for c in astcenc-avx2 astcenc-sse4.1 astcenc-sse2 astcenc; do
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

# Auto-detect HDR vs LDR from the first block's CEM. .astc layout is a 16-byte
# header followed by 16-byte blocks; CEM lives at block bits 13-16, which for
# our single-partition single-CEM encoder is:
#   bits[13:15] = byte[1] >> 5
#   bit  16     = byte[2] & 1
# HDR CEMs are {2, 3, 7, 11, 14, 15}; the encoder we ship here only emits
# CEM 11, so we just check for that exact value.
B1=$(od -An -N1 -j17 -tu1 "$ASTC_FILE" | tr -d ' \n')
B2=$(od -An -N1 -j18 -tu1 "$ASTC_FILE" | tr -d ' \n')
CEM=$(( ((B1 >> 5) & 7) | ((B2 & 1) << 3) ))
IS_HDR=0
case "$CEM" in 2|3|7|11|14|15) IS_HDR=1 ;; esac

ORIG_DIR="$(dirname  "$ORIGINAL")"
ORIG_STEM="$(basename "$ORIGINAL")"
ORIG_STEM="${ORIG_STEM%.*}"

if [[ "$IS_HDR" == 1 ]]; then
	# HDR decode → Radiance .hdr next to the original. Skip the LDR-only
	# astc_validate PSNR/SSIM run (it uses stb_image, doesn't load .hdr) —
	# eyeballing the round-tripped HDR file is the test for v1.
	DECODED_HDR="$ORIG_DIR/${ORIG_STEM}_decoded.hdr"
	echo "--- HDR decode (CEM $CEM): $ASTC_FILE → $DECODED_HDR"
	"$ASTCENC" -dh "$ASTC_FILE" "$DECODED_HDR" >/dev/null
	echo "decoded image: $DECODED_HDR"
else
	# LDR decode → PNG, plus the validate-binary comparison.
	DECODED_PNG="$ORIG_DIR/${ORIG_STEM}_decoded.png"
	echo "--- LDR decode (CEM $CEM): $ASTC_FILE → $DECODED_PNG"
	"$ASTCENC" -dl "$ASTC_FILE" "$DECODED_PNG" >/dev/null

	echo
	"$VALIDATE_BIN" "$ORIGINAL" "$ASTC_FILE" --reference medium --astcenc "$ASTCENC"
	echo "decoded image: $DECODED_PNG"
fi
