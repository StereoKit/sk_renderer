/*
 * generate_training_data.c — ETC2 mode classifier training data generator.
 * Loads images, evaluates all 7 ETC2 modes for each 4x4 block, writes binary.
 *
 * Usage: ./generate_training_data <image_folder> <output.bin>
 *
 * Output format:
 *   Header: uint32 magic=0x45544332, uint32 block_count
 *   Per block: 48 bytes (16 RGB pixels row-major) + 1 byte (winning mode 0-6)
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "etc2_modes.h"

/* ------------------------------------------------------------------ */
/*  Recursive image finder                                             */
/* ------------------------------------------------------------------ */

static int _has_image_ext(const char *name) {
	const char *dot = strrchr(name, '.');
	if (!dot) return 0;
	dot++;
	char lower[8];
	int32_t i;
	for (i = 0; dot[i] && i < 7; i++)
		lower[i] = (dot[i] >= 'A' && dot[i] <= 'Z') ? (char)(dot[i] + 32) : dot[i];
	lower[i] = 0;
	return strcmp(lower, "png")  == 0 || strcmp(lower, "jpg") == 0 ||
	       strcmp(lower, "jpeg") == 0 || strcmp(lower, "bmp") == 0 ||
	       strcmp(lower, "tga")  == 0;
}

typedef struct {
	char   **paths;
	int32_t  count;
	int32_t  capacity;
} file_list_t;

static void _add_path(file_list_t *list, const char *path) {
	if (list->count >= list->capacity) {
		list->capacity = list->capacity ? list->capacity * 2 : 256;
		list->paths = realloc(list->paths, (size_t)list->capacity * sizeof(char *));
	}
	list->paths[list->count++] = strdup(path);
}

static void _find_images(const char *dir, file_list_t *list) {
	DIR *d = opendir(dir);
	if (!d) return;
	struct dirent *entry;
	while ((entry = readdir(d)) != NULL) {
		if (entry->d_name[0] == '.') continue;
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
		struct stat st;
		if (stat(path, &st) != 0) continue;
		if (S_ISDIR(st.st_mode)) {
			_find_images(path, list);
		} else if (S_ISREG(st.st_mode) && _has_image_ext(entry->d_name)) {
			_add_path(list, path);
		}
	}
	closedir(d);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <image_folder> <output.bin>\n", argv[0]);
		return 1;
	}

	const char *image_folder = argv[1];
	const char *output_path  = argv[2];

	/* Find images */
	file_list_t images = {0};
	_find_images(image_folder, &images);
	if (images.count == 0) {
		fprintf(stderr, "No images found in %s\n", image_folder);
		return 1;
	}
	printf("Found %d images\n", images.count);

	/* Open output */
	FILE *out = fopen(output_path, "wb");
	if (!out) {
		fprintf(stderr, "Cannot open %s for writing\n", output_path);
		return 1;
	}

	/* Write header with placeholder count */
	uint32_t magic = 0x45544333; /* "ETC3" — v2 format with label2 */
	uint32_t block_count = 0;
	fwrite(&magic, 4, 1, out);
	fwrite(&block_count, 4, 1, out);

	int32_t mode_dist[5]  = {0};
	int32_t images_loaded  = 0;

	for (int32_t img_i = 0; img_i < images.count; img_i++) {
		int w, h, channels;
		uint8_t *data = stbi_load(images.paths[img_i], &w, &h, &channels, 3);
		if (!data) {
			fprintf(stderr, "Warning: failed to load %s\n", images.paths[img_i]);
			continue;
		}
		images_loaded++;
		int32_t stride = w * 3;

		/* Process all complete 4x4 blocks */
		int32_t bx_count = w / 4;
		int32_t by_count = h / 4;

		for (int32_t by = 0; by < by_count; by++) {
			for (int32_t bx = 0; bx < bx_count; bx++) {
				/* Extract 4x4 block as 48 bytes of RGB */
				uint8_t block[48];
				for (int32_t y = 0; y < 4; y++) {
					const uint8_t *row = data + (by * 4 + y) * stride + bx * 4 * 3;
					memcpy(block + y * 12, row, 12);
				}

				/* Evaluate all 7 modes, merge flip variants:
				 *   0: differential (best of flip=0,1)
				 *   1: individual   (best of flip=0,1)
				 *   2: planar
				 *   3: T-mode
				 *   4: H-mode
				 */
				int32_t errors[7];
				etc2_evaluate_all(block, errors);

				int32_t merged[5] = {
					errors[0] < errors[1] ? errors[0] : errors[1], /* diff */
					errors[2] < errors[3] ? errors[2] : errors[3], /* ind */
					errors[4],                                      /* planar */
					errors[5],                                      /* T */
					errors[6],                                      /* H */
				};

				int32_t best_mode = 0;
				for (int32_t m = 1; m < 5; m++)
					if (merged[m] < merged[best_mode]) best_mode = m;

				/* Find 2nd-best within 2.5% margin */
				int32_t best_err = merged[best_mode];
				int32_t second_mode = -1;
				int32_t second_err  = INT32_MAX;
				for (int32_t m = 0; m < 5; m++) {
					if (m == best_mode) continue;
					if (merged[m] < second_err) {
						second_err = merged[m];
						second_mode = m;
					}
				}
				uint8_t label2 = 0xFF;
				if (best_err == 0) {
					if (second_err == 0) label2 = (uint8_t)second_mode;
				} else if ((float)(second_err - best_err) / (float)best_err <= 0.025f) {
					label2 = (uint8_t)second_mode;
				}

				/* Write block + label + label2 (50 bytes per block) */
				fwrite(block, 48, 1, out);
				uint8_t label = (uint8_t)best_mode;
				fwrite(&label, 1, 1, out);
				fwrite(&label2, 1, 1, out);

				mode_dist[best_mode]++;
				block_count++;

				if (block_count % 10000 == 0) {
					printf("Processed %u blocks from %d images, dist: [diff=%d ind=%d pln=%d T=%d H=%d]\n",
						block_count, images_loaded,
						mode_dist[0], mode_dist[1], mode_dist[2],
						mode_dist[3], mode_dist[4]);
				}
			}
		}

		stbi_image_free(data);
	}

	/* Update header with actual count */
	fseek(out, 4, SEEK_SET);
	fwrite(&block_count, 4, 1, out);
	fclose(out);

	printf("\nDone! Wrote %u blocks from %d images to %s\n", block_count, images_loaded, output_path);
	printf("Final: [diff=%d ind=%d pln=%d T=%d H=%d]\n",
		mode_dist[0], mode_dist[1], mode_dist[2], mode_dist[3], mode_dist[4]);

	/* Cleanup */
	for (int32_t i = 0; i < images.count; i++)
		free(images.paths[i]);
	free(images.paths);

	return 0;
}
