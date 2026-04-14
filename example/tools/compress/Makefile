CC      = gcc
CFLAGS  = -O2 -std=c11 -Wall -Wextra -Wno-unused-function
LDFLAGS = -lm

STB_DIR    = ../../../build/_deps/stb-src
IMAGE_DIR  = test_images
SHADER_DIR = ../../assets/shaders

all: generate_training_data train analyze_image

generate_training_data: generate_training_data.c etc2_modes.h
	$(CC) $(CFLAGS) -I$(STB_DIR) -o $@ generate_training_data.c $(LDFLAGS)

train: train.c etc2_modes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

analyze_image: analyze_image.c etc2_modes.h nn_export.h
	$(CC) $(CFLAGS) -I$(STB_DIR) -o $@ analyze_image.c $(LDFLAGS)

# Full pipeline: generate data → train → export weights to shader
pipeline: generate_training_data train
	./generate_training_data $(IMAGE_DIR) training_data.bin
	./train training_data.bin --max-samples 500000 --epochs 50 --derived-only

# Analyze a specific image against the trained model
analyze: analyze_image
	./analyze_image $(IMAGE)

clean:
	rm -f generate_training_data train analyze_image
	rm -f decision_tree_export.h linear_export.h nn_export.h
	rm -f *.bin

.PHONY: all clean pipeline analyze
