/*
 * analyze_image.c — Compare exhaustive mode selection vs NN prediction
 * for a single image. Shows per-block mode distribution, accuracy, and SSE.
 *
 * Usage: ./analyze_image <image_path>
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "etc2_modes.h"
#include "nn_export.h"

#define NUM_CLASSES 5

static void compute_features(const float *raw48, float *out) {
	float min_c[3]={1,1,1}, max_c[3]={0,0,0}, sum_c[3]={0};
	float sum_l[3]={0}, sum_r[3]={0}, sum_t[3]={0}, sum_b[3]={0};
	float hz=0, vt=0, max_adj=0;
	float min_lum=99, max_lum=-1;
	float min_lum_rgb[3]={0}, max_lum_rgb[3]={0};

	for (int y=0;y<4;y++) for (int x=0;x<4;x++) {
		int idx = (y*4+x)*3;
		float r=raw48[idx], g=raw48[idx+1], b=raw48[idx+2];
		for (int c=0;c<3;c++) {
			float v=raw48[idx+c];
			if (v<min_c[c]) min_c[c]=v; if (v>max_c[c]) max_c[c]=v;
			sum_c[c]+=v;
			if (x<2) sum_l[c]+=v; else sum_r[c]+=v;
			if (y<2) sum_t[c]+=v; else sum_b[c]+=v;
		}
		float lum=(2*r+4*g+b)/7.0f;
		if (lum<min_lum){min_lum=lum; min_lum_rgb[0]=r;min_lum_rgb[1]=g;min_lum_rgb[2]=b;}
		if (lum>max_lum){max_lum=lum; max_lum_rgb[0]=r;max_lum_rgb[1]=g;max_lum_rgb[2]=b;}
		if (x<3) for (int c=0;c<3;c++){float d=raw48[idx+c]-raw48[(y*4+x+1)*3+c]; if(d<0)d=-d; hz+=d; if(d>max_adj)max_adj=d;}
		if (y<3) for (int c=0;c<3;c++){float d=raw48[idx+c]-raw48[((y+1)*4+x)*3+c]; if(d<0)d=-d; vt+=d; if(d>max_adj)max_adj=d;}
	}

	float max_range=0, sub_diff_h=0;
	for (int c=0;c<3;c++){float rng=max_c[c]-min_c[c]; if(rng>max_range)max_range=rng;
		float d=(sum_l[c]-sum_r[c])/8.0f; if(d<0)d=-d; sub_diff_h+=d;}

	float diff_val_v=1, diff_val_h=1;
	float ch_vals[2][3]; /* sum_l/sum_r for flip=0, sum_t/sum_b for flip=1 */
	for (int c=0;c<3;c++){
		int a0=(int)(sum_l[c]/8.0f*255+0.5f), a1=(int)(sum_r[c]/8.0f*255+0.5f);
		int q0=a0*31/255, q1=a1*31/255; if(q0>31)q0=31; if(q1>31)q1=31;
		if(q1-q0<-4||q1-q0>3) diff_val_v=0;
		a0=(int)(sum_t[c]/8.0f*255+0.5f); a1=(int)(sum_b[c]/8.0f*255+0.5f);
		q0=a0*31/255; q1=a1*31/255; if(q0>31)q0=31; if(q1>31)q1=31;
		if(q1-q0<-4||q1-q0>3) diff_val_h=0;
	}

	float mean_var=0;
	for (int c=0;c<3;c++){float m=sum_c[c]/16; float v=0;
		for(int i=0;i<16;i++){float d=raw48[i*3+c]-m; v+=d*d;} mean_var+=v/16;} mean_var/=3;

	float b1[3],b2[3]; float cpd=0;
	for (int c=0;c<3;c++){
		int v1=(int)(min_lum_rgb[c]*255+0.5f+8)>>4; if(v1>15)v1=15;
		int v2=(int)(max_lum_rgb[c]*255+0.5f+8)>>4; if(v2>15)v2=15;
		b1[c]=(float)((v1<<4)|v1)/255.0f; b2[c]=(float)((v2<<4)|v2)/255.0f;
		float d=b1[c]-b2[c]; cpd+=d*d;} cpd=sqrtf(cpd);

	float sum_near=0,max_near=0,wvar1=0,wvar2=0; int cnt1=0,cnt2=0;
	for (int i=0;i<16;i++){
		float r=raw48[i*3],g=raw48[i*3+1],bb=raw48[i*3+2];
		float d1=(r-b1[0])*(r-b1[0])+(g-b1[1])*(g-b1[1])+(bb-b1[2])*(bb-b1[2]);
		float d2=(r-b2[0])*(r-b2[0])+(g-b2[1])*(g-b2[1])+(bb-b2[2])*(bb-b2[2]);
		float near=d1<d2?d1:d2; sum_near+=near; if(near>max_near)max_near=near;
		if(d1<d2){cnt1++;wvar1+=d1;}else{cnt2++;wvar2+=d2;}}
	float clust_tight=sum_near/16;
	float between_d=(b1[0]-b2[0])*(b1[0]-b2[0])+(b1[1]-b2[1])*(b1[1]-b2[1])+(b1[2]-b2[2])*(b1[2]-b2[2]);
	float within_avg=0; if(cnt1>0)within_avg+=wvar1/cnt1; if(cnt2>0)within_avg+=wvar2/cnt2;
	within_avg=within_avg*0.5f+1e-6f;
	float bimodal=between_d/within_avg;
	float total_w=wvar1+wvar2, total_v=mean_var*3*16+1e-6f;
	float var_ratio=total_w/total_v;

	float sub_diff_v=0;
	for(int c=0;c<3;c++){float d=(sum_t[c]-sum_b[c])/8; if(d<0)d=-d; sub_diff_v+=d;}

	float smooth=1-max_adj; if(smooth<0)smooth=0;

	out[0]=max_range; out[1]=sub_diff_h; out[2]=max_adj;
	out[3]=hz/36; out[4]=vt/36;
	out[5]=diff_val_v; out[6]=diff_val_h;
	out[7]=max_c[0]-min_c[0]; out[8]=max_c[1]-min_c[1]; out[9]=max_c[2]-min_c[2];
	out[10]=max_lum-min_lum; out[11]=sub_diff_v; out[12]=mean_var; out[13]=cpd;
	out[14]=clust_tight; out[15]=max_near; out[16]=(float)cnt1/16; out[17]=bimodal;
	out[18]=var_ratio;
	out[19]=clust_tight*cpd; out[20]=max_near*var_ratio; out[21]=max_range*var_ratio;
	out[22]=bimodal*cpd; out[23]=(max_lum-min_lum)*clust_tight;
	out[24]=sub_diff_v*smooth+sub_diff_h*smooth;
}

static int nn_predict(const float feat[25]) {
	/* Standardize */
	float f[25];
	for (int i=0;i<25;i++) f[i]=(feat[i]-nn_feat_mean[i])/nn_feat_std[i];

	/* Layer 1: ReLU */
	float h[16];
	for (int j=0;j<16;j++){
		float s=nn_b1[j];
		for (int i=0;i<25;i++) s+=f[i]*nn_w1[i*16+j];
		h[j]=s>0?s:0;
	}

	/* Layer 2: argmax */
	int best=0; float best_s=nn_b2[0];
	for (int j=0;j<16;j++) best_s+=h[j]*nn_w2[j*5];
	for (int k=1;k<5;k++){
		float s=nn_b2[k];
		for (int j=0;j<16;j++) s+=h[j]*nn_w2[j*5+k];
		if (s>best_s){best_s=s;best=k;}
	}
	return best;
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "Usage: %s <image>\n", argv[0]); return 1; }

	int w, h, ch;
	uint8_t *data = stbi_load(argv[1], &w, &h, &ch, 4); /* load as RGBA */
	if (!data) { fprintf(stderr, "Failed to load %s\n", argv[1]); return 1; }
	printf("Loaded %s: %dx%d (%d channels)\n\n", argv[1], w, h, ch);

	int bx_count = w / 4, by_count = h / 4;
	int total_blocks = bx_count * by_count;

	static const char *names[5] = {"diff", "ind ", "plan", "T   ", "H   "};
	int true_counts[5]={0}, pred_counts[5]={0}, correct[5]={0};
	int confusion[5][5]={{0}};
	double true_sse_sum[5]={0}, pred_sse_sum[5]={0};
	int mispred_count = 0;
	double mispred_sse_extra = 0;

	for (int by=0; by<by_count; by++) {
		for (int bx=0; bx<bx_count; bx++) {
			/* Extract and premultiply */
			uint8_t block[48];
			float raw48[48];
			for (int y=0;y<4;y++) for (int x=0;x<4;x++){
				const uint8_t *src = data + (by*4+y)*w*4 + (bx*4+x)*4;
				uint8_t a = src[3];
				int idx = (y*4+x)*3;
				block[idx+0] = (uint8_t)((src[0]*a+127)/255);
				block[idx+1] = (uint8_t)((src[1]*a+127)/255);
				block[idx+2] = (uint8_t)((src[2]*a+127)/255);
				raw48[idx+0] = block[idx+0]/255.0f;
				raw48[idx+1] = block[idx+1]/255.0f;
				raw48[idx+2] = block[idx+2]/255.0f;
			}

			/* Exhaustive: all 7 modes → 5-class true label */
			int32_t errors[7];
			etc2_evaluate_all(block, errors);
			int32_t merged[5] = {
				errors[0]<errors[1]?errors[0]:errors[1],
				errors[2]<errors[3]?errors[2]:errors[3],
				errors[4], errors[5], errors[6],
			};
			int true_mode = 0;
			for (int m=1;m<5;m++) if(merged[m]<merged[true_mode]) true_mode=m;

			/* NN prediction */
			float feat[25];
			compute_features(raw48, feat);
			int pred = nn_predict(feat);

			/* Stats */
			true_counts[true_mode]++;
			pred_counts[pred]++;
			confusion[true_mode][pred]++;
			if (pred == true_mode) correct[true_mode]++;

			true_sse_sum[true_mode] += merged[true_mode];
			pred_sse_sum[true_mode] += merged[pred];
			if (pred != true_mode) {
				mispred_count++;
				mispred_sse_extra += (merged[pred] - merged[true_mode]);
			}
		}
	}

	stbi_image_free(data);

	/* Report */
	printf("=== Mode Distribution (%d blocks) ===\n\n", total_blocks);
	printf("  %-6s  %7s  %5s  %7s  %5s\n", "Mode", "True", "%", "Pred", "%");
	printf("  %-6s  %7s  %5s  %7s  %5s\n", "------", "-------", "-----", "-------", "-----");
	for (int c=0;c<5;c++)
		printf("  %-6s  %7d  %4.1f%%  %7d  %4.1f%%\n", names[c],
			true_counts[c], (double)true_counts[c]/total_blocks*100,
			pred_counts[c], (double)pred_counts[c]/total_blocks*100);

	printf("\n=== Confusion Matrix ===\n\n");
	printf("          ");
	for (int j=0;j<5;j++) printf(" %5s", names[j]);
	printf("  |  acc\n");
	for (int i=0;i<5;i++){
		printf("  True %s:", names[i]);
		for (int j=0;j<5;j++) printf(" %5d", confusion[i][j]);
		printf("  | %5.1f%%\n", true_counts[i]>0?(double)correct[i]/true_counts[i]*100:0);
	}

	int total_correct = 0;
	for (int c=0;c<5;c++) total_correct += correct[c];
	printf("\n  Overall accuracy: %d/%d = %.1f%%\n", total_correct, total_blocks,
		(double)total_correct/total_blocks*100);

	printf("\n=== SSE Analysis ===\n\n");
	printf("  %-6s  %10s  %10s  %8s\n", "Mode", "Avg OptSSE", "Avg PrdSSE", "Overhead");
	printf("  %-6s  %10s  %10s  %8s\n", "------", "----------", "----------", "--------");
	for (int c=0;c<5;c++){
		if (true_counts[c]==0) continue;
		double avg_opt = true_sse_sum[c]/true_counts[c];
		double avg_prd = pred_sse_sum[c]/true_counts[c];
		printf("  %-6s  %10.1f  %10.1f  %7.1f%%\n", names[c], avg_opt, avg_prd,
			avg_opt>0?(avg_prd-avg_opt)/avg_opt*100:0);
	}

	double total_opt=0, total_prd=0;
	for (int c=0;c<5;c++){total_opt+=true_sse_sum[c]; total_prd+=pred_sse_sum[c];}
	printf("  %-6s  %10.1f  %10.1f  %7.1f%%\n", "TOTAL",
		total_opt/total_blocks, total_prd/total_blocks,
		total_opt>0?(total_prd-total_opt)/total_opt*100:0);
	printf("\n  Mispredictions: %d/%d (%.1f%%), avg extra SSE: %.1f\n",
		mispred_count, total_blocks, (double)mispred_count/total_blocks*100,
		mispred_count>0?mispred_sse_extra/mispred_count:0);

	return 0;
}
