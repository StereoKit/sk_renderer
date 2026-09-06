// SPDX-License-Identifier: MIT
// Dear ImGui sk_renderer Backend Implementation

#include "imgui_impl_sk_renderer.h"
#include "imgui.h"
#include "sk_renderer.h"

#include <stdlib.h>
#include <string.h>

// Generated shader header (compiled by skshaderc to header file)
#include "imgui.hlsl.h"

// Backend data stored in io.BackendRendererUserData
struct ImGui_ImplSkRenderer_Data {
	skr_shader_t       shader;
	skr_material_t     material;
	skr_mesh_t         mesh;
	skr_vert_type_t    vertex_type;

	int32_t mesh_vertex_capacity;  // Vertex capacity (count)
	int32_t mesh_index_capacity;   // Index capacity (count)
};

static ImGui_ImplSkRenderer_Data* ImGui_ImplSkRenderer_GetBackendData() {
	return ImGui::GetCurrentContext() ? (ImGui_ImplSkRenderer_Data*)ImGui::GetIO().BackendRendererUserData : nullptr;
}

// Create vertex format matching ImDrawVert
static bool ImGui_ImplSkRenderer_CreateVertexFormat(skr_vert_type_t* out_vertex_type) {
	// ImDrawVert: ImVec2 pos (8 bytes), ImVec2 uv (8 bytes), ImU32 col (4 bytes) = 20 bytes total
	skr_vert_component_t components[] = {
		{ .format = skr_vertex_fmt_f32,            .count = 2, .semantic = skr_semantic_position, .semantic_slot = 0 },  // pos
		{ .format = skr_vertex_fmt_f32,            .count = 2, .semantic = skr_semantic_texcoord, .semantic_slot = 0 },  // uv
		{ .format = skr_vertex_fmt_ui8_normalized, .count = 4, .semantic = skr_semantic_color,    .semantic_slot = 0 },  // col (RGBA8)
	};

	return skr_vert_type_create(components, 3, out_vertex_type) == skr_err_success;
}

// ImGui owns texture lifetime since 1.92, and queues the work here. Each
// skr_tex_t is heap allocated so its address can serve as the ImTextureID.
static void ImGui_ImplSkRenderer_DestroyTexture(ImTextureData* tex) {
	skr_tex_t* skr_tex = (skr_tex_t*)tex->BackendUserData;
	if (!skr_tex) return;

	// Frames in flight may still reference it
	skr_future_t pending = skr_future_get();
	skr_future_wait(&pending);

	skr_tex_destroy(skr_tex);
	free(skr_tex);

	tex->BackendUserData = nullptr;
	tex->SetTexID(ImTextureID_Invalid);
	tex->SetStatus(ImTextureStatus_Destroyed);
}

static void ImGui_ImplSkRenderer_UpdateTexture(ImTextureData* tex) {
	// The shader samples RGBA; an Alpha8 atlas would need its own variant
	IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);

	if (tex->Status == ImTextureStatus_WantCreate) {
		skr_tex_sampler_t sampler = {
			.sample         = skr_tex_sample_linear,
			.address        = skr_tex_address_clamp,
			.sample_compare = skr_compare_never,
		};

		skr_tex_t* skr_tex = (skr_tex_t*)calloc(1, sizeof(skr_tex_t));
		if (!skr_tex) return;

		skr_tex_data_t data = {.data = tex->GetPixels(), .mip_count = 1, .layer_count = 1};
		if (skr_tex_create(skr_tex_fmt_rgba32_linear, skr_tex_flags_dynamic, sampler,
						   (skr_vec3i_t){tex->Width, tex->Height, 1}, 1, 1, &data, skr_tex) != skr_err_success) {
			free(skr_tex);
			return;
		}
		skr_tex_set_name(skr_tex, "ImGui Atlas");

		tex->BackendUserData = skr_tex;
		tex->SetTexID((ImTextureID)skr_tex);
		tex->SetStatus(ImTextureStatus_OK);
	}
	else if (tex->Status == ImTextureStatus_WantUpdates) {
		// skr uploads whole mips, so tex->Updates' sub-rects don't buy anything here
		skr_tex_t*     skr_tex = (skr_tex_t*)tex->BackendUserData;
		skr_tex_data_t data    = {.data = tex->GetPixels(), .mip_count = 1, .layer_count = 1};
		if (skr_tex_set_data(skr_tex, &data) == skr_err_success)
			tex->SetStatus(ImTextureStatus_OK);
	}
	else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
		ImGui_ImplSkRenderer_DestroyTexture(tex);
	}
}

extern "C" bool ImGui_ImplSkRenderer_Init() {
	ImGuiIO& io = ImGui::GetIO();
	IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized!");

	// Allocate backend data
	ImGui_ImplSkRenderer_Data* bd = (ImGui_ImplSkRenderer_Data*)calloc(1, sizeof(ImGui_ImplSkRenderer_Data));
	if (!bd) return false;

	io.BackendRendererUserData = (void*)bd;
	io.BackendRendererName = "imgui_impl_sk_renderer";
	io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;

	// Create vertex format
	if (!ImGui_ImplSkRenderer_CreateVertexFormat(&bd->vertex_type)) {
		free(bd);
		io.BackendRendererUserData = nullptr;
		return false;
	}

	// Load shader from embedded header
	if (skr_shader_create(sks_imgui_hlsl, sizeof(sks_imgui_hlsl), &bd->shader) != skr_err_success) {
		skr_vert_type_destroy(&bd->vertex_type);
		free(bd);
		io.BackendRendererUserData = nullptr;
		return false;
	}
	skr_shader_set_name(&bd->shader, "ImGui");

	// Create material with alpha blending
	skr_material_create((skr_material_info_t){
		.shader       = &bd->shader,
		.cull         = skr_cull_none,
		.write_mask   = skr_write_default,
		.depth_test   = skr_compare_always,
		.blend_state  = skr_blend_alpha,
		.queue_offset = 100,  // Render last
	}, &bd->material);

	// Mesh will be created dynamically on first frame
	bd->mesh_vertex_capacity = 0;
	bd->mesh_index_capacity = 0;

	return true;
}

// Shutdown backend
extern "C" void ImGui_ImplSkRenderer_Shutdown() {
	ImGui_ImplSkRenderer_Data* bd = ImGui_ImplSkRenderer_GetBackendData();
	if (!bd) return;

	ImGuiIO& io = ImGui::GetIO();

	// Textures shared with another context are that context's to release
	ImVector<ImTextureData*>& textures = ImGui::GetPlatformIO().Textures;
	for (int i = 0; i < textures.Size; i++) {
		if (textures[i]->RefCount == 1)
			ImGui_ImplSkRenderer_DestroyTexture(textures[i]);
	}

	// Destroy resources
	if (bd->mesh_vertex_capacity > 0) {
		skr_mesh_destroy(&bd->mesh);
	}
	skr_material_destroy(&bd->material);
	skr_shader_destroy(&bd->shader);
	skr_vert_type_destroy(&bd->vertex_type);

	// Free backend data
	io.BackendRendererUserData = nullptr;
	io.BackendFlags &= ~ImGuiBackendFlags_RendererHasTextures;
	free(bd);
}

// Per-frame update
extern "C" void ImGui_ImplSkRenderer_NewFrame() {
	ImGui_ImplSkRenderer_Data* bd = ImGui_ImplSkRenderer_GetBackendData();
	IM_ASSERT(bd != nullptr && "Backend not initialized!");
	IM_UNUSED(bd);
}

// Resize mesh if needed
static void ImGui_ImplSkRenderer_ResizeMeshIfNeeded(ImGui_ImplSkRenderer_Data* bd,
													  int32_t required_vtx_count,
													  int32_t required_idx_count) {
	// Check if resize needed
	if (required_vtx_count <= bd->mesh_vertex_capacity && required_idx_count <= bd->mesh_index_capacity) {
		return;  // Current mesh is large enough
	}

	// Destroy old mesh
	if (bd->mesh_vertex_capacity > 0) {
		skr_mesh_destroy(&bd->mesh);
	}

	// Grow by 1.5x of required size
	bd->mesh_vertex_capacity = required_vtx_count + required_vtx_count / 2;
	bd->mesh_index_capacity = required_idx_count + required_idx_count / 2;

	// Determine index type
	skr_index_fmt_ index_type = (sizeof(ImDrawIdx) == 2) ? skr_index_fmt_u16 : skr_index_fmt_u32;

	// Create new mesh with larger capacity (no data yet, will upload per-frame)
	if (skr_mesh_create(&bd->vertex_type, index_type, nullptr, bd->mesh_vertex_capacity,
						nullptr, bd->mesh_index_capacity, &bd->mesh) != skr_err_success) {
		bd->mesh_vertex_capacity = 0;
		bd->mesh_index_capacity = 0;
		return;
	}

	skr_mesh_set_name(&bd->mesh, "ImGui Mesh");
}

// Setup orthographic projection matrix
static void ImGui_ImplSkRenderer_SetupProjection(ImGui_ImplSkRenderer_Data* bd, ImDrawData* draw_data) {
	float L = draw_data->DisplayPos.x;
	float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
	float T = draw_data->DisplayPos.y;
	float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

	// Orthographic projection with Y flipped to match negative viewport height
	float projection[4][4] = {
		{ 2.0f/(R-L),    0.0f,          0.0f, 0.0f },
		{ 0.0f,          2.0f/(T-B),    0.0f, 0.0f },
		{ 0.0f,          0.0f,         -1.0f, 0.0f },
		{ (R+L)/(L-R),   (T+B)/(B-T),   0.0f, 1.0f },
	};

	// Bind projection matrix to material (inline constant buffer data)
	skr_material_set_params(&bd->material, projection, sizeof(projection));
}

// Phase 1: Prepare draw data (upload mesh, MUST be called OUTSIDE render pass)
extern "C" void ImGui_ImplSkRenderer_PrepareDrawData(void) {
	ImDrawData* draw_data = ImGui::GetDrawData();

	// Avoid rendering when minimized
	if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
		return;

	ImGui_ImplSkRenderer_Data* bd = ImGui_ImplSkRenderer_GetBackendData();
	IM_ASSERT(bd != nullptr && "Backend not initialized!");

	// Texture uploads copy through a staging buffer, so they belong out here
	// alongside the mesh upload rather than inside the pass
	if (draw_data->Textures != nullptr) {
		for (int i = 0; i < draw_data->Textures->Size; i++) {
			ImTextureData* tex = (*draw_data->Textures)[i];
			if (tex->Status != ImTextureStatus_OK)
				ImGui_ImplSkRenderer_UpdateTexture(tex);
		}
	}

	if (draw_data->TotalVtxCount == 0 || draw_data->TotalIdxCount == 0)
		return;

	// Resize mesh if needed
	ImGui_ImplSkRenderer_ResizeMeshIfNeeded(bd, draw_data->TotalVtxCount, draw_data->TotalIdxCount);

	if (bd->mesh_vertex_capacity == 0) {
		return;  // Mesh creation failed
	}

	// Upload all vertex and index data to mesh buffers
	int32_t global_vtx_offset = 0;
	int32_t global_idx_offset = 0;

	for (int n = 0; n < draw_data->CmdListsCount; n++) {
		const ImDrawList* cmd_list = draw_data->CmdLists[n];
		global_vtx_offset += cmd_list->VtxBuffer.Size;
		global_idx_offset += cmd_list->IdxBuffer.Size;
	}

	// Now upload all data in one go by copying to temporary buffers
	ImDrawVert* all_vertices = (ImDrawVert*)malloc(sizeof(ImDrawVert) * draw_data->TotalVtxCount);
	ImDrawIdx*  all_indices  = (ImDrawIdx*) malloc(sizeof(ImDrawIdx) * draw_data->TotalIdxCount);

	global_vtx_offset = 0;
	global_idx_offset = 0;
	for (int n = 0; n < draw_data->CmdListsCount; n++) {
		const ImDrawList* cmd_list = draw_data->CmdLists[n];
		memcpy(&all_vertices[global_vtx_offset], cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(&all_indices[global_idx_offset],  cmd_list->IdxBuffer.Data,  cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
		global_vtx_offset += cmd_list->VtxBuffer.Size;
		global_idx_offset += cmd_list->IdxBuffer.Size;
	}

	// Upload to mesh (this does vkCmdCopyBuffer, so MUST be outside render pass)
	skr_mesh_set_data(&bd->mesh, all_vertices, draw_data->TotalVtxCount, all_indices, draw_data->TotalIdxCount);

	free(all_vertices);
	free(all_indices);

	// Setup projection matrix
	ImGui_ImplSkRenderer_SetupProjection(bd, draw_data);
}

// Phase 2: Render draw data (draw ImGui, MUST be called INSIDE render pass)
extern "C" void ImGui_ImplSkRenderer_RenderDrawData(int width, int height) {
	ImDrawData* draw_data = ImGui::GetDrawData();
	
	// Avoid rendering when minimized
	if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
		return;

	ImGui_ImplSkRenderer_Data* bd = ImGui_ImplSkRenderer_GetBackendData();
	IM_ASSERT(bd != nullptr && "Backend not initialized!");

	if (draw_data->TotalVtxCount == 0 || draw_data->TotalIdxCount == 0)
		return;

	// NOTE: PrepareDrawData must have been called before this!
	// We're now INSIDE a render pass, just drawing.

	// Set viewport
	skr_renderer_set_viewport((skr_rect_t){0, 0, (float)width, (float)height});

	// Will project scissor/clipping rectangles into framebuffer space
	ImVec2 clip_off = draw_data->DisplayPos;          // (0,0) unless using multi-viewports
	ImVec2 clip_scale = draw_data->FramebufferScale;  // (1,1) unless using retina display

	// Draw all command lists with immediate mode (allows per-command scissor rects)
	int32_t global_vtx_offset = 0;
	int32_t global_idx_offset = 0;

	for (int n = 0; n < draw_data->CmdListsCount; n++) {
		const ImDrawList* cmd_list = draw_data->CmdLists[n];

		for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
			const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];

			if (pcmd->UserCallback != nullptr) {
				// User callback
				if (pcmd->UserCallback == ImDrawCallback_ResetRenderState) {
					// Reset render state (nothing to do)
				} else {
					pcmd->UserCallback(cmd_list, pcmd);
				}
			} else {
				// Project scissor/clipping rectangles into framebuffer space
				ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x, (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
				ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x, (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

				// Clamp to viewport
				if (clip_min.x < 0.0f) clip_min.x = 0.0f;
				if (clip_min.y < 0.0f) clip_min.y = 0.0f;
				if (clip_max.x > (float)width) clip_max.x = (float)width;
				if (clip_max.y > (float)height) clip_max.y = (float)height;
				if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
					continue;

				// Apply scissor rectangle
				skr_recti_t scissor = {
					.x = (int32_t)clip_min.x,
					.y = (int32_t)clip_min.y,
					.w = (int32_t)(clip_max.x - clip_min.x),
					.h = (int32_t)(clip_max.y - clip_min.y),
				};
				skr_renderer_set_scissor(scissor);

				// Bind texture (update material)
				skr_tex_t* texture = (skr_tex_t*)pcmd->GetTexID();
				skr_material_set_tex(&bd->material, "texture0", texture);

				// Draw immediately with per-command scissor rect
				skr_renderer_draw_mesh_immediate(
					&bd->mesh,
					&bd->material,
					global_idx_offset + pcmd->IdxOffset,  // first_index
					pcmd->ElemCount,                       // index_count
					global_vtx_offset + pcmd->VtxOffset,   // vertex_offset
					1                                      // instance_count
				);
			}
		}

		global_idx_offset += cmd_list->IdxBuffer.Size;
		global_vtx_offset += cmd_list->VtxBuffer.Size;
	}

	// NOTE: No end_pass - we're drawing in the same pass as the scene!
}
