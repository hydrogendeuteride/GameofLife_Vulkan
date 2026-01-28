#pragma once

#include "render/renderpass.h"

#include <render/graph/types.h>

#include <cstdint>

class RenderGraph;
class EngineContext;

class GameOfLifePass final : public IRenderPass
{
public:
    void init(EngineContext *context) override;
    void cleanup() override;
    void execute(VkCommandBuffer cmd) override;
    const char *getName() const override { return "GameOfLife"; }

    void register_graph(RenderGraph *graph);

    // --- UI / gameplay controls ---
    void set_enabled(bool enabled);
    bool enabled() const { return _enabled; }

    void set_paused(bool paused);
    bool paused() const { return _paused; }

    void set_wrap(bool wrap);
    bool wrap() const { return _wrap; }

    void set_simulation_hz(float hz);
    float simulation_hz() const { return _simHz; }

    void set_preview_enabled(bool enabled) { _previewEnabled = enabled; }
    bool preview_enabled() const { return _previewEnabled; }

    // Called from game update; accumulates dt and queues steps.
    void update(float dt_sec);

    void request_randomize(float fill, uint32_t seed);
    void request_clear();
    void request_step(uint32_t steps = 1);

    // Simulation grid size in cells. Note: width must be a multiple of 32 for 1-bit packing.
    VkExtent2D extent() const { return _simExtent; }
    // Preview texture size used for ImGui display (downsampled).
    VkExtent2D preview_extent() const { return _previewExtent; }
    uint64_t generation() const { return _generation; }

    // Returns ImTextureID (VkDescriptorSet) for ImGui::Image().
    void *imgui_texture_id();

private:
    enum class PendingInit
    {
        None,
        Random,
        Clear,
    };

    void ensure_pipelines();
    void ensure_buffers();
    void ensure_preview_image();
    void ensure_imgui_textures();

    EngineContext *_context = nullptr;

    VkExtent2D _simExtent{64, 64};
    VkExtent2D _previewExtent{2048, 2048};

    bool _enabled = false;
    bool _paused = false;
    bool _wrap = true;

    float _simHz = 30.0f;
    float _accum = 0.0f;
    uint32_t _queuedSteps = 0;

    PendingInit _pendingInit = PendingInit::None;
    float _pendingFill = 0.15f;
    uint32_t _pendingSeed = 1;

    uint32_t _wordsPerRow = 0;
    VkDeviceSize _bitsSizeBytes = 0;

    AllocatedBuffer _bits[2]{};
    VkPipelineStageFlags2 _bufStage[2]{VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_NONE};
    VkAccessFlags2 _bufAccess[2]{0, 0};
    uint32_t _current = 0;

    AllocatedImage _preview{};
    VkImageLayout _previewLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 _previewStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 _previewAccess = 0;

    void *_imguiPreviewTex = nullptr;
    bool _previewEnabled = true;

    uint64_t _generation = 0;

    bool _instancesCreated = false;
};
