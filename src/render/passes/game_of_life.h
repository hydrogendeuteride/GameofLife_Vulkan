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

    // Called from game update; accumulates dt and queues steps.
    void update(float dt_sec);

    void request_randomize(float fill, uint32_t seed);
    void request_clear();
    void request_step(uint32_t steps = 1);

    VkExtent2D extent() const { return _extent; }
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
    void ensure_images();
    void ensure_imgui_textures();

    EngineContext *_context = nullptr;

    VkExtent2D _extent{4096, 4096};

    bool _enabled = false;
    bool _paused = false;
    bool _wrap = true;

    float _simHz = 30.0f;
    float _accum = 0.0f;
    uint32_t _queuedSteps = 0;

    PendingInit _pendingInit = PendingInit::None;
    float _pendingFill = 0.15f;
    uint32_t _pendingSeed = 1;

    AllocatedImage _state[2]{};
    VkImageLayout _layout[2]{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    VkPipelineStageFlags2 _stage[2]{VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_NONE};
    VkAccessFlags2 _access[2]{0, 0};
    uint32_t _current = 0;

    void *_imguiTex[2]{nullptr, nullptr};

    uint64_t _generation = 0;

    bool _instancesCreated = false;
};
