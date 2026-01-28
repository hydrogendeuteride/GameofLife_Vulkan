#include "game_of_life.h"

#include <core/assets/manager.h>
#include <core/context.h>
#include <core/device/resource.h>
#include <core/pipeline/manager.h>
#include <core/pipeline/sampler.h>

#include <render/graph/graph.h>

#include "imgui_impl_vulkan.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace
{
    constexpr uint32_t kLocalSize = 16;
    constexpr uint32_t kMaxStepsPerFrame = 8;

    struct StepPush
    {
        uint32_t width;
        uint32_t height;
        uint32_t wrap;
        uint32_t pad;
    };

    struct InitPush
    {
        uint32_t width;
        uint32_t height;
        float fill;
        uint32_t seed;
    };

    static VkPipelineStageFlags2 default_stage_for_layout(VkImageLayout layout)
    {
        switch (layout)
        {
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            default:
                return VK_PIPELINE_STAGE_2_NONE;
        }
    }

    static VkAccessFlags2 default_access_for_layout(VkImageLayout layout)
    {
        switch (layout)
        {
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            default:
                return 0;
        }
    }
} // namespace

void GameOfLifePass::init(EngineContext *context)
{
    _context = context;
    ensure_pipelines();
}

void GameOfLifePass::cleanup()
{
    // Note: VulkanEngine currently destroys ImGuiSystem before RenderPassManager,
    // so calling ImGui_ImplVulkan_RemoveTexture() here may run after shutdown.
    // Descriptor sets are allocated from ImGui's descriptor pool and will be
    // released when that pool is destroyed.
    _imguiTex[0] = nullptr;
    _imguiTex[1] = nullptr;

    if (_context && _context->getResources())
    {
        for (auto &img: _state)
        {
            if (img.image != VK_NULL_HANDLE)
            {
                _context->getResources()->destroy_image(img);
            }
            img = {};
        }
    }

    if (_context && _context->pipelines)
    {
        _context->pipelines->destroyComputeInstance("game_of_life.init");
        _context->pipelines->destroyComputePipeline("game_of_life.init");
        _context->pipelines->destroyComputeInstance("game_of_life.step");
        _context->pipelines->destroyComputePipeline("game_of_life.step");
    }

    _instancesCreated = false;
    _context = nullptr;
}

void GameOfLifePass::execute(VkCommandBuffer)
{
    // Executed via RenderGraph.
}

void GameOfLifePass::set_enabled(bool enabled)
{
    _enabled = enabled;
    if (_enabled)
    {
        // If images haven't been created yet, request an init so the content is valid.
        if (_state[0].image == VK_NULL_HANDLE || _state[1].image == VK_NULL_HANDLE)
        {
            _pendingInit = PendingInit::Random;
            _pendingFill = std::clamp(_pendingFill, 0.0f, 1.0f);
        }
    }
}

void GameOfLifePass::set_paused(bool paused)
{
    _paused = paused;
}

void GameOfLifePass::set_wrap(bool wrap)
{
    _wrap = wrap;
}

void GameOfLifePass::set_simulation_hz(float hz)
{
    if (!std::isfinite(hz)) return;
    _simHz = std::clamp(hz, 0.0f, 240.0f);
}

void GameOfLifePass::update(float dt_sec)
{
    if (!_enabled || _paused || _simHz <= 0.0f) return;
    if (!std::isfinite(dt_sec) || dt_sec <= 0.0f) return;

    _accum += dt_sec;
    const float step_dt = 1.0f / _simHz;

    uint32_t steps = 0;
    while (_accum >= step_dt && steps < kMaxStepsPerFrame)
    {
        _accum -= step_dt;
        ++steps;
    }

    if (steps)
    {
        _queuedSteps = std::min<uint32_t>(_queuedSteps + steps, kMaxStepsPerFrame);
    }
}

void GameOfLifePass::request_randomize(float fill, uint32_t seed)
{
    _pendingInit = PendingInit::Random;
    _pendingFill = std::clamp(fill, 0.0f, 1.0f);
    _pendingSeed = (seed == 0) ? 1u : seed;
    _accum = 0.0f;
    _queuedSteps = 0;
    _generation = 0;
    _current = 0;
}

void GameOfLifePass::request_clear()
{
    _pendingInit = PendingInit::Clear;
    _pendingFill = 0.0f;
    _pendingSeed = 1;
    _accum = 0.0f;
    _queuedSteps = 0;
    _generation = 0;
    _current = 0;
}

void GameOfLifePass::request_step(uint32_t steps)
{
    if (!_enabled) return;
    if (steps == 0) return;
    _queuedSteps = std::min<uint32_t>(_queuedSteps + steps, kMaxStepsPerFrame);
}

void *GameOfLifePass::imgui_texture_id()
{
    if (!_enabled) return nullptr;
    ensure_images();
    ensure_imgui_textures();
    return _imguiTex[_current];
}

void GameOfLifePass::ensure_pipelines()
{
    if (!_context || !_context->pipelines || !_context->getAssets()) return;

    if (!_context->pipelines->hasComputePipeline("game_of_life.init"))
    {
        ComputePipelineCreateInfo ci{};
        ci.shaderPath = _context->getAssets()->shaderPath("game_of_life_init.comp.spv");
        ci.descriptorTypes = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        ci.pushConstantSize = sizeof(InitPush);
        _context->pipelines->createComputePipeline("game_of_life.init", ci);
    }

    if (!_context->pipelines->hasComputePipeline("game_of_life.step"))
    {
        ComputePipelineCreateInfo ci{};
        ci.shaderPath = _context->getAssets()->shaderPath("game_of_life.comp.spv");
        ci.descriptorTypes = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        ci.pushConstantSize = sizeof(StepPush);
        _context->pipelines->createComputePipeline("game_of_life.step", ci);
    }

    if (!_instancesCreated)
    {
        _context->pipelines->createComputeInstance("game_of_life.init", "game_of_life.init");
        _context->pipelines->createComputeInstance("game_of_life.step", "game_of_life.step");
        _instancesCreated = true;
    }
}

void GameOfLifePass::ensure_images()
{
    if (!_context || !_context->getResources()) return;
    if (_state[0].image != VK_NULL_HANDLE && _state[1].image != VK_NULL_HANDLE) return;

    VkExtent3D extent3d{_extent.width, _extent.height, 1};
    constexpr VkFormat fmt = VK_FORMAT_R8_UNORM;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    for (int i = 0; i < 2; ++i)
    {
        if (_state[i].image == VK_NULL_HANDLE)
        {
            _state[i] = _context->getResources()->create_image(extent3d, fmt, usage, false);
            _layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
            _stage[i] = VK_PIPELINE_STAGE_2_NONE;
            _access[i] = 0;
        }
    }
}

void GameOfLifePass::ensure_imgui_textures()
{
    if (_imguiTex[0] && _imguiTex[1]) return;
    if (!_context || !_context->getSamplers()) return;
    if (_state[0].imageView == VK_NULL_HANDLE || _state[1].imageView == VK_NULL_HANDLE) return;

    const VkSampler samp = _context->getSamplers()->nearestClampEdge();
    if (samp == VK_NULL_HANDLE) return;

    if (!_imguiTex[0])
    {
        VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(
            samp,
            _state[0].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        _imguiTex[0] = reinterpret_cast<void *>(ds);
    }
    if (!_imguiTex[1])
    {
        VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(
            samp,
            _state[1].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        _imguiTex[1] = reinterpret_cast<void *>(ds);
    }
}

void GameOfLifePass::register_graph(RenderGraph *graph)
{
    if (!_enabled || !_context || !graph) return;
    if (!_context->pipelines) return;

    ensure_pipelines();
    ensure_images();

    if (_state[0].imageView == VK_NULL_HANDLE || _state[1].imageView == VK_NULL_HANDLE)
    {
        return;
    }

    RGImageHandle h[2]{};
    for (int i = 0; i < 2; ++i)
    {
        RGImportedImageDesc desc{};
        desc.name = std::string("gol.state.") + std::to_string(i);
        desc.image = _state[i].image;
        desc.imageView = _state[i].imageView;
        desc.format = _state[i].imageFormat;
        desc.extent = _extent;
        desc.currentLayout = _layout[i];
        desc.currentStage = (_stage[i] != VK_PIPELINE_STAGE_2_NONE) ? _stage[i] : default_stage_for_layout(_layout[i]);
        desc.currentAccess = (_access[i] != 0) ? _access[i] : default_access_for_layout(_layout[i]);
        h[i] = graph->import_image(desc);
    }

    uint32_t srcIndex = _current;
    uint32_t dstIndex = 1u - _current;

    bool touched[2]{false, false};

    // (Re)initialize state if requested.
    if (_pendingInit != PendingInit::None)
    {
        const float fill = (_pendingInit == PendingInit::Random) ? _pendingFill : 0.0f;
        const uint32_t seed = (_pendingSeed == 0) ? 1u : _pendingSeed;

        const RGImageHandle hDst = h[srcIndex];

        graph->add_pass(
            "GameOfLife.Init",
            RGPassType::Compute,
            [hDst](RGPassBuilder &builder, EngineContext *) {
                builder.write(hDst, RGImageUsage::ComputeWrite);
            },
            [this, hDst, fill, seed](VkCommandBuffer cmd, const RGPassResources &res, EngineContext *ctx) {
                EngineContext *ctxLocal = ctx ? ctx : _context;
                if (!ctxLocal || !ctxLocal->pipelines) return;

                VkImageView dstView = res.image_view(hDst);
                if (dstView == VK_NULL_HANDLE) return;

                ctxLocal->pipelines->setComputeInstanceStorageImage("game_of_life.init", 0, dstView,
                                                                   VK_IMAGE_LAYOUT_GENERAL);

                InitPush pc{};
                pc.width = _extent.width;
                pc.height = _extent.height;
                pc.fill = fill;
                pc.seed = seed;

                ComputeDispatchInfo di = ComputeManager::createDispatch2D(_extent.width, _extent.height,
                                                                         kLocalSize, kLocalSize);
                di.pushConstants = &pc;
                di.pushConstantSize = sizeof(pc);
                ctxLocal->pipelines->dispatchComputeInstance(cmd, "game_of_life.init", di);
            });

        touched[srcIndex] = true;
        _pendingInit = PendingInit::None;
    }

    // Step passes (ping-pong).
    const uint32_t steps = _queuedSteps;
    if (steps > 0)
    {
        for (uint32_t i = 0; i < steps; ++i)
        {
            const RGImageHandle hSrc = h[srcIndex];
            const RGImageHandle hDst = h[dstIndex];

            const std::string passName = std::string("GameOfLife.Step.") + std::to_string(i);

            graph->add_pass(
                passName.c_str(),
                RGPassType::Compute,
                [hSrc, hDst](RGPassBuilder &builder, EngineContext *) {
                    builder.read(hSrc, RGImageUsage::SampledCompute);
                    builder.write(hDst, RGImageUsage::ComputeWrite);
                },
                [this, hSrc, hDst](VkCommandBuffer cmd, const RGPassResources &res, EngineContext *ctx) {
                    EngineContext *ctxLocal = ctx ? ctx : _context;
                    if (!ctxLocal || !ctxLocal->pipelines || !ctxLocal->getSamplers()) return;

                    VkImageView srcView = res.image_view(hSrc);
                    VkImageView dstView = res.image_view(hDst);
                    if (srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE) return;

                    ctxLocal->pipelines->setComputeInstanceSampledImage("game_of_life.step", 0, srcView,
                                                                        ctxLocal->getSamplers()->defaultNearest(),
                                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    ctxLocal->pipelines->setComputeInstanceStorageImage("game_of_life.step", 1, dstView,
                                                                       VK_IMAGE_LAYOUT_GENERAL);

                    StepPush pc{};
                    pc.width = _extent.width;
                    pc.height = _extent.height;
                    pc.wrap = _wrap ? 1u : 0u;
                    pc.pad = 0u;

                    ComputeDispatchInfo di = ComputeManager::createDispatch2D(_extent.width, _extent.height,
                                                                             kLocalSize, kLocalSize);
                    di.pushConstants = &pc;
                    di.pushConstantSize = sizeof(pc);
                    ctxLocal->pipelines->dispatchComputeInstance(cmd, "game_of_life.step", di);
                });

            touched[srcIndex] = true;
            touched[dstIndex] = true;

            std::swap(srcIndex, dstIndex);
        }

        _generation += steps;
        _queuedSteps = 0;
        _current = srcIndex;
    }

    // Ensure the current state is in SHADER_READ_ONLY_OPTIMAL for ImGui sampling next frame.
    const RGImageHandle hDisplay = h[_current];
    graph->add_pass(
        "GameOfLife.PrepareForImGui",
        RGPassType::Compute,
        [hDisplay](RGPassBuilder &builder, EngineContext *) {
            builder.read(hDisplay, RGImageUsage::SampledFragment);
        },
        [](VkCommandBuffer, const RGPassResources &, EngineContext *) {
            // no-op; barrier-only
        });

    touched[_current] = true;

    // Persist best-effort known state into next frame's import.
    for (int i = 0; i < 2; ++i)
    {
        if (touched[i])
        {
            _layout[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            _stage[i] = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            _access[i] = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        }
    }
}
