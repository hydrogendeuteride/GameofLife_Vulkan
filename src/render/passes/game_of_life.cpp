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
    constexpr uint32_t kMaxStepsPerFrame = 8;

    // Word-grid dispatch (1 invocation processes 32 cells in X).
    constexpr uint32_t kWordLocalX = 8;
    constexpr uint32_t kWordLocalY = 8;

    // Preview generation (1 invocation processes 1 preview pixel).
    constexpr uint32_t kPreviewLocal = 16;

    struct StepBitsPush
    {
        uint32_t width;
        uint32_t height;
        uint32_t wordsPerRow;
        uint32_t wrap;
    };

    struct InitBitsPush
    {
        uint32_t width;
        uint32_t height;
        uint32_t wordsPerRow;
        uint32_t seed;
        uint32_t fillThreshold;
        uint32_t _pad0;
        uint32_t _pad1;
        uint32_t _pad2;
    };

    struct PreviewPush
    {
        uint32_t simWidth;
        uint32_t simHeight;
        uint32_t previewWidth;
        uint32_t previewHeight;
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
    _imguiPreviewTex = nullptr;

    if (_context && _context->getResources())
    {
        for (auto &buf: _bits)
        {
            if (buf.buffer != VK_NULL_HANDLE)
            {
                _context->getResources()->destroy_buffer(buf);
            }
            buf = {};
        }

        if (_preview.image != VK_NULL_HANDLE)
        {
            _context->getResources()->destroy_image(_preview);
        }
        _preview = {};
    }

    if (_context && _context->pipelines)
    {
        _context->pipelines->destroyComputeInstance("game_of_life_bits.init");
        _context->pipelines->destroyComputePipeline("game_of_life_bits.init");
        _context->pipelines->destroyComputeInstance("game_of_life_bits.step");
        _context->pipelines->destroyComputePipeline("game_of_life_bits.step");
        _context->pipelines->destroyComputeInstance("game_of_life_bits.preview");
        _context->pipelines->destroyComputePipeline("game_of_life_bits.preview");
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
        // If buffers haven't been created yet, request an init so the content is valid.
        if (_bits[0].buffer == VK_NULL_HANDLE || _bits[1].buffer == VK_NULL_HANDLE)
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
    if (!_enabled || !_previewEnabled) return nullptr;
    ensure_preview_image();
    ensure_imgui_textures();
    return _imguiPreviewTex;
}

void GameOfLifePass::ensure_pipelines()
{
    if (!_context || !_context->pipelines || !_context->getAssets()) return;

    if (!_context->pipelines->hasComputePipeline("game_of_life_bits.init"))
    {
        ComputePipelineCreateInfo ci{};
        ci.shaderPath = _context->getAssets()->shaderPath("game_of_life_bits_init.comp.spv");
        ci.descriptorTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        ci.pushConstantSize = sizeof(InitBitsPush);
        _context->pipelines->createComputePipeline("game_of_life_bits.init", ci);
    }

    if (!_context->pipelines->hasComputePipeline("game_of_life_bits.step"))
    {
        ComputePipelineCreateInfo ci{};
        ci.shaderPath = _context->getAssets()->shaderPath("game_of_life_bits_step.comp.spv");
        ci.descriptorTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        ci.pushConstantSize = sizeof(StepBitsPush);
        _context->pipelines->createComputePipeline("game_of_life_bits.step", ci);
    }

    if (!_context->pipelines->hasComputePipeline("game_of_life_bits.preview"))
    {
        ComputePipelineCreateInfo ci{};
        ci.shaderPath = _context->getAssets()->shaderPath("game_of_life_bits_preview.comp.spv");
        ci.descriptorTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        ci.pushConstantSize = sizeof(PreviewPush);
        _context->pipelines->createComputePipeline("game_of_life_bits.preview", ci);
    }

    if (!_instancesCreated)
    {
        _context->pipelines->createComputeInstance("game_of_life_bits.init", "game_of_life_bits.init");
        _context->pipelines->createComputeInstance("game_of_life_bits.step", "game_of_life_bits.step");
        _context->pipelines->createComputeInstance("game_of_life_bits.preview", "game_of_life_bits.preview");
        _instancesCreated = true;
    }
}

void GameOfLifePass::ensure_buffers()
{
    if (!_context || !_context->getResources()) return;
    if (_bits[0].buffer != VK_NULL_HANDLE && _bits[1].buffer != VK_NULL_HANDLE) return;

    if (_simExtent.width < 32) _simExtent.width = 32;
    if ((_simExtent.width & 31u) != 0u)
    {
        // Keep 1-bit packing simple: enforce a 32-cell word boundary.
        _simExtent.width &= ~31u;
        if (_simExtent.width < 32) _simExtent.width = 32;
    }

    _wordsPerRow = _simExtent.width / 32u;
    _bitsSizeBytes = static_cast<VkDeviceSize>(_wordsPerRow) * static_cast<VkDeviceSize>(_simExtent.height) *
                     static_cast<VkDeviceSize>(sizeof(uint32_t));

    for (int i = 0; i < 2; ++i)
    {
        if (_bits[i].buffer == VK_NULL_HANDLE)
        {
            _bits[i] = _context->getResources()->create_buffer(static_cast<size_t>(_bitsSizeBytes),
                                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                               VMA_MEMORY_USAGE_GPU_ONLY);
            _bufStage[i] = VK_PIPELINE_STAGE_2_NONE;
            _bufAccess[i] = 0;
        }
    }
}

void GameOfLifePass::ensure_preview_image()
{
    if (!_context || !_context->getResources()) return;
    if (_preview.image != VK_NULL_HANDLE) return;

    VkExtent3D extent3d{_previewExtent.width, _previewExtent.height, 1};
    constexpr VkFormat fmt = VK_FORMAT_R8_UNORM;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    _preview = _context->getResources()->create_image(extent3d, fmt, usage, false);
    _previewLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    _previewStage = VK_PIPELINE_STAGE_2_NONE;
    _previewAccess = 0;
}

void GameOfLifePass::ensure_imgui_textures()
{
    if (_imguiPreviewTex) return;
    if (!_context || !_context->getSamplers()) return;
    if (_preview.imageView == VK_NULL_HANDLE) return;

    const VkSampler samp = _context->getSamplers()->nearestClampEdge();
    if (samp == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(
        samp,
        _preview.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    _imguiPreviewTex = reinterpret_cast<void *>(ds);
}

void GameOfLifePass::register_graph(RenderGraph *graph)
{
    if (!_enabled || !_context || !graph) return;
    if (!_context->pipelines) return;

    ensure_pipelines();
    ensure_buffers();

    if (_bits[0].buffer == VK_NULL_HANDLE || _bits[1].buffer == VK_NULL_HANDLE)
    {
        return;
    }

    RGBufferHandle b[2]{};
    for (int i = 0; i < 2; ++i)
    {
        RGImportedBufferDesc desc{};
        desc.name = std::string("gol.bits.") + std::to_string(i);
        desc.buffer = _bits[i].buffer;
        desc.size = _bitsSizeBytes;
        desc.currentStage = _bufStage[i];
        desc.currentAccess = _bufAccess[i];
        b[i] = graph->import_buffer(desc);
    }

    RGImageHandle hPreview{};
    if (_previewEnabled)
    {
        ensure_preview_image();
        if (_preview.imageView == VK_NULL_HANDLE)
        {
            return;
        }

        RGImportedImageDesc desc{};
        desc.name = "gol.preview";
        desc.image = _preview.image;
        desc.imageView = _preview.imageView;
        desc.format = _preview.imageFormat;
        desc.extent = _previewExtent;
        desc.currentLayout = _previewLayout;
        desc.currentStage = (_previewStage != VK_PIPELINE_STAGE_2_NONE) ? _previewStage : default_stage_for_layout(_previewLayout);
        desc.currentAccess = (_previewAccess != 0) ? _previewAccess : default_access_for_layout(_previewLayout);
        hPreview = graph->import_image(desc);
    }

    uint32_t srcIndex = _current;
    uint32_t dstIndex = 1u - _current;

    bool bufTouched[2]{false, false};
    bool previewTouched = false;

    // (Re)initialize state if requested.
    if (_pendingInit != PendingInit::None)
    {
        const float fill = (_pendingInit == PendingInit::Random) ? _pendingFill : 0.0f;
        const uint32_t seed = (_pendingSeed == 0) ? 1u : _pendingSeed;

        uint32_t fillThreshold = 0u;
        if (fill >= 1.0f)
        {
            fillThreshold = 0xFFFFFFFFu;
        }
        else if (fill > 0.0f)
        {
            double scaled = static_cast<double>(fill) * 4294967295.0;
            scaled = std::clamp(scaled, 0.0, 4294967295.0);
            fillThreshold = static_cast<uint32_t>(scaled);
        }

        const RGBufferHandle hDst = b[srcIndex];

        graph->add_pass(
            "GameOfLife.InitBits",
            RGPassType::Compute,
            [hDst](RGPassBuilder &builder, EngineContext *) {
                builder.write_buffer(hDst, RGBufferUsage::StorageReadWrite);
            },
            [this, hDst, fillThreshold, seed](VkCommandBuffer cmd, const RGPassResources &res, EngineContext *ctx) {
                EngineContext *ctxLocal = ctx ? ctx : _context;
                if (!ctxLocal || !ctxLocal->pipelines) return;

                VkBuffer dstBuf = res.buffer(hDst);
                if (dstBuf == VK_NULL_HANDLE) return;

                ctxLocal->pipelines->setComputeInstanceBuffer("game_of_life_bits.init", 0, dstBuf, _bitsSizeBytes,
                                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);

                InitBitsPush pc{};
                pc.width = _simExtent.width;
                pc.height = _simExtent.height;
                pc.wordsPerRow = _wordsPerRow;
                pc.seed = seed;
                pc.fillThreshold = fillThreshold;
                pc._pad0 = pc._pad1 = pc._pad2 = 0u;

                ComputeDispatchInfo di = ComputeManager::createDispatch2D(_wordsPerRow, _simExtent.height,
                                                                         kWordLocalX, kWordLocalY);
                di.pushConstants = &pc;
                di.pushConstantSize = sizeof(pc);
                ctxLocal->pipelines->dispatchComputeInstance(cmd, "game_of_life_bits.init", di);
            });

        bufTouched[srcIndex] = true;
        _pendingInit = PendingInit::None;
    }

    // Step passes (ping-pong).
    const uint32_t steps = _queuedSteps;
    if (steps > 0)
    {
        for (uint32_t i = 0; i < steps; ++i)
        {
            const RGBufferHandle bSrc = b[srcIndex];
            const RGBufferHandle bDst = b[dstIndex];

            const std::string passName = std::string("GameOfLife.Step.") + std::to_string(i);

            graph->add_pass(
                passName.c_str(),
                RGPassType::Compute,
                [bSrc, bDst](RGPassBuilder &builder, EngineContext *) {
                    builder.read_buffer(bSrc, RGBufferUsage::StorageRead);
                    builder.write_buffer(bDst, RGBufferUsage::StorageReadWrite);
                },
                [this, bSrc, bDst](VkCommandBuffer cmd, const RGPassResources &res, EngineContext *ctx) {
                    EngineContext *ctxLocal = ctx ? ctx : _context;
                    if (!ctxLocal || !ctxLocal->pipelines) return;

                    VkBuffer srcBuf = res.buffer(bSrc);
                    VkBuffer dstBuf = res.buffer(bDst);
                    if (srcBuf == VK_NULL_HANDLE || dstBuf == VK_NULL_HANDLE) return;

                    ctxLocal->pipelines->setComputeInstanceBuffer("game_of_life_bits.step", 0, srcBuf, _bitsSizeBytes,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);
                    ctxLocal->pipelines->setComputeInstanceBuffer("game_of_life_bits.step", 1, dstBuf, _bitsSizeBytes,
                                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);

                    StepBitsPush pc{};
                    pc.width = _simExtent.width;
                    pc.height = _simExtent.height;
                    pc.wordsPerRow = _wordsPerRow;
                    pc.wrap = _wrap ? 1u : 0u;

                    ComputeDispatchInfo di = ComputeManager::createDispatch2D(_wordsPerRow, _simExtent.height,
                                                                             kWordLocalX, kWordLocalY);
                    di.pushConstants = &pc;
                    di.pushConstantSize = sizeof(pc);
                    ctxLocal->pipelines->dispatchComputeInstance(cmd, "game_of_life_bits.step", di);
                });

            bufTouched[srcIndex] = true;
            bufTouched[dstIndex] = true;

            std::swap(srcIndex, dstIndex);
        }

        _generation += steps;
        _queuedSteps = 0;
        _current = srcIndex;
    }

    if (_previewEnabled && hPreview.valid())
    {
        // Downsample current bit-grid into preview image for ImGui sampling.
        const RGBufferHandle bCur = b[_current];
        graph->add_pass(
            "GameOfLife.Preview",
            RGPassType::Compute,
            [bCur, hPreview](RGPassBuilder &builder, EngineContext *) {
                builder.read_buffer(bCur, RGBufferUsage::StorageRead);
                builder.write(hPreview, RGImageUsage::ComputeWrite);
            },
            [this, bCur, hPreview](VkCommandBuffer cmd, const RGPassResources &res, EngineContext *ctx) {
                EngineContext *ctxLocal = ctx ? ctx : _context;
                if (!ctxLocal || !ctxLocal->pipelines) return;

                VkBuffer srcBuf = res.buffer(bCur);
                VkImageView dstView = res.image_view(hPreview);
                if (srcBuf == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE) return;

                ctxLocal->pipelines->setComputeInstanceBuffer("game_of_life_bits.preview", 0, srcBuf, _bitsSizeBytes,
                                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);
                ctxLocal->pipelines->setComputeInstanceStorageImage("game_of_life_bits.preview", 1, dstView,
                                                                    VK_IMAGE_LAYOUT_GENERAL);

                PreviewPush pc{};
                pc.simWidth = _simExtent.width;
                pc.simHeight = _simExtent.height;
                pc.previewWidth = _previewExtent.width;
                pc.previewHeight = _previewExtent.height;

                ComputeDispatchInfo di = ComputeManager::createDispatch2D(_previewExtent.width, _previewExtent.height,
                                                                         kPreviewLocal, kPreviewLocal);
                di.pushConstants = &pc;
                di.pushConstantSize = sizeof(pc);
                ctxLocal->pipelines->dispatchComputeInstance(cmd, "game_of_life_bits.preview", di);
            });

        bufTouched[_current] = true;
        previewTouched = true;

        // Ensure preview is in SHADER_READ_ONLY_OPTIMAL for ImGui sampling next frame.
        graph->add_pass(
            "GameOfLife.PrepareForImGui",
            RGPassType::Compute,
            [hPreview](RGPassBuilder &builder, EngineContext *) {
                builder.read(hPreview, RGImageUsage::SampledFragment);
            },
            [](VkCommandBuffer, const RGPassResources &, EngineContext *) {
                // no-op; barrier-only
            });
    }

    // Persist best-effort known state into next frame's import.
    for (int i = 0; i < 2; ++i)
    {
        if (bufTouched[i])
        {
            _bufStage[i] = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            _bufAccess[i] = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        }
    }

    if (previewTouched)
    {
        _previewLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        _previewStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        _previewAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }
}
