#include "minimal_game.h"

#include "runtime/game_runtime.h"
#include "core/engine.h"

#include "render/passes/game_of_life.h"

#include "imgui.h"

namespace Game
{

void MinimalGame::on_init(GameRuntime::Runtime &runtime)
{
    _runtime = &runtime;

    if (VulkanEngine *renderer = runtime.renderer())
    {
        _render_scale = renderer->renderScale;

        if (renderer->ui())
        {
            renderer->ui()->add_draw_callback([this]() { draw_ui(); });
        }
    }
}

void MinimalGame::on_update(float dt)
{
    if (!_runtime)
    {
        return;
    }

    VulkanEngine *renderer = _runtime->renderer();
    if (!renderer || !renderer->_renderPassManager)
    {
        return;
    }

    if (auto *gol = renderer->_renderPassManager->getPass<GameOfLifePass>())
    {
        gol->update(dt);
    }
}

void MinimalGame::on_fixed_update(float /*fixed_dt*/)
{
}

void MinimalGame::on_shutdown()
{
    _runtime = nullptr;
}

void MinimalGame::draw_ui()
{
    if (!_ui_open)
    {
        return;
    }

    if (!ImGui::Begin("Minimal (Compute + ImGui)", &_ui_open, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }

    if (!_runtime)
    {
        ImGui::TextUnformatted("Runtime not available");
        ImGui::End();
        return;
    }

    VulkanEngine *renderer = _runtime->renderer();
    if (renderer)
    {
        ImGui::Text("Frame: %d", renderer->_frameNumber);
        ImGui::Text("Frame time: %.2f ms", renderer->stats.frametime);
    }

    ImGui::Separator();
    ImGui::TextUnformatted("This project keeps game code minimal.");
    ImGui::TextUnformatted("Rendering pipeline still exists; game logic is intentionally empty.");

    ImGui::Separator();
    if (renderer && renderer->_renderPassManager)
    {
        if (auto *gol = renderer->_renderPassManager->getPass<GameOfLifePass>())
        {
            VkExtent2D e = gol->extent();
            ImGui::Text("Game of Life (Compute, %ux%u)", e.width, e.height);

            bool enabled = gol->enabled();
            if (ImGui::Checkbox("Enable", &enabled))
            {
                gol->set_enabled(enabled);
            }

            bool paused = gol->paused();
            ImGui::SameLine();
            if (ImGui::Checkbox("Paused", &paused))
            {
                gol->set_paused(paused);
            }

            bool wrap = gol->wrap();
            ImGui::SameLine();
            if (ImGui::Checkbox("Wrap", &wrap))
            {
                gol->set_wrap(wrap);
            }

            bool preview = gol->preview_enabled();
            ImGui::SameLine();
            if (ImGui::Checkbox("Preview", &preview))
            {
                gol->set_preview_enabled(preview);
            }

            float hz = gol->simulation_hz();
            if (ImGui::SliderFloat("Sim Hz", &hz, 0.0f, 120.0f, "%.1f"))
            {
                gol->set_simulation_hz(hz);
            }

            ImGui::SliderFloat("Fill", &_gol_fill, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Zoom", &_gol_zoom, 0.01f, 0.5f, "%.3f");
            ImGui::InputScalar("Seed", ImGuiDataType_U32, &_gol_seed);

            if (ImGui::Button("Randomize"))
            {
                const uint32_t seed = (_gol_seed == 0) ? static_cast<uint32_t>(renderer->_frameNumber + 1) : _gol_seed;
                gol->set_enabled(true);
                gol->request_randomize(_gol_fill, seed);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear"))
            {
                gol->set_enabled(true);
                gol->request_clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Step"))
            {
                gol->set_enabled(true);
                gol->request_step(1);
            }

            ImGui::Text("Generation: %llu", static_cast<unsigned long long>(gol->generation()));

            if (void *tex = gol->imgui_texture_id())
            {
                VkExtent2D e = gol->preview_extent();
                ImVec2 size{static_cast<float>(e.width) * _gol_zoom, static_cast<float>(e.height) * _gol_zoom};
                ImGui::Image(tex, size);
            }
            else
            {
                ImGui::TextUnformatted(gol->preview_enabled() ? "Enable the pass to view output." : "Preview disabled.");
            }
        }
        else
        {
            ImGui::TextUnformatted("GameOfLifePass not available");
        }
    }

    ImGui::Separator();
    if (renderer)
    {
        if (ImGui::SliderFloat("Render scale", &_render_scale, 0.25f, 2.0f, "%.2f"))
        {
            renderer->set_render_scale(_render_scale);
        }
    }

    if (ImGui::Button("Quit"))
    {
        _runtime->request_quit();
    }

    ImGui::End();
}

} // namespace Game
