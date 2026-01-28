#pragma once

#include "runtime/i_game_callbacks.h"

#include <cstdint>

namespace GameRuntime
{
class Runtime;
}

namespace Game
{

class MinimalGame final : public GameRuntime::IGameCallbacks
{
public:
    MinimalGame() = default;
    ~MinimalGame() override = default;

    void on_init(GameRuntime::Runtime &runtime) override;
    void on_update(float dt) override;
    void on_fixed_update(float fixed_dt) override;
    void on_shutdown() override;

private:
    void draw_ui();

    GameRuntime::Runtime *_runtime{nullptr};
    bool _ui_open{true};
    float _render_scale{1.0f};

    // Game of Life UI controls
    float _gol_zoom{0.125f};
    float _gol_fill{0.15f};
    uint32_t _gol_seed{1};
};

} // namespace Game
