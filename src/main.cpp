#include "core/engine.h"
#include "runtime/game_runtime.h"
#include "game/minimal_game.h"

#include <memory>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    VulkanEngine engine;
    engine.init();

    {
        GameRuntime::Runtime runtime(&engine);
        std::unique_ptr<GameRuntime::IGameCallbacks> game = std::make_unique<Game::MinimalGame>();
        runtime.run(game.get());
    }

    engine.cleanup();
    return 0;
}
