/**
 * Keyboard teleop for Booster T1 — WASD + Space.
 *
 * Uses B1LocoClient from the Booster Robotics SDK (C++).
 * No ROS2 required.
 *
 * Build inside the container:
 *   cd /app/exchange/teleop_cpp
 *   mkdir build && cd build
 *   cmake .. && make
 *   ./teleop
 *
 * Controls:
 *   w / s      forward / backward   (vx ±0.1)
 *   a / d      strafe left / right  (vy ±0.1)
 *   q / e      rotate left / right  (vyaw ±0.15)
 *   Space      stop
 *   p          Walking mode
 *   o          Prepare mode
 *   i          Damping mode (safe stop)
 *   Ctrl+C     quit + stop
 */

#include <booster/robot/b1/b1_loco_client.hpp>
#include <booster/robot/channel/channel_factory.hpp>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <termios.h>
#include <unistd.h>

using namespace booster::robot;
using namespace booster::robot::b1;

// ---- Raw terminal input ----
struct RawTerm {
    termios orig;
    RawTerm() {
        tcgetattr(STDIN_FILENO, &orig);
        termios raw = orig;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN]  = 0;   // non-blocking
        raw.c_cc[VTIME] = 1;   // 100ms timeout
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
    ~RawTerm() { tcsetattr(STDIN_FILENO, TCSANOW, &orig); }
};

// ---- Velocity limits ----
static constexpr float VX_STEP  = 0.10f, VX_MAX  = 0.8f;
static constexpr float VY_STEP  = 0.10f, VY_MAX  = 0.4f;
static constexpr float YAW_STEP = 0.15f, YAW_MAX = 0.6f;

static float clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

// Shared state for signal handler
static B1LocoClient* g_client = nullptr;

static void on_signal(int) {
    if (g_client) g_client->Move(0, 0, 0);
    std::printf("\nExiting.\n");
    std::exit(0);
}

static void print_state(float vx, float vy, float vyaw, const char* mode) {
    std::printf("\033[2J\033[H");   // clear screen
    std::printf("=== Booster T1 Teleop ===\n\n");
    std::printf("  vx   = %+.2f m/s    (w/s)\n", vx);
    std::printf("  vy   = %+.2f m/s    (a/d)\n", vy);
    std::printf("  vyaw = %+.2f rad/s  (q/e)\n", vyaw);
    std::printf("\n  mode = %s\n", mode);
    std::printf("\n  [space]=stop  [p]=walk  [o]=prepare  [i]=damp  [Ctrl+C]=quit\n");
    std::fflush(stdout);
}

int main() {
    ChannelFactory::Instance()->Init(0);

    B1LocoClient client;
    client.Init();
    g_client = &client;

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Start in Prepare mode so the robot stands up immediately
    client.ChangeMode(RobotMode::kPrepare);

    float vx = 0, vy = 0, vyaw = 0;
    const char* mode = "Prepare";

    RawTerm raw;
    print_state(vx, vy, vyaw, mode);

    while (true) {
        char ch = 0;
        read(STDIN_FILENO, &ch, 1);

        bool changed = true;
        switch (ch) {
            case 'w': vx   = clamp(vx   + VX_STEP,  -VX_MAX,  VX_MAX);  break;
            case 's': vx   = clamp(vx   - VX_STEP,  -VX_MAX,  VX_MAX);  break;
            case 'a': vy   = clamp(vy   + VY_STEP,  -VY_MAX,  VY_MAX);  break;
            case 'd': vy   = clamp(vy   - VY_STEP,  -VY_MAX,  VY_MAX);  break;
            case 'q': vyaw = clamp(vyaw + YAW_STEP, -YAW_MAX, YAW_MAX); break;
            case 'e': vyaw = clamp(vyaw - YAW_STEP, -YAW_MAX, YAW_MAX); break;
            case ' ': vx = vy = vyaw = 0;                                 break;
            case 'p': client.ChangeMode(RobotMode::kWalking); mode = "Walking"; break;
            case 'o': client.ChangeMode(RobotMode::kPrepare); mode = "Prepare"; break;
            case 'i': vx = vy = vyaw = 0;
                      client.ChangeMode(RobotMode::kDamping); mode = "Damping"; break;
            case 'g': client.GetUp(); mode = "GetUp"; break;
            case 3:   // Ctrl+C
                client.Move(0, 0, 0);
                return 0;
            default:  changed = false; break;
        }

        if (changed) {
            client.Move(vx, vy, vyaw);
            print_state(vx, vy, vyaw, mode);
        }
    }
}
