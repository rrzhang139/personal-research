#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>

static constexpr int NUM_HISTORY = 8;
static constexpr int COLOR_PLANE = 2 * NUM_HISTORY; // plane 16

// Per-thread scratch space for flood fill — never shared between threads.
struct FloodFillScratch {
    uint8_t visited[361]; // max 19x19
    int stack[361];

    void clear_visited(const int* indices, int count) {
        for (int i = 0; i < count; i++)
            visited[indices[i]] = 0;
    }
};

class GoGame {
public:
    int size;
    int n2;
    int num_planes;       // 17
    int nn_input_size;    // 17 * n2
    int pass_action;      // n2
    int state_size;       // nn_input_size + 2

    // Precomputed neighbor table: neighbors_[idx] = {nbr0, nbr1, ...}, count in neighbor_count_
    int neighbors_[361][4];
    int neighbor_count_[361];

    GoGame(int size = 9);

    // State accessors
    float* get_planes(float* state) const { return state; }
    const float* get_planes(const float* state) const { return state; }

    // Board = plane[0] - plane[8]  (current positions: +1 black, -1 white)
    void get_current_board(const float* state, float* board) const;

    int get_pass_count(const float* state) const {
        return static_cast<int>(state[nn_input_size]);
    }
    void set_pass_count(float* state, int count) const {
        state[nn_input_size] = static_cast<float>(count);
    }
    int get_ko_point(const float* state) const {
        return static_cast<int>(state[nn_input_size + 1]);
    }
    void set_ko_point(float* state, int ko) const {
        state[nn_input_size + 1] = static_cast<float>(ko);
    }
    int get_color_to_move(const float* state) const {
        return (state[COLOR_PLANE * n2] > 0.5f) ? 1 : -1;
    }

    // Flood fill helpers (take scratch explicitly for thread safety)
    // Returns group size, writes group into scratch.stack[0..return-1]
    // liberty_count is output param
    int find_group(const float* board, int idx, FloodFillScratch& s, int& liberty_count) const;

    bool group_has_liberty(const float* board, int idx, FloodFillScratch& s) const;

    // Capture opponent groups adjacent to idx. Modifies board in-place.
    // Returns number of captured stones, writes captured indices into captured_out.
    int capture_opponent(float* board, int idx, int player, FloodFillScratch& s,
                         int* captured_out) const;

    bool is_suicide(float* board, int idx, int player, FloodFillScratch& s) const;

    // Game interface
    void get_initial_state(float* state) const;

    // Returns new state in out (caller allocates state_size floats)
    void get_next_state(const float* state, int action, int player, float* out) const;
    // Version using scratch for thread safety
    void get_next_state(const float* state, int action, int player, float* out,
                        FloodFillScratch& s) const;

    // Writes valid moves into valid (caller allocates n2+1 floats, zeroed)
    void get_valid_moves(const float* state, int player, float* valid,
                         FloodFillScratch& s) const;

    // Returns true if terminal, value from player's perspective
    bool check_terminal(const float* state, int action, int player, float& value) const;
    bool check_terminal(const float* state, int action, int player, float& value,
                        FloodFillScratch& s) const;

    float tromp_taylor_score(const float* board, FloodFillScratch& s) const;

    // Canonical state: if player==1, copy nn planes; if player==-1, swap planes
    void get_canonical_state(const float* state, int player, float* out) const;
};
