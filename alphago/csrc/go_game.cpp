#include "go_game.h"
#include <algorithm>
#include <cstring>

GoGame::GoGame(int size) : size(size) {
    n2 = size * size;
    num_planes = 2 * NUM_HISTORY + 1; // 17
    nn_input_size = num_planes * n2;
    pass_action = n2;
    state_size = nn_input_size + 2;

    // Build neighbor table
    for (int idx = 0; idx < n2; idx++) {
        int r = idx / size, c = idx % size;
        int cnt = 0;
        if (r > 0)        neighbors_[idx][cnt++] = idx - size;
        if (r < size - 1)  neighbors_[idx][cnt++] = idx + size;
        if (c > 0)        neighbors_[idx][cnt++] = idx - 1;
        if (c < size - 1)  neighbors_[idx][cnt++] = idx + 1;
        neighbor_count_[idx] = cnt;
    }
}

void GoGame::get_current_board(const float* state, float* board) const {
    const float* p0 = state;                       // plane 0
    const float* p8 = state + NUM_HISTORY * n2;    // plane 8
    for (int i = 0; i < n2; i++) {
        board[i] = p0[i] - p8[i];
    }
}

int GoGame::find_group(const float* board, int idx, FloodFillScratch& s,
                       int& liberty_count) const {
    float color = board[idx];
    int group_size = 0;
    liberty_count = 0;

    int to_clean[361];
    int clean_count = 0;

    s.visited[idx] = 1;
    to_clean[clean_count++] = idx;
    s.stack[0] = idx;
    int sp = 1;

    while (sp > 0) {
        int pos = s.stack[--sp];
        s.stack[group_size++] = pos; // reuse stack front as group storage after pop
        // Actually, let's use a separate approach to avoid overwriting
        // We'll store group in the first group_size entries of a local array
    }

    // Redo properly: group stored separately
    group_size = 0;
    liberty_count = 0;
    clean_count = 0;

    // Reset visited
    s.visited[idx] = 0;

    // Fresh start
    s.visited[idx] = 1;
    to_clean[clean_count++] = idx;

    // Use a separate stack for BFS
    int bfs_stack[361];
    bfs_stack[0] = idx;
    sp = 1;

    while (sp > 0) {
        int pos = bfs_stack[--sp];
        s.stack[group_size++] = pos; // store group in s.stack

        int nc = neighbor_count_[pos];
        for (int ni = 0; ni < nc; ni++) {
            int nbr = neighbors_[pos][ni];
            if (s.visited[nbr]) continue;

            float bval = board[nbr];
            if (bval == color) {
                s.visited[nbr] = 1;
                to_clean[clean_count++] = nbr;
                bfs_stack[sp++] = nbr;
            } else if (bval == 0.0f) {
                s.visited[nbr] = 2; // liberty marker
                to_clean[clean_count++] = nbr;
                liberty_count++;
            }
        }
    }

    // Clean up visited
    for (int i = 0; i < clean_count; i++)
        s.visited[to_clean[i]] = 0;

    // Group is stored in s.stack[0..group_size-1]
    return group_size;
}

bool GoGame::group_has_liberty(const float* board, int idx, FloodFillScratch& s) const {
    float color = board[idx];

    int to_clean[361];
    int clean_count = 0;

    s.visited[idx] = 1;
    to_clean[clean_count++] = idx;

    int bfs_stack[361];
    bfs_stack[0] = idx;
    int sp = 1;
    bool found = false;

    while (sp > 0 && !found) {
        int pos = bfs_stack[--sp];
        int nc = neighbor_count_[pos];
        for (int ni = 0; ni < nc; ni++) {
            int nbr = neighbors_[pos][ni];
            if (s.visited[nbr]) continue;

            float bval = board[nbr];
            if (bval == 0.0f) {
                found = true;
                break;
            }
            if (bval == color) {
                s.visited[nbr] = 1;
                to_clean[clean_count++] = nbr;
                bfs_stack[sp++] = nbr;
            }
        }
    }

    for (int i = 0; i < clean_count; i++)
        s.visited[to_clean[i]] = 0;

    return found;
}

int GoGame::capture_opponent(float* board, int idx, int player, FloodFillScratch& s,
                             int* captured_out) const {
    float opponent = static_cast<float>(-player);
    int total_captured = 0;

    // Track groups we've already checked (using visited value 3)
    int all_marked[361];
    int marked_count = 0;

    int nc = neighbor_count_[idx];
    for (int ni = 0; ni < nc; ni++) {
        int nbr = neighbors_[idx][ni];
        if (board[nbr] == opponent && !s.visited[nbr]) {
            int lib_count;
            int group_size = find_group(board, nbr, s, lib_count);
            // Group is in s.stack[0..group_size-1]

            // Mark group stones so we skip if another neighbor belongs to same group
            for (int gi = 0; gi < group_size; gi++) {
                int pos = s.stack[gi];
                s.visited[pos] = 3;
                all_marked[marked_count++] = pos;
            }

            if (lib_count == 0) {
                for (int gi = 0; gi < group_size; gi++) {
                    int pos = s.stack[gi];
                    captured_out[total_captured++] = pos;
                    board[pos] = 0.0f;
                }
            }
        }
    }

    // Clean up markers
    for (int i = 0; i < marked_count; i++)
        s.visited[all_marked[i]] = 0;

    return total_captured;
}

bool GoGame::is_suicide(float* board, int idx, int player, FloodFillScratch& s) const {
    // Fast path: if any neighbor is empty, not suicide
    int nc = neighbor_count_[idx];
    for (int ni = 0; ni < nc; ni++) {
        if (board[neighbors_[idx][ni]] == 0.0f)
            return false;
    }

    // Surrounded by stones — need full check
    float fplayer = static_cast<float>(player);
    float opponent = static_cast<float>(-player);

    board[idx] = fplayer;

    // Check if placing captures any opponent
    for (int ni = 0; ni < nc; ni++) {
        int nbr = neighbors_[idx][ni];
        if (board[nbr] == opponent) {
            if (!group_has_liberty(board, nbr, s)) {
                board[idx] = 0.0f;
                return false; // captures → not suicide
            }
        }
    }

    // Check own group liberties
    bool has_lib = group_has_liberty(board, idx, s);
    board[idx] = 0.0f;
    return !has_lib;
}

void GoGame::get_initial_state(float* state) const {
    std::memset(state, 0, state_size * sizeof(float));
    // Color plane: 1.0 = player 1 to move
    float* planes = state;
    for (int i = 0; i < n2; i++) {
        planes[COLOR_PLANE * n2 + i] = 1.0f;
    }
    set_pass_count(state, 0);
    set_ko_point(state, -1);
}

void GoGame::get_next_state(const float* state, int action, int player, float* out) const {
    FloodFillScratch scratch;
    std::memset(scratch.visited, 0, n2);
    get_next_state(state, action, player, out, scratch);
}

void GoGame::get_next_state(const float* state, int action, int player, float* out,
                            FloodFillScratch& s) const {
    // Copy state
    std::memcpy(out, state, state_size * sizeof(float));

    float* planes = out;

    if (action == pass_action) {
        set_pass_count(out, get_pass_count(out) + 1);
        set_ko_point(out, -1);
        // Flip color plane
        for (int i = 0; i < n2; i++)
            planes[COLOR_PLANE * n2 + i] = 1.0f - planes[COLOR_PLANE * n2 + i];
        return;
    }

    // Non-pass: reset pass count
    set_pass_count(out, 0);

    // Get current board, place stone, capture
    float board[361];
    get_current_board(out, board);
    board[action] = static_cast<float>(player);

    int captured_indices[361];
    int num_captured = capture_opponent(board, action, player, s, captured_indices);

    // Ko detection
    int ko = -1;
    if (num_captured == 1) {
        int lib_count;
        int group_size = find_group(board, action, s, lib_count);
        if (group_size == 1 && lib_count == 1) {
            ko = captured_indices[0];
        }
    }
    set_ko_point(out, ko);

    // Shift history planes for player 1 (planes 0-7)
    // planes[1..7] = planes[0..6]  (shift right)
    for (int p = NUM_HISTORY - 1; p >= 1; p--) {
        std::memcpy(planes + p * n2, planes + (p - 1) * n2, n2 * sizeof(float));
    }
    // Plane 0 = (board == 1.0)
    for (int i = 0; i < n2; i++) {
        planes[i] = (board[i] == 1.0f) ? 1.0f : 0.0f;
    }

    // Shift history planes for player -1 (planes 8-15)
    for (int p = 2 * NUM_HISTORY - 1; p >= NUM_HISTORY + 1; p--) {
        std::memcpy(planes + p * n2, planes + (p - 1) * n2, n2 * sizeof(float));
    }
    // Plane 8 = (board == -1.0)
    for (int i = 0; i < n2; i++) {
        planes[NUM_HISTORY * n2 + i] = (board[i] == -1.0f) ? 1.0f : 0.0f;
    }

    // Flip color plane
    for (int i = 0; i < n2; i++) {
        planes[COLOR_PLANE * n2 + i] = 1.0f - planes[COLOR_PLANE * n2 + i];
    }
}

void GoGame::get_valid_moves(const float* state, int player, float* valid,
                             FloodFillScratch& s) const {
    std::memset(valid, 0, (n2 + 1) * sizeof(float));

    float board[361];
    get_current_board(state, board);
    int ko = get_ko_point(state);

    for (int idx = 0; idx < n2; idx++) {
        if (board[idx] != 0.0f) continue;
        if (idx == ko) continue;

        // Fast path: any empty neighbor = has a liberty
        bool has_liberty = false;
        int nc = neighbor_count_[idx];
        for (int ni = 0; ni < nc; ni++) {
            if (board[neighbors_[idx][ni]] == 0.0f) {
                has_liberty = true;
                break;
            }
        }

        if (has_liberty) {
            valid[idx] = 1.0f;
        } else if (!is_suicide(board, idx, player, s)) {
            valid[idx] = 1.0f;
        }
    }

    // Pass is always valid
    valid[pass_action] = 1.0f;
}

bool GoGame::check_terminal(const float* state, int action, int player, float& value) const {
    FloodFillScratch scratch;
    std::memset(scratch.visited, 0, n2);
    return check_terminal(state, action, player, value, scratch);
}

bool GoGame::check_terminal(const float* state, int action, int player, float& value,
                            FloodFillScratch& s) const {
    if (get_pass_count(state) < 2) {
        value = 0.0f;
        return false;
    }

    float board[361];
    get_current_board(state, board);
    float score = tromp_taylor_score(board, s);

    if (score > 0.0f) {
        value = (player == 1) ? 1.0f : -1.0f;
    } else if (score < 0.0f) {
        value = (player == -1) ? 1.0f : -1.0f;
    } else {
        value = 0.0f;
    }
    return true;
}

float GoGame::tromp_taylor_score(const float* board, FloodFillScratch& s) const {
    float komi = 7.5f;
    float p1_score = 0.0f, p2_score = 0.0f;

    // Count stones
    for (int i = 0; i < n2; i++) {
        if (board[i] == 1.0f) p1_score += 1.0f;
        else if (board[i] == -1.0f) p2_score += 1.0f;
    }

    // Flood-fill empty regions
    uint8_t region_visited[361];
    std::memset(region_visited, 0, n2);

    for (int idx = 0; idx < n2; idx++) {
        if (board[idx] != 0.0f || region_visited[idx]) continue;

        // Flood fill this empty region
        int region_size = 0;
        int borders = 0; // bitmask: bit 0 = player 1, bit 1 = player -1

        int ff_stack[361];
        ff_stack[0] = idx;
        int sp = 1;
        region_visited[idx] = 1;

        while (sp > 0) {
            int pos = ff_stack[--sp];
            region_size++;

            int nc = neighbor_count_[pos];
            for (int ni = 0; ni < nc; ni++) {
                int nbr = neighbors_[pos][ni];
                if (region_visited[nbr]) continue;

                float bval = board[nbr];
                if (bval == 0.0f) {
                    region_visited[nbr] = 1;
                    ff_stack[sp++] = nbr;
                } else {
                    region_visited[nbr] = 1;
                    if (bval == 1.0f) borders |= 1;
                    else if (bval == -1.0f) borders |= 2;
                }
            }
        }

        if (borders == 1) p1_score += region_size;       // only black borders
        else if (borders == 2) p2_score += region_size;   // only white borders
    }

    return p1_score - p2_score - komi;
}

void GoGame::get_canonical_state(const float* state, int player, float* out) const {
    if (player == 1) {
        std::memcpy(out, state, nn_input_size * sizeof(float));
        return;
    }

    // player == -1: swap planes 0-7 <-> 8-15, flip color plane
    const float* src = state;

    // Copy planes 8-15 → out planes 0-7
    std::memcpy(out, src + NUM_HISTORY * n2, NUM_HISTORY * n2 * sizeof(float));
    // Copy planes 0-7 → out planes 8-15
    std::memcpy(out + NUM_HISTORY * n2, src, NUM_HISTORY * n2 * sizeof(float));
    // Flip color plane
    for (int i = 0; i < n2; i++) {
        out[COLOR_PLANE * n2 + i] = 1.0f - src[COLOR_PLANE * n2 + i];
    }
}
