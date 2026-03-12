#include "self_play.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

// --- SelfPlayWorker ---

SelfPlayWorker::SelfPlayWorker(const GoGame& game, const MCTSCppConfig& config,
                               int worker_id, unsigned seed)
    : game_(game), config_(config), worker_id_(worker_id),
      arena_(8 * 1024 * 1024), rng_(seed) {
    std::memset(scratch_.visited, 0, sizeof(scratch_.visited));
}

void SelfPlayWorker::add_dirichlet_noise(MCTSNode* root) {
    if (config_.dirichlet_epsilon == 0.0f) return;
    int n = root->num_children;
    if (n == 0) return;

    // Sample Dirichlet via gamma distribution
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    std::vector<float> noise(n);
    float noise_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        noise[i] = gamma(rng_);
        noise_sum += noise[i];
    }
    if (noise_sum > 0.0f) {
        float inv = 1.0f / noise_sum;
        for (int i = 0; i < n; i++) noise[i] *= inv;
    }

    float eps = config_.dirichlet_epsilon;
    for (int i = 0; i < n; i++) {
        root->children[i]->P = (1.0f - eps) * root->children[i]->P + eps * noise[i];
    }
}

void SelfPlayWorker::evaluate_leaf(const float* canonical_state,
                                    float* policy_out, float& value_out) {
    // Call predict_fn for a single state (dereference pointer)
    (*predict_fn_)(canonical_state, 1, game_.nn_input_size, policy_out, &value_out,
                   game_.n2 + 1);
}

void SelfPlayWorker::extract_policy(MCTSNode* root, float temperature, float* policy_out) {
    int action_size = game_.n2 + 1;
    std::memset(policy_out, 0, action_size * sizeof(float));

    for (int i = 0; i < root->num_children; i++) {
        policy_out[root->children[i]->action] = static_cast<float>(root->children[i]->N);
    }

    // Apply temperature
    float total = 0.0f;
    for (int i = 0; i < action_size; i++) total += policy_out[i];

    if (total > 0.0f) {
        if (temperature <= 0.01f) {
            // Greedy
            int best = 0;
            float best_val = policy_out[0];
            for (int i = 1; i < action_size; i++) {
                if (policy_out[i] > best_val) {
                    best_val = policy_out[i];
                    best = i;
                }
            }
            std::memset(policy_out, 0, action_size * sizeof(float));
            policy_out[best] = 1.0f;
        } else {
            float inv_temp = 1.0f / temperature;
            total = 0.0f;
            for (int i = 0; i < action_size; i++) {
                if (policy_out[i] > 0.0f) {
                    policy_out[i] = std::pow(policy_out[i], inv_temp);
                }
                total += policy_out[i];
            }
            if (total > 0.0f) {
                float inv = 1.0f / total;
                for (int i = 0; i < action_size; i++) policy_out[i] *= inv;
            }
        }
    }
}

void SelfPlayWorker::run_search(MCTSNode* root, int num_sims) {
    float fpu = config_.fpu_reduction;
    float root_fpu = (config_.root_fpu_reduction < 0.0f) ? fpu : config_.root_fpu_reduction;
    int action_size = game_.n2 + 1;

    // Allocate buffers for NN eval
    std::vector<float> canonical(game_.nn_input_size);
    std::vector<float> policy(action_size);
    std::vector<float> valid(action_size);

    int nn_batch = config_.nn_batch_size;

    if (nn_batch <= 1) {
        // Sequential search
        for (int sim = 0; sim < num_sims; sim++) {
            MCTSNode* node = root;

            // SELECT
            while (node->is_expanded && node->num_children > 0) {
                float f = (node->parent == nullptr) ? root_fpu : fpu;
                node = node->select_child(config_.c_puct, f, config_.c_puct_base);
            }

            ensure_state(node, game_, arena_, scratch_);

            // Terminal check
            if (node->parent != nullptr) {
                float terminal_value;
                if (game_.check_terminal(node->state, node->action, node->parent->player,
                                         terminal_value, scratch_)) {
                    node->backpropagate(-terminal_value);
                    continue;
                }
            }

            // EXPAND & EVALUATE
            game_.get_canonical_state(node->state, node->player, canonical.data());
            float value;
            evaluate_leaf(canonical.data(), policy.data(), value);
            game_.get_valid_moves(node->state, node->player, valid.data(), scratch_);
            expand_node(node, game_, policy.data(), valid.data(), arena_);
            node->backpropagate(value);
        }
    } else {
        // Batched search with virtual loss
        // Pre-allocate batch buffers outside loop to avoid repeated allocation
        std::vector<MCTSNode*> leaves;
        leaves.reserve(nn_batch);
        std::vector<MCTSNode*> unique_leaves;
        unique_leaves.reserve(nn_batch);
        std::vector<int> leaf_to_unique(nn_batch);  // maps leaf index -> unique index
        std::vector<float> batch_states(nn_batch * game_.nn_input_size);
        std::vector<float> batch_policies(nn_batch * action_size);
        std::vector<float> batch_values(nn_batch);

        int sims_done = 0;
        while (sims_done < num_sims) {
            int batch = std::min(nn_batch, num_sims - sims_done);

            // Phase 1: Select leaves with virtual loss
            leaves.clear();

            for (int b = 0; b < batch; b++) {
                MCTSNode* node = root;
                while (node->is_expanded && node->num_children > 0) {
                    float f = (node->parent == nullptr) ? root_fpu : fpu;
                    node = node->select_child(config_.c_puct, f, config_.c_puct_base);
                }

                ensure_state(node, game_, arena_, scratch_);

                if (node->parent != nullptr) {
                    float terminal_value;
                    if (game_.check_terminal(node->state, node->action, node->parent->player,
                                             terminal_value, scratch_)) {
                        node->backpropagate(-terminal_value);
                        sims_done++;
                        continue;
                    }
                }

                node->apply_virtual_loss();
                leaves.push_back(node);
            }

            if (leaves.empty()) continue;

            // Phase 2: Batch evaluate (deduplicated)
            // Build unique leaves using leaf_to_unique map (O(n) amortized)
            unique_leaves.clear();
            for (size_t li = 0; li < leaves.size(); li++) {
                MCTSNode* leaf = leaves[li];
                // Check if already in unique_leaves via linear scan
                // (batch sizes are small, typically <=64, so this is fast)
                int ui = -1;
                for (size_t j = 0; j < unique_leaves.size(); j++) {
                    if (unique_leaves[j] == leaf) { ui = static_cast<int>(j); break; }
                }
                if (ui < 0) {
                    ui = static_cast<int>(unique_leaves.size());
                    unique_leaves.push_back(leaf);
                }
                leaf_to_unique[li] = ui;
            }

            int ul_count = static_cast<int>(unique_leaves.size());

            // Build batch of canonical states
            for (int i = 0; i < ul_count; i++) {
                game_.get_canonical_state(unique_leaves[i]->state, unique_leaves[i]->player,
                                          batch_states.data() + i * game_.nn_input_size);
            }

            // Batch NN eval
            (*predict_fn_)(batch_states.data(), ul_count, game_.nn_input_size,
                          batch_policies.data(), batch_values.data(), action_size);

            // Phase 3: Revert VL, expand, backprop
            for (size_t li = 0; li < leaves.size(); li++) {
                MCTSNode* leaf = leaves[li];
                int ui = leaf_to_unique[li];

                leaf->revert_virtual_loss();

                if (!leaf->is_expanded) {
                    game_.get_valid_moves(leaf->state, leaf->player, valid.data(), scratch_);
                    expand_node(leaf, game_,
                                batch_policies.data() + ui * action_size,
                                valid.data(), arena_);
                }
                leaf->backpropagate(batch_values[ui]);
                sims_done++;
            }
        }
    }
}

SelfPlayWorker::GameResult SelfPlayWorker::play_game() {
    int action_size = game_.n2 + 1;
    int full_sims = config_.num_simulations;
    bool use_playout_cap = config_.playout_cap_prob < 1.0f;
    int cheap_sims = std::max(1, static_cast<int>(full_sims * config_.playout_cap_cheap_fraction));

    // Game state lives on the heap (NOT in the arena), so arena resets per move
    // don't destroy it.
    std::vector<float> game_state(game_.state_size);
    game_.get_initial_state(game_state.data());
    int player = 1;
    int move_count = 0;

    struct TrajectoryEntry {
        std::vector<float> canonical;
        int traj_player;
        std::vector<float> policy;
        bool is_full;
    };
    std::vector<TrajectoryEntry> trajectory;

    std::vector<float> canonical(game_.nn_input_size);
    std::vector<float> policy_buf(action_size);
    std::vector<float> valid_buf(action_size);
    std::vector<float> next_state(game_.state_size);

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    while (true) {
        // Reset arena each move — MCTS tree only lives for one move.
        // This prevents arena growth/realloc which would invalidate pointers.
        arena_.reset();

        game_.get_canonical_state(game_state.data(), player, canonical.data());

        // Decide full vs cheap
        bool is_full = !use_playout_cap || (uniform(rng_) < config_.playout_cap_prob);
        int sims = is_full ? full_sims : cheap_sims;

        // Temperature
        float temp;
        if (config_.temp_decay_halflife > 0) {
            float temp_start = config_.temperature;
            float temp_end = 0.1f;
            temp = temp_end + (temp_start - temp_end) *
                   std::pow(0.5f, static_cast<float>(move_count) / config_.temp_decay_halflife);
        } else if (move_count < config_.temp_threshold) {
            temp = 1.0f;
        } else {
            temp = 0.01f;
        }

        // Create root (in arena — will be freed on next move's reset)
        MCTSNode* root = arena_.alloc_node();
        root->state = arena_.alloc_state(game_.state_size);
        std::memcpy(root->state, game_state.data(), game_.state_size * sizeof(float));
        root->player = player;
        root->action = -1;
        root->parent = nullptr;

        // Get initial policy for root expansion
        float root_value;
        evaluate_leaf(canonical.data(), policy_buf.data(), root_value);
        game_.get_valid_moves(game_state.data(), player, valid_buf.data(), scratch_);
        expand_node(root, game_, policy_buf.data(), valid_buf.data(), arena_);

        // Add Dirichlet noise
        add_dirichlet_noise(root);

        // Run search
        run_search(root, sims);

        // Extract policy
        std::vector<float> pi(action_size);
        extract_policy(root, temp, pi.data());

        // Record trajectory
        trajectory.push_back({
            std::vector<float>(canonical.data(), canonical.data() + game_.nn_input_size),
            player,
            pi,
            is_full,
        });

        // Sample action
        std::discrete_distribution<int> dist(pi.begin(), pi.end());
        int action = dist(rng_);

        // Apply action — write to heap-allocated next_state, then swap
        game_.get_next_state(game_state.data(), action, player, next_state.data(), scratch_);
        game_state.swap(next_state);
        move_count++;

        // Terminal check
        float terminal_value;
        if (game_.check_terminal(game_state.data(), action, player, terminal_value, scratch_)) {
            // Determine outcome
            int outcome;
            if (terminal_value == 0.0f) {
                outcome = 0;
            } else {
                outcome = (terminal_value > 0.0f) ? player : -player;
            }

            // Build examples
            std::vector<Example> examples;
            for (auto& entry : trajectory) {
                if (use_playout_cap && !entry.is_full) continue;

                float v;
                if (entry.traj_player == player) {
                    v = terminal_value;
                } else {
                    v = -terminal_value;
                }
                examples.push_back({entry.canonical, entry.policy, v});
            }

            return {std::move(examples), outcome, move_count};
        }

        player = -player;
    }
}

// --- Multi-threaded generation ---

std::pair<std::vector<Example>, GameStats>
generate_self_play_data(int board_size, int num_games, const MCTSCppConfig& config,
                        const PredictFn& predict_fn, int num_threads) {
    GoGame game(board_size);

    std::atomic<int> games_remaining(num_games);
    std::mutex results_mutex;
    std::vector<Example> all_examples;
    GameStats stats;
    std::vector<int> game_lengths;

    // Keep predict_fn on the main thread's stack; workers access it via pointer
    // to avoid copying the std::function (which may contain Python objects).
    auto worker_fn = [&](int worker_id) {
        SelfPlayWorker worker(game, config, worker_id,
                              static_cast<unsigned>(42 + worker_id));
        worker.set_predict_fn(&predict_fn);

        while (true) {
            int remaining = games_remaining.fetch_sub(1);
            if (remaining <= 0) {
                games_remaining.fetch_add(1); // undo
                break;
            }

            auto result = worker.play_game();

            std::lock_guard<std::mutex> lock(results_mutex);

            if (result.outcome == 1) stats.p1_wins++;
            else if (result.outcome == -1) stats.p2_wins++;
            else stats.draws++;

            game_lengths.push_back(result.game_length);
            all_examples.insert(all_examples.end(),
                                std::make_move_iterator(result.examples.begin()),
                                std::make_move_iterator(result.examples.end()));
        }
    };

    if (num_threads <= 1) {
        // Single-threaded: run on calling thread (avoids GIL issues with std::thread)
        worker_fn(0);
    } else {
        // Multi-threaded: launch worker threads
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(worker_fn, i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    // Compute stats
    if (!game_lengths.empty()) {
        float sum = 0.0f;
        for (int len : game_lengths) sum += len;
        stats.mean_game_length = sum / game_lengths.size();
    }

    return {std::move(all_examples), stats};
}
