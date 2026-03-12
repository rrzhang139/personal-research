#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/function.h>

#include "go_game.h"
#include "mcts_node.h"
#include "self_play.h"

namespace nb = nanobind;
using namespace nb::literals;

// Helper: create a 1D numpy float32 array from raw data, taking ownership
static nb::ndarray<nb::numpy, float, nb::ndim<1>>
make_numpy_1d(float* data, size_t n) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(data, {n}, owner);
}

NB_MODULE(_mcts_cpp, m) {
    m.doc() = "C++ MCTS engine with nanobind bindings";

    // --- GoGame ---
    nb::class_<GoGame>(m, "GoGame")
        .def(nb::init<int>(), "size"_a = 9)
        .def_ro("size", &GoGame::size)
        .def_ro("n2", &GoGame::n2)
        .def_ro("nn_input_size", &GoGame::nn_input_size)
        .def_ro("pass_action", &GoGame::pass_action)
        .def_ro("state_size", &GoGame::state_size)
        .def("get_initial_state", [](const GoGame& g) {
            float* data = new float[g.state_size];
            g.get_initial_state(data);
            return make_numpy_1d(data, g.state_size);
        })
        .def("get_next_state", [](const GoGame& g,
                                   nb::ndarray<nb::numpy, const float, nb::ndim<1>> state,
                                   int action, int player) {
            float* out = new float[g.state_size];
            g.get_next_state(state.data(), action, player, out);
            return make_numpy_1d(out, g.state_size);
        }, "state"_a, "action"_a, "player"_a)
        .def("get_valid_moves", [](const GoGame& g,
                                    nb::ndarray<nb::numpy, const float, nb::ndim<1>> state,
                                    int player) {
            int action_size = g.n2 + 1;
            float* valid = new float[action_size];
            FloodFillScratch scratch;
            std::memset(scratch.visited, 0, g.n2);
            g.get_valid_moves(state.data(), player, valid, scratch);
            return make_numpy_1d(valid, action_size);
        }, "state"_a, "player"_a = 1)
        .def("check_terminal", [](const GoGame& g,
                                   nb::ndarray<nb::numpy, const float, nb::ndim<1>> state,
                                   int action, int player) {
            float value;
            bool is_terminal = g.check_terminal(state.data(), action, player, value);
            return std::make_pair(is_terminal, value);
        }, "state"_a, "action"_a, "player"_a = 1)
        .def("get_canonical_state", [](const GoGame& g,
                                        nb::ndarray<nb::numpy, const float, nb::ndim<1>> state,
                                        int player) {
            float* out = new float[g.nn_input_size];
            g.get_canonical_state(state.data(), player, out);
            return make_numpy_1d(out, g.nn_input_size);
        }, "state"_a, "player"_a)
        .def("get_board_size", [](const GoGame& g) { return g.nn_input_size; })
        .def("get_action_size", [](const GoGame& g) { return g.n2 + 1; })
        .def("get_board_shape", [](const GoGame& g) {
            return std::make_tuple(g.num_planes, g.size, g.size);
        });

    // --- MCTSCppConfig ---
    nb::class_<MCTSCppConfig>(m, "MCTSConfig")
        .def(nb::init<>())
        .def_rw("num_simulations", &MCTSCppConfig::num_simulations)
        .def_rw("c_puct", &MCTSCppConfig::c_puct)
        .def_rw("dirichlet_alpha", &MCTSCppConfig::dirichlet_alpha)
        .def_rw("dirichlet_epsilon", &MCTSCppConfig::dirichlet_epsilon)
        .def_rw("temperature", &MCTSCppConfig::temperature)
        .def_rw("temp_threshold", &MCTSCppConfig::temp_threshold)
        .def_rw("temp_decay_halflife", &MCTSCppConfig::temp_decay_halflife)
        .def_rw("nn_batch_size", &MCTSCppConfig::nn_batch_size)
        .def_rw("playout_cap_prob", &MCTSCppConfig::playout_cap_prob)
        .def_rw("playout_cap_cheap_fraction", &MCTSCppConfig::playout_cap_cheap_fraction)
        .def_rw("fpu_reduction", &MCTSCppConfig::fpu_reduction)
        .def_rw("root_fpu_reduction", &MCTSCppConfig::root_fpu_reduction)
        .def_rw("c_puct_base", &MCTSCppConfig::c_puct_base);

    // --- GameStats ---
    nb::class_<GameStats>(m, "GameStats")
        .def(nb::init<>())
        .def_ro("p1_wins", &GameStats::p1_wins)
        .def_ro("p2_wins", &GameStats::p2_wins)
        .def_ro("draws", &GameStats::draws)
        .def_ro("mean_game_length", &GameStats::mean_game_length)
        .def_ro("mean_root_value", &GameStats::mean_root_value)
        .def_ro("mean_policy_entropy", &GameStats::mean_policy_entropy)
        .def_ro("mean_search_depth", &GameStats::mean_search_depth);

    // --- Example ---
    nb::class_<Example>(m, "Example")
        .def_ro("value", &Example::value)
        .def("get_state", [](const Example& ex) {
            size_t n = ex.state.size();
            float* data = new float[n];
            std::memcpy(data, ex.state.data(), n * sizeof(float));
            return make_numpy_1d(data, n);
        })
        .def("get_policy", [](const Example& ex) {
            size_t n = ex.policy.size();
            float* data = new float[n];
            std::memcpy(data, ex.policy.data(), n * sizeof(float));
            return make_numpy_1d(data, n);
        });

    // --- test_cpp_only: pure C++ self-play (no Python callback) ---
    m.def("test_cpp_only",
        [](int board_size, int num_games, int num_sims) {
            GoGame game(board_size);
            MCTSCppConfig config;
            config.num_simulations = num_sims;
            config.nn_batch_size = 1;
            config.dirichlet_epsilon = 0.0f;  // no noise

            int action_size = board_size * board_size + 1;

            // Pure C++ uniform predict — no Python, no GIL
            PredictFn predict_fn = [action_size](
                const float*, int batch_size, int,
                float* policies_out, float* values_out, int) {
                for (int b = 0; b < batch_size; b++) {
                    float inv = 1.0f / action_size;
                    for (int i = 0; i < action_size; i++)
                        policies_out[b * action_size + i] = inv;
                    values_out[b] = 0.0f;
                }
            };

            auto [examples, stats] = generate_self_play_data(
                board_size, num_games, config, predict_fn, 1);

            return static_cast<int>(examples.size());
        },
        "board_size"_a, "num_games"_a = 1, "num_sims"_a = 5,
        "Test C++ self-play with uniform predict (no Python callback)."
    );

    // --- generate_self_play_data ---
    m.def("generate_self_play_data",
        [](int board_size, int num_games,
           MCTSCppConfig config,
           nb::object predict_fn_py,
           int num_threads) {

            int nn_input_size = (2 * NUM_HISTORY + 1) * board_size * board_size;
            int action_size = board_size * board_size + 1;

            // Wrap Python callable into C++ PredictFn.
            // We must copy data into owned numpy arrays (not views) because
            // nanobind ndarray with null owner is unreliable across platforms.
            PredictFn predict_fn = [predict_fn_py, nn_input_size, action_size]
                (const float* states, int batch_size, int nn_sz,
                 float* policies_out, float* values_out, int act_sz) {

                nb::gil_scoped_acquire gil;

                // Create an owned numpy array by copying the C++ data
                size_t total = (size_t)batch_size * (size_t)nn_input_size;
                float* data_copy = new float[total];
                std::memcpy(data_copy, states, total * sizeof(float));
                nb::capsule owner(data_copy, [](void* p) noexcept {
                    delete[] static_cast<float*>(p);
                });
                size_t state_shape[] = {(size_t)batch_size, (size_t)nn_input_size};
                auto states_arr = nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                    data_copy, 2, state_shape, owner);

                nb::object result = predict_fn_py(states_arr);
                nb::tuple result_tuple = nb::cast<nb::tuple>(result);

                auto policies_nb = nb::cast<nb::ndarray<nb::numpy, float, nb::ndim<2>>>(
                    result_tuple[0]);
                auto values_nb = nb::cast<nb::ndarray<nb::numpy, float, nb::ndim<1>>>(
                    result_tuple[1]);

                std::memcpy(policies_out, policies_nb.data(),
                            batch_size * action_size * sizeof(float));
                std::memcpy(values_out, values_nb.data(), batch_size * sizeof(float));
            };

            nb::gil_scoped_release release;

            auto [examples, stats] = generate_self_play_data(
                board_size, num_games, config, predict_fn, num_threads);

            return std::make_pair(std::move(examples), stats);
        },
        "board_size"_a, "num_games"_a, "config"_a, "predict_fn"_a, "num_threads"_a = 4,
        "Generate self-play training data with C++ MCTS engine."
    );
}
