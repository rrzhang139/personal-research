#pragma once
#include <cmath>
#include <cstring>
#include <cstdint>
#include <atomic>

// Forward declare
class GoGame;

struct MCTSNode {
    float* state;          // nullptr for lazy expansion (allocated from arena)
    int player;
    int action;
    int N;                 // visit count
    float W;               // total value
    float P;               // prior probability
    MCTSNode* parent;
    MCTSNode** children;
    int num_children;
    bool is_expanded;

    float Q() const { return N > 0 ? W / N : 0.0f; }

    MCTSNode* select_child(float c_puct, float fpu_reduction, float c_puct_base) const;

    void backpropagate(float value);
    void apply_virtual_loss();
    void revert_virtual_loss();
};

// Bump-pointer arena allocator for MCTSNode and state arrays.
// Reset between games in O(1). Each worker thread owns one arena.
class NodeArena {
public:
    NodeArena(size_t capacity_bytes = 4 * 1024 * 1024);  // 4MB default
    ~NodeArena();

    // Allocate a new MCTSNode (zeroed)
    MCTSNode* alloc_node();

    // Allocate an array of MCTSNode* (for children pointers)
    MCTSNode** alloc_children(int count);

    // Allocate a float array (for state storage)
    float* alloc_state(int num_floats);

    // Reset arena (O(1) — just moves pointer back to start)
    void reset();

    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }

private:
    char* data_;
    size_t capacity_;
    size_t offset_;

    void* alloc(size_t bytes, size_t align);
};

// Expand a node: create children for legal actions with sufficient prior.
// action_priors: array of size action_size (n2+1).
// valid_moves: array of size action_size (from game.get_valid_moves).
void expand_node(MCTSNode* node, const GoGame& game, const float* action_priors,
                 const float* valid_moves, NodeArena& arena);

// Lazy state computation: compute state from parent if nullptr
void ensure_state(MCTSNode* node, const GoGame& game, NodeArena& arena,
                  struct FloodFillScratch& scratch);
