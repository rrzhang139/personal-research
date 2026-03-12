#include "mcts_node.h"
#include "go_game.h"
#include <cstdlib>
#include <algorithm>

// --- MCTSNode ---

MCTSNode* MCTSNode::select_child(float c_puct, float fpu_reduction, float c_puct_base) const {
    if (num_children == 0) return nullptr;

    float sqrt_parent = std::sqrt(static_cast<float>(N));

    if (c_puct_base > 0.0f) {
        c_puct = c_puct * (std::log((N + c_puct_base + 1.0f) / c_puct_base) + 1.0f);
    }

    float fpu_value = 0.0f;
    if (fpu_reduction > 0.0f && N > 0) {
        fpu_value = Q() - fpu_reduction;
    }

    float best_score = -1e18f;
    MCTSNode* best_child = children[0];

    for (int i = 0; i < num_children; i++) {
        MCTSNode* child = children[i];
        float exploit;
        if (child->N > 0) {
            exploit = -child->W / child->N;
        } else {
            exploit = fpu_value;
        }
        float explore = c_puct * child->P * sqrt_parent / (1.0f + child->N);
        float score = exploit + explore;
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }

    return best_child;
}

void MCTSNode::backpropagate(float value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->N += 1;
        node->W += value;
        value = -value;
        node = node->parent;
    }
}

void MCTSNode::apply_virtual_loss() {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->N += 1;
        node->W += 1.0f;
        node = node->parent;
    }
}

void MCTSNode::revert_virtual_loss() {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->N -= 1;
        node->W -= 1.0f;
        node = node->parent;
    }
}

// --- NodeArena ---

NodeArena::NodeArena(size_t capacity_bytes)
    : capacity_(capacity_bytes), offset_(0) {
    data_ = static_cast<char*>(std::malloc(capacity_bytes));
}

NodeArena::~NodeArena() {
    std::free(data_);
}

void NodeArena::reset() {
    offset_ = 0;
}

void* NodeArena::alloc(size_t bytes, size_t align) {
    // Align offset
    size_t mask = align - 1;
    offset_ = (offset_ + mask) & ~mask;

    if (offset_ + bytes > capacity_) {
        // Grow arena (double size)
        size_t new_cap = std::max(capacity_ * 2, offset_ + bytes);
        data_ = static_cast<char*>(std::realloc(data_, new_cap));
        capacity_ = new_cap;
    }

    void* ptr = data_ + offset_;
    offset_ += bytes;
    return ptr;
}

MCTSNode* NodeArena::alloc_node() {
    MCTSNode* node = static_cast<MCTSNode*>(alloc(sizeof(MCTSNode), alignof(MCTSNode)));
    std::memset(node, 0, sizeof(MCTSNode));
    return node;
}

MCTSNode** NodeArena::alloc_children(int count) {
    return static_cast<MCTSNode**>(alloc(count * sizeof(MCTSNode*), alignof(MCTSNode*)));
}

float* NodeArena::alloc_state(int num_floats) {
    return static_cast<float*>(alloc(num_floats * sizeof(float), alignof(float)));
}

// --- expand_node ---

void expand_node(MCTSNode* node, const GoGame& game, const float* action_priors,
                 const float* valid_moves, NodeArena& arena) {
    int action_size = game.n2 + 1;

    // Mask and normalize priors
    float masked[362]; // max 19*19+1
    float prior_sum = 0.0f;
    for (int i = 0; i < action_size; i++) {
        masked[i] = action_priors[i] * valid_moves[i];
        prior_sum += masked[i];
    }
    if (prior_sum > 0.0f) {
        float inv = 1.0f / prior_sum;
        for (int i = 0; i < action_size; i++) masked[i] *= inv;
    } else {
        float valid_sum = 0.0f;
        for (int i = 0; i < action_size; i++) valid_sum += valid_moves[i];
        if (valid_sum > 0.0f) {
            float inv = 1.0f / valid_sum;
            for (int i = 0; i < action_size; i++) masked[i] = valid_moves[i] * inv;
        }
    }

    // Count children (valid moves with sufficient prior)
    int count = 0;
    for (int i = 0; i < action_size; i++) {
        if (valid_moves[i] > 0.0f && masked[i] > 1e-6f) count++;
    }

    if (count == 0) {
        node->children = nullptr;
        node->num_children = 0;
        node->is_expanded = true;
        return;
    }

    MCTSNode** child_ptrs = arena.alloc_children(count);
    int ci = 0;
    int next_player = -node->player;

    for (int i = 0; i < action_size; i++) {
        if (valid_moves[i] <= 0.0f || masked[i] <= 1e-6f) continue;

        MCTSNode* child = arena.alloc_node();
        child->state = nullptr;  // lazy
        child->player = next_player;
        child->action = i;
        child->N = 0;
        child->W = 0.0f;
        child->P = masked[i];
        child->parent = node;
        child->children = nullptr;
        child->num_children = 0;
        child->is_expanded = false;

        child_ptrs[ci++] = child;
    }

    node->children = child_ptrs;
    node->num_children = count;
    node->is_expanded = true;
}

// --- ensure_state ---

void ensure_state(MCTSNode* node, const GoGame& game, NodeArena& arena,
                  FloodFillScratch& scratch) {
    if (node->state == nullptr && node->parent != nullptr && node->parent->state != nullptr) {
        node->state = arena.alloc_state(game.state_size);
        game.get_next_state(node->parent->state, node->action, node->parent->player,
                            node->state, scratch);
    }
}
