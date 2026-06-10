#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

static const int DEMO_DIMS = 16;

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

inline float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline float cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0;
    float normA = 0;
    float normB = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA < 1e-9f || normB < 1e-9f) {
        return 1.0f;
    }
    return 1.0f - dot / (std::sqrt(normA) * std::sqrt(normB));
}

inline float manhattanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

inline DistFn getDistanceFn(const std::string& metric) {
    if (metric == "cosine") {
        return cosineDistance;
    }
    if (metric == "manhattan") {
        return manhattanDistance;
    }
    return euclideanDistance;
}

class BruteForceIndex {
public:
    void insert(const VectorItem& v) {
        items.push_back(v);
    }

    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, DistFn dist) const {
        std::vector<std::pair<float, int>> result;
        result.reserve(items.size());
        for (const auto& v : items) {
            result.push_back({dist(q, v.emb), v.id});
        }
        std::sort(result.begin(), result.end());
        if ((int)result.size() > k) {
            result.resize(k);
        }
        return result;
    }

    void remove(int id) {
        items.erase(
            std::remove_if(items.begin(), items.end(), [id](const VectorItem& v) { return v.id == id; }),
            items.end());
    }

    void clear() {
        items.clear();
    }

private:
    std::vector<VectorItem> items;
};

struct KDNode {
    VectorItem item;
    KDNode* left = nullptr;
    KDNode* right = nullptr;

    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTreeIndex {
public:
    explicit KDTreeIndex(int dims) : dims_(dims) {}

    ~KDTreeIndex() {
        destroy(root_);
    }

    KDTreeIndex(const KDTreeIndex&) = delete;
    KDTreeIndex& operator=(const KDTreeIndex&) = delete;

    void insert(const VectorItem& v) {
        root_ = insertNode(root_, v, 0);
    }

    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, DistFn dist) const {
        std::priority_queue<std::pair<float, int>> heap;
        search(root_, q, k, 0, dist, heap);

        std::vector<std::pair<float, int>> result;
        while (!heap.empty()) {
            result.push_back(heap.top());
            heap.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root_);
        root_ = nullptr;
        for (const auto& item : items) {
            insert(item);
        }
    }

private:
    KDNode* root_ = nullptr;
    int dims_;

    static void destroy(KDNode* n) {
        if (!n) {
            return;
        }
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    KDNode* insertNode(KDNode* n, const VectorItem& v, int depth) {
        if (!n) {
            return new KDNode(v);
        }
        const int axis = depth % dims_;
        if (v.emb[axis] < n->item.emb[axis]) {
            n->left = insertNode(n->left, v, depth + 1);
        } else {
            n->right = insertNode(n->right, v, depth + 1);
        }
        return n;
    }

    void search(
        KDNode* n,
        const std::vector<float>& q,
        int k,
        int depth,
        DistFn dist,
        std::priority_queue<std::pair<float, int>>& heap) const
    {
        if (!n) {
            return;
        }

        const float nodeDist = dist(q, n->item.emb);
        if ((int)heap.size() < k || nodeDist < heap.top().first) {
            heap.push({nodeDist, n->item.id});
            if ((int)heap.size() > k) {
                heap.pop();
            }
        }

        const int axis = depth % dims_;
        const float diff = q[axis] - n->item.emb[axis];
        KDNode* closer = diff < 0 ? n->left : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        search(closer, q, k, depth + 1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first) {
            search(farther, q, k, depth + 1, dist, heap);
        }
    }
};

class HNSWIndex {
public:
    struct GraphInfo {
        int topLayer;
        int nodeCount;
        std::vector<int> nodesPerLayer;
        std::vector<int> edgesPerLayer;

        struct NodeView {
            int id;
            std::string metadata;
            std::string category;
            int maxLayer;
        };

        struct EdgeView {
            int src;
            int dst;
            int layer;
        };

        std::vector<NodeView> nodes;
        std::vector<EdgeView> edges;
    };

    HNSWIndex(int m = 16, int efBuild = 200)
        : maxNeighbors_(m),
          maxNeighbors0_(2 * m),
          efBuild_(efBuild),
          levelMult_(1.0f / std::log((float)m)),
          rng_(42)
    {
    }

    void insert(const VectorItem& item, DistFn dist) {
        const int id = item.id;
        const int level = randomLevel();
        graph_[id] = {item, level, std::vector<std::vector<int>>(level + 1)};

        if (entryPoint_ == -1) {
            entryPoint_ = id;
            topLayer_ = level;
            return;
        }

        int ep = entryPoint_;
        for (int layer = topLayer_; layer > level; layer--) {
            if (layer < (int)graph_[ep].neighbors.size()) {
                auto nearest = searchLayer(item.emb, ep, 1, layer, dist);
                if (!nearest.empty()) {
                    ep = nearest[0].second;
                }
            }
        }

        for (int layer = std::min(topLayer_, level); layer >= 0; layer--) {
            auto candidates = searchLayer(item.emb, ep, efBuild_, layer, dist);
            const int limit = layer == 0 ? maxNeighbors0_ : maxNeighbors_;
            auto selected = selectNeighbors(candidates, limit);
            graph_[id].neighbors[layer] = selected;

            for (int neighborId : selected) {
                if (!graph_.count(neighborId)) {
                    continue;
                }
                if ((int)graph_[neighborId].neighbors.size() <= layer) {
                    graph_[neighborId].neighbors.resize(layer + 1);
                }
                auto& connections = graph_[neighborId].neighbors[layer];
                connections.push_back(id);
                if ((int)connections.size() > limit) {
                    std::vector<std::pair<float, int>> distances;
                    for (int candidate : connections) {
                        if (graph_.count(candidate)) {
                            distances.push_back({
                                dist(graph_[neighborId].item.emb, graph_[candidate].item.emb),
                                candidate});
                        }
                    }
                    std::sort(distances.begin(), distances.end());
                    connections.clear();
                    for (int i = 0; i < limit && i < (int)distances.size(); i++) {
                        connections.push_back(distances[i].second);
                    }
                }
            }
            if (!candidates.empty()) {
                ep = candidates[0].second;
            }
        }

        if (level > topLayer_) {
            topLayer_ = level;
            entryPoint_ = id;
        }
    }

    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q,
        int k,
        int ef,
        DistFn dist) const
    {
        if (entryPoint_ == -1) {
            return {};
        }

        int ep = entryPoint_;
        for (int layer = topLayer_; layer > 0; layer--) {
            if (layer < (int)graph_.at(ep).neighbors.size()) {
                auto nearest = searchLayer(q, ep, 1, layer, dist);
                if (!nearest.empty()) {
                    ep = nearest[0].second;
                }
            }
        }

        auto result = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)result.size() > k) {
            result.resize(k);
        }
        return result;
    }

    void remove(int id) {
        if (!graph_.count(id)) {
            return;
        }

        for (auto& kv : graph_) {
            for (auto& layer : kv.second.neighbors) {
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
            }
        }

        if (entryPoint_ == id) {
            entryPoint_ = -1;
            for (const auto& kv : graph_) {
                if (kv.first != id) {
                    entryPoint_ = kv.first;
                    break;
                }
            }
        }
        graph_.erase(id);
        recomputeTopLayer();
    }

    void clear() {
        graph_.clear();
        topLayer_ = -1;
        entryPoint_ = -1;
    }

    GraphInfo getInfo() const {
        GraphInfo info;
        info.topLayer = topLayer_;
        info.nodeCount = (int)graph_.size();
        const int layerCount = std::max(topLayer_ + 1, 1);
        info.nodesPerLayer.assign(layerCount, 0);
        info.edgesPerLayer.assign(layerCount, 0);

        for (const auto& kv : graph_) {
            const int id = kv.first;
            const auto& node = kv.second;
            info.nodes.push_back({id, node.item.metadata, node.item.category, node.maxLayer});
            for (int layer = 0; layer <= node.maxLayer && layer < layerCount; layer++) {
                info.nodesPerLayer[layer]++;
                if (layer < (int)node.neighbors.size()) {
                    for (int neighbor : node.neighbors[layer]) {
                        if (id < neighbor) {
                            info.edgesPerLayer[layer]++;
                            info.edges.push_back({id, neighbor, layer});
                        }
                    }
                }
            }
        }
        return info;
    }

private:
    struct Node {
        VectorItem item;
        int maxLayer;
        std::vector<std::vector<int>> neighbors;
    };

    std::unordered_map<int, Node> graph_;
    int maxNeighbors_;
    int maxNeighbors0_;
    int efBuild_;
    float levelMult_;
    int topLayer_ = -1;
    int entryPoint_ = -1;
    mutable std::mt19937 rng_;

    int randomLevel() {
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        return (int)std::floor(-std::log(uniform(rng_)) * levelMult_);
    }

    std::vector<std::pair<float, int>> searchLayer(
        const std::vector<float>& q,
        int entryPoint,
        int ef,
        int layer,
        DistFn dist) const
    {
        std::unordered_map<int, bool> visited;
        std::priority_queue<
            std::pair<float, int>,
            std::vector<std::pair<float, int>>,
            std::greater<>> candidates;
        std::priority_queue<std::pair<float, int>> found;

        const float entryDist = dist(q, graph_.at(entryPoint).item.emb);
        visited[entryPoint] = true;
        candidates.push({entryDist, entryPoint});
        found.push({entryDist, entryPoint});

        while (!candidates.empty()) {
            auto [candidateDist, candidateId] = candidates.top();
            candidates.pop();
            if ((int)found.size() >= ef && candidateDist > found.top().first) {
                break;
            }
            if (layer >= (int)graph_.at(candidateId).neighbors.size()) {
                continue;
            }
            for (int neighborId : graph_.at(candidateId).neighbors[layer]) {
                if (visited[neighborId] || !graph_.count(neighborId)) {
                    continue;
                }
                visited[neighborId] = true;
                const float neighborDist = dist(q, graph_.at(neighborId).item.emb);
                if ((int)found.size() < ef || neighborDist < found.top().first) {
                    candidates.push({neighborDist, neighborId});
                    found.push({neighborDist, neighborId});
                    if ((int)found.size() > ef) {
                        found.pop();
                    }
                }
            }
        }

        std::vector<std::pair<float, int>> result;
        while (!found.empty()) {
            result.push_back(found.top());
            found.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    static std::vector<int> selectNeighbors(
        const std::vector<std::pair<float, int>>& candidates,
        int limit)
    {
        std::vector<int> result;
        for (int i = 0; i < std::min((int)candidates.size(), limit); i++) {
            result.push_back(candidates[i].second);
        }
        return result;
    }

    void recomputeTopLayer() {
        topLayer_ = -1;
        for (const auto& kv : graph_) {
            topLayer_ = std::max(topLayer_, kv.second.maxLayer);
        }
    }
};

class VectorDB {
public:
    struct Hit {
        int id;
        std::string metadata;
        std::string category;
        std::vector<float> embedding;
        float distance;
    };

    struct SearchResult {
        std::vector<Hit> hits;
        long long latencyUs;
        std::string algorithm;
        std::string metric;
    };

    struct BenchmarkResult {
        long long bruteForceUs;
        long long kdTreeUs;
        long long hnswUs;
        int itemCount;
    };

    explicit VectorDB(int dims) : kdTree_(dims), hnsw_(16, 200), dims(dims) {}

    int insert(
        const std::string& metadata,
        const std::string& category,
        const std::vector<float>& embedding,
        DistFn dist)
    {
        std::lock_guard<std::mutex> lock(mu_);
        VectorItem item{nextId_++, metadata, category, embedding};
        store_[item.id] = item;
        bruteForce_.insert(item);
        kdTree_.insert(item);
        hnsw_.insert(item, dist);
        return item.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!store_.count(id)) {
            return false;
        }
        store_.erase(id);
        bruteForce_.remove(id);
        hnsw_.remove(id);

        std::vector<VectorItem> remaining;
        for (const auto& kv : store_) {
            remaining.push_back(kv.second);
        }
        kdTree_.rebuild(remaining);
        return true;
    }

    SearchResult search(
        const std::vector<float>& q,
        int k,
        const std::string& metric,
        const std::string& algorithm) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto dist = getDistanceFn(metric);
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float, int>> raw;
        if (algorithm == "bruteforce") {
            raw = bruteForce_.knn(q, k, dist);
        } else if (algorithm == "kdtree") {
            raw = kdTree_.knn(q, k, dist);
        } else {
            raw = hnsw_.knn(q, k, 50, dist);
        }

        const long long elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchResult result;
        result.latencyUs = elapsedUs;
        result.algorithm = algorithm;
        result.metric = metric;
        for (const auto& pair : raw) {
            const float distance = pair.first;
            const int id = pair.second;
            auto it = store_.find(id);
            if (it != store_.end()) {
                result.hits.push_back({
                    id,
                    it->second.metadata,
                    it->second.category,
                    it->second.emb,
                    distance});
            }
        }
        return result;
    }

    BenchmarkResult benchmark(const std::vector<float>& q, int k, const std::string& metric) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto dist = getDistanceFn(metric);
        auto time = [](auto fn) -> long long {
            const auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
        };

        return {
            time([&] { bruteForce_.knn(q, k, dist); }),
            time([&] { kdTree_.knn(q, k, dist); }),
            time([&] { hnsw_.knn(q, k, 50, dist); }),
            (int)store_.size()};
    }

    std::vector<VectorItem> all() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<VectorItem> result;
        for (const auto& kv : store_) {
            result.push_back(kv.second);
        }
        return result;
    }

    HNSWIndex::GraphInfo hnswInfo() const {
        std::lock_guard<std::mutex> lock(mu_);
        return hnsw_.getInfo();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return store_.size();
    }

    const int dims;

private:
    std::unordered_map<int, VectorItem> store_;
    BruteForceIndex bruteForce_;
    KDTreeIndex kdTree_;
    HNSWIndex hnsw_;
    mutable std::mutex mu_;
    int nextId_ = 1;
};
