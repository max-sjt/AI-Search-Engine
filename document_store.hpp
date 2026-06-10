#pragma once

#include "vector_index.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

struct DocItem {
    int id;
    std::string title;
    std::string text;
    std::vector<float> embedding;
};

class DocumentStore {
public:
    explicit DocumentStore(const std::string& path = "data/documents.tsv")
        : path_(path), hnsw_(16, 200)
    {
        load();
    }

    int insert(const std::string& title, const std::string& text, const std::vector<float>& embedding) {
        if (dims_ == 0) {
            dims_ = (int)embedding.size();
        }
        DocItem item{nextId_++, title, text, embedding};
        insertUnlocked(item);
        saveUnlocked();
        return item.id;
    }

    bool remove(int id) {
        if (!store_.count(id)) {
            return false;
        }
        store_.erase(id);
        rebuildUnlocked();
        saveUnlocked();
        return true;
    }

    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q,
        int k,
        float maxDistance = 0.7f) const
    {
        if (store_.empty()) {
            return {};
        }
        auto raw = store_.size() < 10
            ? bruteForce_.knn(q, k, cosineDistance)
            : hnsw_.knn(q, k, 50, cosineDistance);

        std::vector<std::pair<float, DocItem>> out;
        for (const auto& pair : raw) {
            const float distance = pair.first;
            const int id = pair.second;
            auto it = store_.find(id);
            if (it != store_.end() && distance <= maxDistance) {
                out.push_back({distance, it->second});
            }
        }
        return out;
    }

    std::vector<DocItem> all() const {
        std::vector<DocItem> docs;
        for (const auto& kv : store_) {
            docs.push_back(kv.second);
        }
        std::sort(docs.begin(), docs.end(), [](const DocItem& a, const DocItem& b) {
            return a.id < b.id;
        });
        return docs;
    }

    size_t size() const {
        return store_.size();
    }

    int dims() const {
        return dims_;
    }

private:
    std::string path_;
    std::unordered_map<int, DocItem> store_;
    HNSWIndex hnsw_;
    BruteForceIndex bruteForce_;
    int nextId_ = 1;
    int dims_ = 0;

    void insertUnlocked(const DocItem& item) {
        store_[item.id] = item;
        VectorItem vectorItem{item.id, item.title, "doc", item.embedding};
        bruteForce_.insert(vectorItem);
        if (store_.size() == 10) {
            rebuildHnswUnlocked();
        } else if (store_.size() > 10) {
            hnsw_.insert(vectorItem, cosineDistance);
        }
        nextId_ = std::max(nextId_, item.id + 1);
        if (dims_ == 0) {
            dims_ = (int)item.embedding.size();
        }
    }

    void rebuildUnlocked() {
        bruteForce_.clear();
        dims_ = 0;
        for (const auto& kv : store_) {
            const auto& item = kv.second;
            VectorItem vectorItem{item.id, item.title, "doc", item.embedding};
            bruteForce_.insert(vectorItem);
            if (dims_ == 0) {
                dims_ = (int)item.embedding.size();
            }
        }
        rebuildHnswUnlocked();
    }

    void rebuildHnswUnlocked() {
        hnsw_.clear();
        if (store_.size() < 10) {
            return;
        }
        for (const auto& kv : store_) {
            const auto& item = kv.second;
            VectorItem vectorItem{item.id, item.title, "doc", item.embedding};
            hnsw_.insert(vectorItem, cosineDistance);
        }
    }

    void load() {
        FILE* file = std::fopen(path_.c_str(), "r");
        if (!file) {
            return;
        }

        char buffer[65536];
        while (std::fgets(buffer, sizeof(buffer), file)) {
            std::string line(buffer);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }
            auto fields = splitTsv(line);
            if (fields.size() != 4) {
                continue;
            }
            try {
                DocItem item{
                    std::stoi(fields[0]),
                    unescapeField(fields[1]),
                    unescapeField(fields[2]),
                    parseEmbedding(fields[3])};
                if (!item.embedding.empty()) {
                    insertUnlocked(item);
                }
            } catch (...) {
            }
        }
        std::fclose(file);
    }

    void saveUnlocked() const {
        ensureParentDirectory(path_);
        FILE* file = std::fopen(path_.c_str(), "w");
        if (!file) {
            return;
        }
        for (const auto& kv : store_) {
            const auto& item = kv.second;
            const std::string line =
                std::to_string(item.id) + "\t" +
                escapeField(item.title) + "\t" +
                escapeField(item.text) + "\t" +
                formatEmbedding(item.embedding) + "\n";
            std::fwrite(line.data(), 1, line.size(), file);
        }
        std::fclose(file);
    }

    static void ensureParentDirectory(const std::string& path) {
        const size_t slash = path.find_last_of("/\\");
        if (slash == std::string::npos || slash == 0) {
            return;
        }
        const std::string dir = path.substr(0, slash);
#ifdef _WIN32
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0755);
#endif
    }

    static std::vector<std::string> splitTsv(const std::string& line) {
        std::vector<std::string> fields;
        std::string field;
        std::istringstream ss(line);
        while (std::getline(ss, field, '\t')) {
            fields.push_back(field);
        }
        return fields;
    }

    static std::string escapeField(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '\\') {
                out += "\\\\";
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c == '\t') {
                out += "\\t";
            } else {
                out += c;
            }
        }
        return out;
    }

    static std::string unescapeField(const std::string& s) {
        std::string out;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                i++;
                if (s[i] == 'n') {
                    out += '\n';
                } else if (s[i] == 'r') {
                    out += '\r';
                } else if (s[i] == 't') {
                    out += '\t';
                } else {
                    out += s[i];
                }
            } else {
                out += s[i];
            }
        }
        return out;
    }

    static std::string formatEmbedding(const std::vector<float>& embedding) {
        std::ostringstream ss;
        for (size_t i = 0; i < embedding.size(); i++) {
            if (i) {
                ss << ',';
            }
            ss << embedding[i];
        }
        return ss.str();
    }

    static std::vector<float> parseEmbedding(const std::string& s) {
        std::vector<float> embedding;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                embedding.push_back(std::stof(token));
            } catch (...) {
            }
        }
        return embedding;
    }
};
