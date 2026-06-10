#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include "ai_provider.hpp"
#include "demo_data.hpp"
#include "document_store.hpp"
#include "eval_harness.hpp"
#include "httplib.h"
#include "json_utils.hpp"
#include "text_chunker.hpp"
#include "vector_index.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

inline void setCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

inline std::string vectorSearchToJson(const VectorDB::SearchResult& result) {
    std::ostringstream ss;
    ss << "{\"results\":[";
    for (size_t i = 0; i < result.hits.size(); i++) {
        if (i) {
            ss << ',';
        }
        const auto& hit = result.hits[i];
        ss << "{\"id\":" << hit.id
           << ",\"metadata\":" << jsonString(hit.metadata)
           << ",\"category\":" << jsonString(hit.category)
           << ",\"distance\":" << std::fixed << std::setprecision(6) << hit.distance
           << ",\"embedding\":" << jsonVector(hit.embedding) << '}';
    }
    ss << "],\"latencyUs\":" << result.latencyUs
       << ",\"algo\":" << jsonString(result.algorithm)
       << ",\"metric\":" << jsonString(result.metric) << '}';
    return ss.str();
}

inline std::string allVectorItemsToJson(const std::vector<VectorItem>& items) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < items.size(); i++) {
        if (i) {
            ss << ',';
        }
        ss << "{\"id\":" << items[i].id
           << ",\"metadata\":" << jsonString(items[i].metadata)
           << ",\"category\":" << jsonString(items[i].category)
           << ",\"embedding\":" << jsonVector(items[i].emb) << '}';
    }
    ss << ']';
    return ss.str();
}

inline std::string hnswInfoToJson(const HNSWIndex::GraphInfo& info) {
    std::ostringstream ss;
    ss << "{\"topLayer\":" << info.topLayer
       << ",\"nodeCount\":" << info.nodeCount
       << ",\"nodesPerLayer\":[";
    for (size_t i = 0; i < info.nodesPerLayer.size(); i++) {
        if (i) {
            ss << ',';
        }
        ss << info.nodesPerLayer[i];
    }
    ss << "],\"edgesPerLayer\":[";
    for (size_t i = 0; i < info.edgesPerLayer.size(); i++) {
        if (i) {
            ss << ',';
        }
        ss << info.edgesPerLayer[i];
    }
    ss << "],\"nodes\":[";
    for (size_t i = 0; i < info.nodes.size(); i++) {
        if (i) {
            ss << ',';
        }
        const auto& node = info.nodes[i];
        ss << "{\"id\":" << node.id
           << ",\"metadata\":" << jsonString(node.metadata)
           << ",\"category\":" << jsonString(node.category)
           << ",\"maxLyr\":" << node.maxLayer << '}';
    }
    ss << "],\"edges\":[";
    for (size_t i = 0; i < info.edges.size(); i++) {
        if (i) {
            ss << ',';
        }
        const auto& edge = info.edges[i];
        ss << "{\"src\":" << edge.src
           << ",\"dst\":" << edge.dst
           << ",\"lyr\":" << edge.layer << '}';
    }
    ss << "]}";
    return ss.str();
}

inline std::string docListToJson(const std::vector<DocItem>& docs) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < docs.size(); i++) {
        if (i) {
            ss << ',';
        }
        std::string preview = docs[i].text.substr(0, 120);
        if (docs[i].text.size() > 120) {
            preview += "...";
        }
        const int words = docs[i].text.empty()
            ? 0
            : (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1;
        ss << "{\"id\":" << docs[i].id
           << ",\"title\":" << jsonString(docs[i].title)
           << ",\"preview\":" << jsonString(preview)
           << ",\"words\":" << words << '}';
    }
    ss << ']';
    return ss.str();
}

inline std::string docContextsArrayToJson(const std::vector<std::pair<float, DocItem>>& hits, bool includeText) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < hits.size(); i++) {
        if (i) {
            ss << ',';
        }
        ss << "{\"id\":" << hits[i].second.id
           << ",\"title\":" << jsonString(hits[i].second.title);
        if (includeText) {
            ss << ",\"text\":" << jsonString(hits[i].second.text);
        }
        ss << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
    }
    ss << ']';
    return ss.str();
}

inline std::string docSearchToJson(const std::vector<std::pair<float, DocItem>>& hits, bool includeText) {
    return "{\"contexts\":" + docContextsArrayToJson(hits, includeText) + "}";
}

inline std::string buildRagPrompt(
    const std::string& question,
    const std::vector<std::pair<float, DocItem>>& hits)
{
    std::ostringstream context;
    for (int i = 0; i < (int)hits.size(); i++) {
        context << "[" << (i + 1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
    }

    return
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context when it is relevant. "
        "If the context is insufficient, say what is missing and answer from general knowledge.\n\n"
        "Context:\n" + context.str() +
        "Question: " + question + "\n\n"
        "Answer:";
}

inline std::string mcpToolsJson() {
    return
        "{\"tools\":["
        "{\"name\":\"search_docs\",\"description\":\"Semantic search over persisted document chunks\","
        "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"question\":{\"type\":\"string\"},\"k\":{\"type\":\"integer\"}},\"required\":[\"question\"]}},"
        "{\"name\":\"list_sources\",\"description\":\"List indexed document chunks\","
        "\"inputSchema\":{\"type\":\"object\",\"properties\":{}}},"
        "{\"name\":\"ask_doc\",\"description\":\"Run the local RAG pipeline over indexed documents\","
        "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"question\":{\"type\":\"string\"},\"k\":{\"type\":\"integer\"}},\"required\":[\"question\"]}}"
        "]}";
}

inline void registerRoutes(
    httplib::Server& svr,
    VectorDB& vectorDb,
    DocumentStore& docs,
    AiProvider& ai)
{
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        res.status = 204;
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        auto q = parseVectorCsv(req.get_param_value("v"));
        if ((int)q.size() != DEMO_DIMS) {
            res.set_content("{\"error\":\"need 16D vector\"}", "application/json");
            return;
        }
        int k = 5;
        try {
            k = std::stoi(req.get_param_value("k"));
        } catch (...) {
        }
        std::string metric = req.get_param_value("metric");
        if (metric.empty()) {
            metric = "cosine";
        }
        std::string algorithm = req.get_param_value("algo");
        if (algorithm.empty()) {
            algorithm = "hnsw";
        }
        res.set_content(vectorSearchToJson(vectorDb.search(q, k, metric, algorithm)), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        std::string metadata;
        std::string category;
        std::vector<float> embedding;
        if (!parseVectorItemBody(req.body, metadata, category, embedding) ||
            (int)embedding.size() != DEMO_DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json");
            return;
        }
        int id = vectorDb.insert(metadata, category, embedding, getDistanceFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const int id = std::stoi(req.matches[1]);
        const bool ok = vectorDb.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        res.set_content(allVectorItemsToJson(vectorDb.all()), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        auto q = parseVectorCsv(req.get_param_value("v"));
        if ((int)q.size() != DEMO_DIMS) {
            res.set_content("{\"error\":\"need 16D vector\"}", "application/json");
            return;
        }
        int k = 5;
        try {
            k = std::stoi(req.get_param_value("k"));
        } catch (...) {
        }
        std::string metric = req.get_param_value("metric");
        if (metric.empty()) {
            metric = "cosine";
        }
        const auto b = vectorDb.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bruteForceUs
           << ",\"kdtreeUs\":" << b.kdTreeUs
           << ",\"hnswUs\":" << b.hnswUs
           << ",\"itemCount\":" << b.itemCount << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        res.set_content(hnswInfoToJson(vectorDb.hnswInfo()), "application/json");
    });

    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const std::string title = extractJsonString(req.body, "title");
        const std::string text = extractJsonString(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json");
            return;
        }

        std::vector<int> ids;
        auto chunks = chunkText(text, 250, 30);
        for (int i = 0; i < (int)chunks.size(); i++) {
            auto embedding = ai.embed(chunks[i]);
            if (embedding.empty()) {
                res.set_content("{\"error\":\"AI provider unavailable\"}", "application/json");
                return;
            }
            const std::string chunkTitle = chunks.size() > 1
                ? title + " [" + std::to_string(i + 1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            ids.push_back(docs.insert(chunkTitle, chunks[i], embedding));
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i) {
                ss << ',';
            }
            ss << ids[i];
        }
        ss << "],\"chunks\":" << chunks.size() << ",\"dims\":" << docs.dims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const int id = std::stoi(req.matches[1]);
        const bool ok = docs.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        res.set_content(docListToJson(docs.all()), "application/json");
    });

    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const std::string question = extractJsonString(req.body, "question");
        const int k = extractJsonInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json");
            return;
        }
        auto qEmbedding = ai.embed(question);
        if (qEmbedding.empty()) {
            res.set_content("{\"error\":\"AI provider unavailable\"}", "application/json");
            return;
        }
        res.set_content(docSearchToJson(docs.search(qEmbedding, k), false), "application/json");
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const std::string question = extractJsonString(req.body, "question");
        const int k = extractJsonInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json");
            return;
        }
        auto qEmbedding = ai.embed(question);
        if (qEmbedding.empty()) {
            res.set_content("{\"error\":\"AI provider unavailable\"}", "application/json");
            return;
        }
        auto hits = docs.search(qEmbedding, k);
        auto answer = ai.generate(buildRagPrompt(question, hits));

        std::ostringstream ss;
        ss << "{\"answer\":" << jsonString(answer)
           << ",\"provider\":" << jsonString(ai.name())
           << ",\"model\":" << jsonString(ai.generationModel())
           << ",\"contexts\":" << docContextsArrayToJson(hits, true)
           << ",\"docCount\":" << docs.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        const bool up = ai.isAvailable();
        const bool ollamaUp = ai.name() == "ollama" && up;
        std::ostringstream ss;
        ss << "{\"aiAvailable\":" << (up ? "true" : "false")
           << ",\"ollamaAvailable\":" << (ollamaUp ? "true" : "false")
           << ",\"provider\":" << jsonString(ai.name())
           << ",\"embedModel\":" << jsonString(ai.embeddingModel())
           << ",\"genModel\":" << jsonString(ai.generationModel())
           << ",\"setupHint\":" << jsonString(ai.setupHint())
           << ",\"docCount\":" << docs.size()
           << ",\"docDims\":" << docs.dims()
           << ",\"demoDims\":" << DEMO_DIMS
           << ",\"demoCount\":" << vectorDb.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        std::ostringstream ss;
        ss << "{\"count\":" << vectorDb.size()
           << ",\"dims\":" << DEMO_DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/mcp/tools", [](const httplib::Request&, httplib::Response& res) {
        setCors(res);
        res.set_content(mcpToolsJson(), "application/json");
    });

    svr.Post("/mcp/call", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const std::string name = extractJsonString(req.body, "name");
        const std::string question = extractJsonString(req.body, "question");
        const int k = extractJsonInt(req.body, "k", 3);

        if (name == "list_sources") {
            res.set_content("{\"content\":" + docListToJson(docs.all()) + "}", "application/json");
            return;
        }
        if (name != "search_docs" && name != "ask_doc") {
            res.set_content("{\"error\":\"unknown tool\"}", "application/json");
            return;
        }
        if (question.empty()) {
            res.set_content("{\"error\":\"question required\"}", "application/json");
            return;
        }
        auto qEmbedding = ai.embed(question);
        if (qEmbedding.empty()) {
            res.set_content("{\"error\":\"AI provider unavailable\"}", "application/json");
            return;
        }
        auto hits = docs.search(qEmbedding, k);
        if (name == "search_docs") {
            res.set_content(docSearchToJson(hits, true), "application/json");
            return;
        }
        std::string answer = ai.generate(buildRagPrompt(question, hits));
        res.set_content("{\"answer\":" + jsonString(answer) +
            ",\"contexts\":" + docContextsArrayToJson(hits, true) + "}", "application/json");
    });

    svr.Post("/eval/retrieval", [&](const httplib::Request& req, httplib::Response& res) {
        setCors(res);
        const std::string question = extractJsonString(req.body, "question");
        const std::string expected = extractJsonString(req.body, "expectedTitle");
        const int k = extractJsonInt(req.body, "k", 3);
        if (question.empty() || expected.empty()) {
            res.set_content("{\"error\":\"need question and expectedTitle\"}", "application/json");
            return;
        }
        auto report = runRetrievalEval({{question, expected}}, docs, ai, k);
        std::ostringstream ss;
        ss << "{\"total\":" << report.total
           << ",\"hits\":" << report.hits
           << ",\"recallAtK\":" << std::fixed << std::setprecision(3) << report.recallAtK
           << ",\"latencyUs\":" << report.latencyUs << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("index.html");
        if (!file.is_open()) {
            res.status = 404;
            return;
        }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()),
            "text/html");
    });
}

inline int runServer() {
    VectorDB vectorDb(DEMO_DIMS);
    DocumentStore docs("data/documents.tsv");
    auto ai = createAiProviderFromEnv();
    loadDemoVectors(vectorDb);

    const bool aiUp = ai->isAvailable();
    std::cout << "=== VectorDB Engine ===\n";
    std::cout << "http://localhost:8080\n";
    std::cout << vectorDb.size() << " demo vectors | " << DEMO_DIMS
              << " dims | HNSW+KD-Tree+BruteForce\n";
    std::cout << docs.size() << " persisted document chunks | MCP+Eval endpoints enabled\n";
    std::cout << "AI provider: " << ai->name() << ' '
              << (aiUp ? "ONLINE" : "OFFLINE") << "\n";
    if (aiUp) {
        std::cout << "  embed model: " << ai->embeddingModel()
                  << "  gen model: " << ai->generationModel() << "\n";
    } else {
        std::cout << "  setup: " << ai->setupHint() << "\n";
    }

    httplib::Server svr;
    registerRoutes(svr, vectorDb, docs, *ai);
    svr.listen("0.0.0.0", 8080);
    return 0;
}
