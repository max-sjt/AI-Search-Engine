#pragma once

#include "ai_provider.hpp"
#include "document_store.hpp"

#include <chrono>
#include <string>
#include <vector>

struct EvalCase {
    std::string question;
    std::string expectedTitle;
};

struct EvalReport {
    int total = 0;
    int hits = 0;
    double recallAtK = 0.0;
    long long latencyUs = 0;
};

inline EvalReport runRetrievalEval(
    const std::vector<EvalCase>& cases,
    const DocumentStore& docs,
    const AiProvider& ai,
    int k)
{
    EvalReport report;
    report.total = (int)cases.size();

    const auto started = std::chrono::high_resolution_clock::now();
    for (const auto& testCase : cases) {
        auto embedding = ai.embed(testCase.question);
        if (embedding.empty()) {
            continue;
        }
        auto hits = docs.search(embedding, k, 2.0f);
        for (const auto& hit : hits) {
            if (hit.second.title.find(testCase.expectedTitle) != std::string::npos) {
                report.hits++;
                break;
            }
        }
    }
    report.latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - started).count();
    if (report.total > 0) {
        report.recallAtK = (double)report.hits / (double)report.total;
    }
    return report;
}
