#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include "../src/ai/ai_provider.hpp"
#include "../src/rag/document_store.hpp"
#include "../src/rag/text_chunker.hpp"
#include "../src/util/json_utils.hpp"
#include "../src/vector/vector_index.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        failures++;
    }
}

void testVectorSearch() {
    VectorDB db(2);
    auto dist = getDistanceFn("cosine");
    db.insert("alpha document", "doc", {1.0f, 0.0f}, dist);
    db.insert("beta document", "doc", {0.0f, 1.0f}, dist);

    auto result = db.search({0.95f, 0.05f}, 1, "cosine", "hnsw");
    check(result.hits.size() == 1, "vector search returns one nearest hit");
    check(result.hits[0].metadata == "alpha document", "vector search ranks closest vector first");
}

void testDocumentPersistence() {
    const std::string path = "semantic_search_smoke_docs.tsv";
    std::remove(path.c_str());

    {
        DocumentStore docs(path);
        docs.insert("Alpha Notes", "alpha retrieval text", {1.0f, 0.0f});
        docs.insert("Beta Notes", "beta retrieval text", {0.0f, 1.0f});
        auto hits = docs.search({0.9f, 0.1f}, 1, 2.0f);
        check(hits.size() == 1, "document search returns one nearest chunk");
        check(hits[0].second.title == "Alpha Notes", "document search ranks closest chunk first");
    }

    {
        DocumentStore reloaded(path);
        check(reloaded.size() == 2, "document store reloads persisted chunks");
        auto hits = reloaded.search({0.1f, 0.9f}, 1, 2.0f);
        check(hits.size() == 1, "reloaded document store can search");
        check(hits[0].second.title == "Beta Notes", "reloaded search uses persisted embeddings");
    }

    std::remove(path.c_str());
}

void testUtilities() {
    std::string body = "{\"title\":\"A\\nB\",\"k\":4,\"embedding\":[1.0,2.5]}";
    check(extractJsonString(body, "title") == "A\nB", "json string extractor handles escapes");
    check(extractJsonInt(body, "k", 0) == 4, "json integer extractor reads numeric fields");
    auto embedding = extractJsonFloatArray(body, "embedding");
    check(embedding.size() == 2 && embedding[1] == 2.5f, "json float array extractor parses embeddings");

    auto chunks = chunkText("one two three four five", 3, 1);
    check(chunks.size() == 2, "chunker splits text with overlap");
    check(chunks[1] == "three four five", "chunker preserves overlap window");
}

void testAiProviders() {
    OllamaProvider ollama;
    check(ollama.name() == "ollama", "ollama provider exposes provider name");
    check(!ollama.embeddingModel().empty(), "ollama provider exposes embedding model");
    check(!ollama.generationModel().empty(), "ollama provider exposes generation model");

    QwenProvider qwen;
    check(qwen.name() == "qwen", "qwen provider exposes provider name");
    check(!qwen.embeddingModel().empty(), "qwen provider exposes embedding model");
    check(!qwen.generationModel().empty(), "qwen provider exposes generation model");
}

int main() {
    testVectorSearch();
    testDocumentPersistence();
    testUtilities();
    testAiProviders();

    if (failures != 0) {
        std::cerr << failures << " smoke test failure(s)\n";
        return 1;
    }
    std::cout << "smoke tests passed\n";
    return 0;
}
