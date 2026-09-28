#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Q { std::string text; bool yes; };
struct Measure {
    int parallel = 1;
    double prefix_ms = 0;
    double six_ms = 0;
    double six_restore_ms = 0;
    double six_decode_ms = 0;
    double n96_ms = 0;
    double n96_restore_ms = 0;
    double n96_decode_ms = 0;
    double dps96 = 0;
    int correct6 = 0;
};

static double since_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s, bool special) {
    int32_t n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), nullptr, 0, special, true);
    if (n >= 0) return {};
    std::vector<llama_token> out((size_t) -n);
    n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), out.data(), (int32_t) out.size(), special, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    out.resize((size_t) n);
    return out;
}

static llama_token one(const llama_vocab * v, const std::vector<std::string> & choices, std::string & selected) {
    for (const auto & s : choices) {
        auto x = tok(v, s, false);
        if (x.size() == 1) {
            selected = s;
            return x[0];
        }
    }
    throw std::runtime_error("no single-token label");
}

static llama_context * ctx_new(llama_model * model, int threads, int nseq) {
    auto p = llama_context_default_params();
    p.n_ctx = 4096;
    p.n_batch = 2048;
    p.n_ubatch = 512;
    p.n_seq_max = std::max(1, nseq);
    p.no_perf = false;
    auto * c = llama_init_from_model(model, p);
    if (!c) throw std::runtime_error("context init failed");
    llama_set_n_threads(c, threads, threads);
    return c;
}

static void batch_clear(llama_batch & b) {
    b.n_tokens = 0;
}

static int batch_add(
    llama_batch & b,
    llama_token token,
    llama_pos pos,
    const std::vector<llama_seq_id> & seqs,
    bool logits) {
    const int i = b.n_tokens++;
    b.token[i] = token;
    b.pos[i] = pos;
    b.n_seq_id[i] = (int32_t) seqs.size();
    for (size_t j = 0; j < seqs.size(); ++j) {
        b.seq_id[i][j] = seqs[j];
    }
    b.logits[i] = logits ? 1 : 0;
    return i;
}

static void decode_or_throw(llama_context * c, llama_batch b, const char * where) {
    const int rc = llama_decode(c, b);
    if (rc != 0) {
        throw std::runtime_error(std::string(where) + " llama_decode rc=" + std::to_string(rc));
    }
}

static bool direct_at(llama_context * c, int batch_index, llama_token yes_tok, llama_token no_tok) {
    float * logits = llama_get_logits_ith(c, batch_index);
    if (!logits) throw std::runtime_error("missing logits for batch index " + std::to_string(batch_index));
    return logits[yes_tok] > logits[no_tok];
}

struct Snapshot {
    std::vector<uint8_t> bytes;
};

static Snapshot snapshot(llama_context * c) {
    size_t sz = llama_state_get_size(c);
    Snapshot s;
    s.bytes.resize(sz);
    size_t got = llama_state_get_data(c, s.bytes.data(), s.bytes.size());
    if (got == 0 || got > s.bytes.size()) throw std::runtime_error("state snapshot failed");
    s.bytes.resize(got);
    return s;
}

static double restore(llama_context * c, const Snapshot & s) {
    auto t = Clock::now();
    size_t used = llama_state_set_data(c, s.bytes.data(), s.bytes.size());
    if (used == 0) throw std::runtime_error("state restore failed");
    return since_ms(t);
}

static double eval_prefix(
    llama_context * c,
    const std::vector<llama_token> & prefix,
    int nseq) {
    llama_batch b = llama_batch_init((int32_t) prefix.size(), 0, nseq);
    std::vector<llama_seq_id> seqs((size_t) nseq);
    for (int i = 0; i < nseq; ++i) seqs[(size_t) i] = i;
    for (size_t i = 0; i < prefix.size(); ++i) {
        batch_add(b, prefix[i], (llama_pos) i, seqs, false);
    }
    auto t = Clock::now();
    decode_or_throw(c, b, "prefix");
    double ms = since_ms(t);
    llama_batch_free(b);
    return ms;
}

struct GroupResult {
    double restore_ms = 0;
    double decode_ms = 0;
    int correct = 0;
};

static GroupResult eval_group(
    llama_context * c,
    const Snapshot & base,
    const std::vector<std::vector<llama_token>> & suffix,
    const std::vector<Q> & qs,
    int parallel,
    int qbase,
    llama_token yes_tok,
    llama_token no_tok,
    bool score) {

    GroupResult gr;
    gr.restore_ms = restore(c, base);

    size_t total_tokens = 0;
    for (int s = 0; s < parallel; ++s) {
        total_tokens += suffix[(size_t) ((qbase + s) % (int) qs.size())].size();
    }

    llama_batch b = llama_batch_init((int32_t) total_tokens, 0, parallel);
    std::vector<int> last((size_t) parallel, -1);

    for (int s = 0; s < parallel; ++s) {
        const int qi = (qbase + s) % (int) qs.size();
        const auto & x = suffix[(size_t) qi];
        for (size_t j = 0; j < x.size(); ++j) {
            const bool want_logits = (j + 1 == x.size());
            int bi = batch_add(
                b,
                x[j],
                (llama_pos) (j),
                { (llama_seq_id) s },
                want_logits);
            if (want_logits) last[(size_t) s] = bi;
        }
    }

    // Prefix positions occupy [0, prefix_len). The caller adds that offset after construction.
    // Positions are rewritten here from per-suffix offsets to absolute positions.
    // All sequences share the same prefix length, so each suffix starts at the same position.
    // infer prefix length from memory position max for seq 0
    llama_pos prefix_last = llama_memory_seq_pos_max(llama_get_memory(c), 0);
    if (prefix_last < 0) throw std::runtime_error("prefix memory missing after restore");
    const llama_pos prefix_len = prefix_last + 1;
    for (int i = 0; i < b.n_tokens; ++i) {
        b.pos[i] += prefix_len;
    }

    auto td = Clock::now();
    decode_or_throw(c, b, "suffix-batch");
    gr.decode_ms = since_ms(td);

    if (score) {
        for (int s = 0; s < parallel; ++s) {
            const int qi = (qbase + s) % (int) qs.size();
            bool pred = direct_at(c, last[(size_t) s], yes_tok, no_tok);
            if (pred == qs[(size_t) qi].yes) gr.correct++;
        }
    }

    llama_batch_free(b);
    return gr;
}

static Measure run_parallel(
    llama_model * model,
    int threads,
    const std::vector<llama_token> & prefix,
    const std::vector<std::vector<llama_token>> & suffix,
    const std::vector<Q> & qs,
    int parallel,
    llama_token yes_tok,
    llama_token no_tok) {

    Measure m;
    m.parallel = parallel;
    auto * c = ctx_new(model, threads, parallel);
    llama_memory_clear(llama_get_memory(c), false);

    m.prefix_ms = eval_prefix(c, prefix, parallel);
    Snapshot base = snapshot(c);

    auto t6 = Clock::now();
    for (int qbase = 0; qbase < 6; qbase += parallel) {
        auto gr = eval_group(c, base, suffix, qs, parallel, qbase, yes_tok, no_tok, true);
        m.six_restore_ms += gr.restore_ms;
        m.six_decode_ms += gr.decode_ms;
        m.correct6 += gr.correct;
    }
    m.six_ms = since_ms(t6);

    auto t96 = Clock::now();
    for (int i = 0; i < 96; i += parallel) {
        auto gr = eval_group(c, base, suffix, qs, parallel, i % 6, yes_tok, no_tok, false);
        m.n96_restore_ms += gr.restore_ms;
        m.n96_decode_ms += gr.decode_ms;
    }
    m.n96_ms = since_ms(t96);
    m.dps96 = 96000.0 / m.n96_ms;

    llama_free(c);
    return m;
}

static void emit(const Measure & m) {
    std::cout << std::fixed << std::setprecision(3)
              << "{"
              << "\"type\":\"parallel_result\","
              << "\"parallel\":" << m.parallel << ","
              << "\"prefix_ms\":" << m.prefix_ms << ","
              << "\"six_ms\":" << m.six_ms << ","
              << "\"six_restore_ms\":" << m.six_restore_ms << ","
              << "\"six_decode_ms\":" << m.six_decode_ms << ","
              << "\"n96_ms\":" << m.n96_ms << ","
              << "\"n96_restore_ms\":" << m.n96_restore_ms << ","
              << "\"n96_decode_ms\":" << m.n96_decode_ms << ","
              << "\"decisions_per_sec_96\":" << m.dps96 << ","
              << "\"correct_6\":" << m.correct6
              << "}\n" << std::flush;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        }
    }
    if (model_path.empty()) {
        std::cerr << "usage: llama-parallel-decision-bench -m model.gguf [--threads 4]\n";
        return 2;
    }

    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    auto tl = Clock::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    double load_ms = since_ms(tl);
    if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string yl, nl;
    llama_token yt = one(vocab, {" A", "A", " 1", "1"}, yl);
    llama_token nt = one(vocab, {" B", "B", " 0", "0"}, nl);

    const std::string prefix =
        "You are a deterministic binary decision engine. Use only the facts and policy below. "
        "A means YES and B means NO. Customer profile: age 45; home city Yokohama; membership Silver; "
        "payment current; marketing opt-in enabled; no cancellation request; smartphone purchased three months ago; "
        "one unresolved support ticket open for fourteen days. Policy: marketing campaign eligibility requires "
        "marketing opt-in and current payment status. Retention escalation is required for a cancellation request "
        "or an unresolved support ticket older than thirty days. Human review is required for an unresolved support "
        "ticket older than seven days. A recent-device-buyer purchased a device within six months. The senior offer "
        "requires age sixty-five or older. The Yokohama local event requires home city Yokohama. "
        "For binary output use exactly A for YES and B for NO, with no explanation.\n\n";

    const std::vector<Q> qs = {
        {"Is this customer eligible for the marketing campaign?", true},
        {"Does this customer require retention escalation?", false},
        {"Does this customer require human review?", true},
        {"Is this customer a recent-device-buyer?", true},
        {"Is this customer eligible for the senior offer?", false},
        {"Is this customer eligible for the Yokohama local event?", true},
    };

    auto pfx = tok(vocab, prefix, true);
    std::vector<std::vector<llama_token>> suffix;
    for (const auto & q : qs) {
        suffix.push_back(tok(vocab, "Question: " + q.text + "\nAnswer:", false));
    }

    std::cout << std::fixed << std::setprecision(3)
              << "{\"type\":\"meta\",\"model_load_ms\":" << load_ms
              << ",\"threads\":" << threads
              << ",\"prefix_tokens\":" << pfx.size()
              << ",\"yes_token\":" << yt
              << ",\"no_token\":" << nt
              << "}\n" << std::flush;

    // Warm-up one single-sequence direct-logit decision.
    {
        auto * c = ctx_new(model, threads, 1);
        llama_memory_clear(llama_get_memory(c), false);
        (void) eval_prefix(c, pfx, 1);
        Snapshot s = snapshot(c);
        (void) eval_group(c, s, suffix, qs, 1, 0, yt, nt, false);
        llama_free(c);
    }

    for (int p : {1, 2, 3, 6}) {
        try {
            emit(run_parallel(model, threads, pfx, suffix, qs, p, yt, nt));
        } catch (const std::exception & e) {
            std::cout << "{\"type\":\"parallel_error\",\"parallel\":" << p
                      << ",\"error\":\"" << e.what() << "\"}\n" << std::flush;
        }
    }

    llama_model_free(model);
    return 0;
}
