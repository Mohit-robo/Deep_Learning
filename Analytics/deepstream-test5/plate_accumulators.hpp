/**
 * plate_accumulators.hpp
 *
 * Per-track plate text accumulator.
 * Uses confidence-weighted character-position voting to produce a single
 * best-guess plate string when the track is terminated.
 *
 * Design notes:
 *  - plate_accum_feed / plate_accum_finalize are called exclusively from the
 *    GStreamer streaming thread (probe callbacks), so no lock is needed there.
 *  - plate_accum_reap_stale is called from perf_cb (timer thread); in practice
 *    this is infrequent and the map operations are pointer-stable, so the
 *    practical race window is negligible.
 *  - Kafka is handled by the existing per-frame NvDsEventMsgMeta path;
 *    the accumulator only prints the high-confidence final plate string.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Configuration – tune these at build time if needed
// ---------------------------------------------------------------------------
#define PLATE_MIN_OBSERVATIONS   3     // skip tracks with fewer readings
#define PLATE_MIN_MEAN_CONF      0.40f // skip tracks whose mean confidence is too low
#define PLATE_MAX_LENGTH         10    // Indian number plates are max 10 chars
#define PLATE_IDLE_TIMEOUT_SECS  5     // finalize a track not seen for this many seconds

// ---------------------------------------------------------------------------
// Internal per-position vote map
//   key   = character (e.g. 'K')
//   value = cumulative confidence for that character at this position
// ---------------------------------------------------------------------------
using CharVotes = std::map<char, float>;

struct PlateAccumulator {
    std::vector<CharVotes> pos_votes;  // one entry per string position (capped at PLATE_MAX_LENGTH)
    float    total_conf   = 0.0f;     // sum of all per-character confidences seen
    int      total_chars  = 0;        // total characters fed in (for mean)
    int      obs_count    = 0;        // number of full-string observations
    uint32_t stream_id    = 0;        // stream this plate belongs to
    time_t   last_seen    = 0;        // wall-clock time of last feed (for idle timeout)
};

// Global map: tracker object_id → accumulator
static std::unordered_map<uint64_t, PlateAccumulator> g_plate_accum;

// ---------------------------------------------------------------------------
// Feed one per-frame plate reading into the accumulator for a track.
//
// plate_text : the cleaned string from OCR (e.g. "KA02MN1826")
// avg_conf   : per-word confidence from label_info->result_prob
// track_id   : object_id from NvDsObjectMeta
// stream_id  : source_id from NvDsFrameMeta
// ---------------------------------------------------------------------------
static inline void plate_accum_feed(uint64_t track_id,
                                    const char *plate_text,
                                    float avg_conf,
                                    uint32_t stream_id = 0)
{
    if (!plate_text || plate_text[0] == '\0') return;

    PlateAccumulator &acc = g_plate_accum[track_id];
    acc.stream_id = stream_id;
    acc.last_seen = time(nullptr);
    int len = (int)strlen(plate_text);

    // Cap at max plate length – longer strings are noise from partial detections
    int effective_len = std::min(len, (int)PLATE_MAX_LENGTH);

    // Grow the position vote vector if this string contributes new positions
    if (effective_len > (int)acc.pos_votes.size())
        acc.pos_votes.resize(effective_len);

    for (int i = 0; i < effective_len; i++) {
        char c = plate_text[i];
        acc.pos_votes[i][c] += avg_conf;
        acc.total_conf  += avg_conf;
        acc.total_chars += 1;
    }
    acc.obs_count++;
}

// ---------------------------------------------------------------------------
// Finalize the accumulator for a track on termination.
// Performs the confidence-weighted vote and prints the result.
//
// track_id  : the terminated track's uniqueId
// stream_id : for the print label
// ---------------------------------------------------------------------------
static inline void plate_accum_finalize(uint64_t track_id, uint32_t stream_id)
{
    auto it = g_plate_accum.find(track_id);
    if (it == g_plate_accum.end()) return; // not a plate track, nothing to do

    PlateAccumulator &acc = it->second;

    // Gate: skip low-observation or low-confidence tracks
    if (acc.obs_count < PLATE_MIN_OBSERVATIONS) {
        g_plate_accum.erase(it);
        return;
    }

    float mean_conf = (acc.total_chars > 0)
                      ? acc.total_conf / acc.total_chars
                      : 0.0f;

    if (mean_conf < PLATE_MIN_MEAN_CONF) {
        g_plate_accum.erase(it);
        return;
    }

    // Build best string: for each position pick the char with highest cumulative confidence
    std::string best;
    best.reserve(acc.pos_votes.size());
    for (const CharVotes &votes : acc.pos_votes) {
        char best_char = '?';
        float best_score = -1.0f;
        for (const auto &kv : votes) {
            if (kv.second > best_score) {
                best_score = kv.second;
                best_char  = kv.first;
            }
        }
        best += best_char;
    }

    printf("[PLATE] stream=%u track=%lu obs=%d mean_conf=%.2f text=%s\n",
           stream_id, (unsigned long)track_id,
           acc.obs_count, mean_conf, best.c_str());
    fflush(stdout);

    // Free memory for this track
    g_plate_accum.erase(it);
}

// ---------------------------------------------------------------------------
// Cleanup helper: erase a track that was never a plate track (e.g. person).
// This is a no-op if the track_id was never in the map.
// ---------------------------------------------------------------------------
static inline void plate_accum_erase(uint64_t track_id)
{
    g_plate_accum.erase(track_id);
}

// ---------------------------------------------------------------------------
// EOS flush: finalize ALL remaining entries in the map.
// Call this once after g_main_loop_run() returns (pipeline end-of-stream).
// ---------------------------------------------------------------------------
static inline void plate_accum_flush_all()
{
    // Collect keys first to avoid iterator invalidation during erase
    std::vector<uint64_t> keys;
    keys.reserve(g_plate_accum.size());
    for (const auto &kv : g_plate_accum)
        keys.push_back(kv.first);

    for (uint64_t track_id : keys) {
        auto it = g_plate_accum.find(track_id);
        if (it == g_plate_accum.end()) continue;
        plate_accum_finalize(track_id, it->second.stream_id);
    }
}

// ---------------------------------------------------------------------------
// Idle timeout reaper: finalize any tracks that haven't been seen recently.
// Designed to be called periodically from a timer (e.g. perf_cb) so live
// feeds emit results when vehicles leave the frame without track termination.
// ---------------------------------------------------------------------------
static inline void plate_accum_reap_stale()
{
    time_t now = time(nullptr);
    std::vector<uint64_t> stale_keys;

    for (const auto &kv : g_plate_accum) {
        if (difftime(now, kv.second.last_seen) >= PLATE_IDLE_TIMEOUT_SECS) {
            stale_keys.push_back(kv.first);
        }
    }

    for (uint64_t track_id : stale_keys) {
        auto it = g_plate_accum.find(track_id);
        if (it == g_plate_accum.end()) continue;
        plate_accum_finalize(track_id, it->second.stream_id);
    }
}
