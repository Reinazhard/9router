import { FORMATS } from "../translator/formats.js";
import { buildErrorBody } from "./error.js";
import { SSE_DONE } from "./sseConstants.js";

const sharedEncoder = new TextEncoder();

// Parse SSE data line
export function parseSSELine(line, format = null) {
  if (!line) return null;

  // NDJSON format (Ollama): raw JSON lines without "data:" prefix
  if (format === FORMATS.OLLAMA) {
    const trimmed = line.trim();
    if (trimmed.startsWith("{")) {
      try {
        return JSON.parse(trimmed);
      } catch (error) {
        return null;
      }
    }
    return null;
  }

  // Standard SSE format: "data: {...}"
  if (line.charCodeAt(0) !== 100) return null; // 'd' = 100

  const data = line.slice(5).trim();
  if (data === "[DONE]") return { done: true };

  try {
    return JSON.parse(data);
  } catch (error) {
    if (data.length > 0 && data.length < 1000) {
      console.log(`[WARN] Failed to parse SSE line (${data.length} chars): ${data.substring(0, 100)}...`);
    }
    return null;
  }
}

// Check if chunk has valuable content (not empty)
export function hasValuableContent(chunk, format) {
  // OpenAI format
  if (format === FORMATS.OPENAI && chunk.choices?.[0]?.delta) {
    const delta = chunk.choices[0].delta;
    return delta.content && delta.content !== "" ||
           delta.reasoning_content && delta.reasoning_content !== "" ||
           delta.tool_calls && delta.tool_calls.length > 0 ||
           chunk.choices[0].finish_reason ||
           delta.role;
  }

  // Claude format
  if (format === FORMATS.CLAUDE) {
    const isContentBlockDelta = chunk.type === "content_block_delta";
    const hasText = chunk.delta?.text && chunk.delta.text !== "";
    const hasThinking = chunk.delta?.thinking && chunk.delta.thinking !== "";
    const hasInputJson = chunk.delta?.partial_json && chunk.delta.partial_json !== "";
    
    if (isContentBlockDelta && !hasText && !hasThinking && !hasInputJson) {
      return false;
    }
    return true;
  }

  return true; // Other formats: keep all chunks
}

// Fix invalid id (generic or too short)
export function fixInvalidId(parsed) {
  if (parsed.id && (parsed.id === "chat" || parsed.id === "completion" || parsed.id.length < 8)) {
    const fallbackId = parsed.extend_fields?.requestId || 
                      parsed.extend_fields?.traceId || 
                      Date.now().toString(36);
    parsed.id = `chatcmpl-${fallbackId}`;
    return true;
  }
  return false;
}

function cleanUsagePayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return payload;
  }

  let cleaned = payload;

  if ("usage" in cleaned) {
    if (cleaned.usage === null) {
      const { usage, ...payloadWithoutUsage } = cleaned;
      cleaned = payloadWithoutUsage;
    } else if (typeof cleaned.usage === "object" && cleaned.usage.perf_metrics === null) {
      const { perf_metrics, ...usageWithoutPerf } = cleaned.usage;
      cleaned = { ...cleaned, usage: usageWithoutPerf };
    }
  }

  if (cleaned.response && typeof cleaned.response === "object" && !Array.isArray(cleaned.response)) {
    const cleanedResponse = cleanUsagePayload(cleaned.response);
    if (cleanedResponse !== cleaned.response) {
      cleaned = { ...cleaned, response: cleanedResponse };
    }
  }

  return cleaned;
}

// Format output as SSE
export function formatSSE(data, sourceFormat) {
  if (data === null || data === undefined) return "data: null\n\n";
  if (data && data.done) return "data: [DONE]\n\n";

  // OpenAI Responses API format
  if (data && data.event && data.data) {
    const cleanedEventData = cleanUsagePayload(data.data);
    return `event: ${data.event}\ndata: ${JSON.stringify(cleanedEventData)}\n\n`;
  }

  data = cleanUsagePayload(data);

  // Claude format
  if (sourceFormat === FORMATS.CLAUDE && data && data.type) {
    return `event: ${data.type}\ndata: ${JSON.stringify(data)}\n\n`;
  }

  return `data: ${JSON.stringify(data)}\n\n`;
}

// Terminal frames for a stream that aborted after HTTP 200 was already sent, so
// the status code can no longer change. OpenAI-compatible clients (openai-python
// raises APIError on any `data:` payload carrying an `error` key, checked before
// [DONE]) need the error frame first, then [DONE]; Anthropic clients need
// `event: error`. Never fabricate a successful finish_reason instead.
//
// Returns encoded bytes: onAbortTerminal callbacks are enqueued verbatim, same
// as buildAbortedResponsesTerminalBytes.
//
// NOTE: non-SSE client formats (Ollama NDJSON) get an SSE frame here — dead in
// practice because detectFormatByEndpoint never resolves to OLLAMA.
export function buildStreamErrorBytes(statusCode, message, clientFormat) {
  const { error } = buildErrorBody(statusCode, message);

  const sse = clientFormat === FORMATS.CLAUDE
    ? formatSSE({ type: "error", error }, FORMATS.CLAUDE)
    : formatSSE({ error }, clientFormat) + SSE_DONE;

  return sharedEncoder.encode(sse);
}

/**
 * Build one SSE keepalive frame for the given client format, emitted while the
 * upstream is silent (long reasoning/prefill) so the client can tell a live
 * stream from a dead one and does not trip its own idle timeout.
 *
 * - Anthropic clients: `event: ping` / `data: {"type":"ping"}` — the protocol's
 *   own keepalive, which Claude Code emits and accepts.
 * - Every other SSE client: a comment line (`: keepalive`), which SSE parsers
 *   ignore by spec.
 *
 * Returns encoded bytes, or null when the client format must not receive extra
 * frames.
 */
export function buildKeepaliveBytes(clientFormat) {
  if (clientFormat === FORMATS.CLAUDE) {
    return sharedEncoder.encode(formatSSE({ type: "ping" }, FORMATS.CLAUDE));
  }
  return sharedEncoder.encode(": keepalive\n\n");
}

