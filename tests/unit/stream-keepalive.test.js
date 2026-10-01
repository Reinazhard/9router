// Regression: while the upstream is silent (a long reasoning/prefill phase on a
// forced-stream provider), the gateway must emit a periodic SSE keepalive so the
// client can tell a live stream from a dead one and does not trip its own idle
// timeout and retry.
//
// createDisconnectAwareStream races the upstream read against a keepalive timer:
// when the timer wins it enqueues one heartbeat frame and keeps the SAME pending
// read (never issuing two concurrent reads, which would split a chunk).
import { describe, expect, it } from "vitest";
import { createDisconnectAwareStream } from "../../open-sse/utils/streamHandler.js";
import { buildKeepaliveBytes } from "../../open-sse/utils/streamHelpers.js";

const FORMATS = { CLAUDE: "claude", OPENAI: "openai" };

// A fake transform: { readable } whose chunks the test controls, plus a no-op
// writable matching the shape createDisconnectAwareStream expects.
function controllableSource() {
  let push;
  const readable = new ReadableStream({
    start(controller) { push = (v) => controller.enqueue(v); },
  });
  return {
    readable,
    writable: { getWriter: () => ({ abort: () => Promise.resolve() }) },
    push,
  };
}

const connectedController = () => ({
  isConnected: () => true,
  handleComplete: () => {},
  handleError: () => {},
  handleDisconnect: () => {},
  abort: () => {},
});

const decoder = new TextDecoder();
const readAll = async (rs, windowMs) => {
  const reader = rs.getReader();
  let out = "";
  const deadline = Date.now() + windowMs;
  while (Date.now() < deadline) {
    const race = await Promise.race([
      reader.read(),
      new Promise((r) => setTimeout(() => r(null), 30)),
    ]);
    if (race === null) continue;
    if (race.done) break;
    out += decoder.decode(race.value, { stream: true });
    if (out.includes("ping")) break;
  }
  try { await reader.cancel(); } catch { /* already closed */ }
  return out;
};

describe("stream keepalive", () => {
  it("emits a Claude `event: ping` during upstream silence", async () => {
    const src = controllableSource();
    const rs = createDisconnectAwareStream(src, connectedController(), null, {
      bytes: () => buildKeepaliveBytes(FORMATS.CLAUDE),
      intervalMs: 20,
    });
    const out = await readAll(rs, 400);
    expect(out).toContain("event: ping");
    expect(out).toContain('"type":"ping"');
  });

  it("emits a comment keepalive for OpenAI clients", async () => {
    const src = controllableSource();
    const rs = createDisconnectAwareStream(src, connectedController(), null, {
      bytes: () => buildKeepaliveBytes(FORMATS.OPENAI),
      intervalMs: 20,
    });
    const out = await readAll(rs, 400);
    expect(out).toContain(": keepalive");
  });

  it("real upstream bytes are still delivered while a keepalive races", async () => {
    const src = controllableSource();
    const rs = createDisconnectAwareStream(src, connectedController(), null, {
      bytes: () => buildKeepaliveBytes(FORMATS.CLAUDE),
      intervalMs: 30,
    });
    // Push a real chunk; it must arrive intact (not split/lost by the race).
    setTimeout(() => src.push(new TextEncoder().encode('data: {"type":"content_block_delta"}\n\n')), 10);
    const out = await readAll(rs, 400);
    expect(out).toContain("content_block_delta");
  });

  it("no keepalive when the option is null", async () => {
    const src = controllableSource();
    const rs = createDisconnectAwareStream(src, connectedController(), null, null);
    const out = await readAll(rs, 120);
    expect(out).not.toContain("ping");
    expect(out).not.toContain("keepalive");
  });
});
