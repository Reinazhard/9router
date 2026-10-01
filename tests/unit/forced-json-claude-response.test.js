// Regression: a Claude-format client (Claude Code) that sends a NON-streaming
// request to a forced-streaming provider (CodeBuddy/Antigravity) must receive an
// Anthropic `Message`, not a `chat.completion`.
//
// The provider always forces upstream streaming, so a non-streaming request is
// aggregated by handleForcedSSEToJson. Its standard Chat Completions path used
// to return the raw OpenAI body for every non-Responses client — including
// Claude — so Claude Code reported:
//   "API returned an empty or malformed response (HTTP 200) ... body is JSON but
//    not a Message ... non-streaming retry of streaming request"
import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/usageDb.js", () => ({
  appendRequestLog: vi.fn(async () => {}),
  saveRequestDetail: vi.fn(async () => {}),
  saveRequestUsage: vi.fn(async () => {}),
}));

const { FORMATS } = await import("../../open-sse/translator/formats.js");
const { handleForcedSSEToJson } = await import("../../open-sse/handlers/chatCore/sseToJsonHandler.js");

// Build an upstream SSE stream from OpenAI chat.completion.chunk lines.
function sseCtx({ sourceFormat, chunks }) {
  const encoder = new TextEncoder();
  const raw = chunks.map((c) => `data: ${JSON.stringify(c)}`).join("\n\n") + "\n\ndata: [DONE]\n\n";
  return {
    providerResponse: new Response(
      new ReadableStream({ start(controller) { controller.enqueue(encoder.encode(raw)); controller.close(); } }),
      { headers: { "content-type": "text/event-stream" } }
    ),
    sourceFormat,
    targetFormat: FORMATS.OPENAI,
    provider: "codebuddy-intl",
    model: "glm-5.3",
    body: { model: "glm-5.3", messages: [] },
    stream: false,
    requestStartTime: Date.now(),
    connectionId: "test-connection",
    clientRawRequest: { endpoint: "/v1/messages" },
    trackDone: vi.fn(),
    appendLog: vi.fn(),
  };
}

const TEXT_CHUNKS = [
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: { role: "assistant", content: "hello " }, finish_reason: null }] },
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: { content: "world" }, finish_reason: null }] },
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 12, completion_tokens: 3 } },
];

const TOOL_CHUNKS = [
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: { tool_calls: [{ index: 0, id: "call_9", type: "function", function: { name: "Bash", arguments: "" } }] }, finish_reason: null }] },
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"command":"ls"}' } }] }, finish_reason: null }] },
  { id: "chatcmpl-sse", object: "chat.completion.chunk", created: 1700000000, model: "glm-5.3", choices: [{ delta: {}, finish_reason: "tool_calls" }] },
];

describe("forced-SSE JSON path for a Claude client behind a forced-streaming provider", () => {
  it("returns an Anthropic Message (not chat.completion) for a text turn", async () => {
    const result = await handleForcedSSEToJson(sseCtx({ sourceFormat: FORMATS.CLAUDE, chunks: TEXT_CHUNKS }));
    expect(result.success).toBe(true);
    const json = await result.response.json();

    expect(json.object).toBeUndefined(); // not a chat.completion
    expect(json.type).toBe("message");
    expect(json.role).toBe("assistant");
    expect(json.content).toEqual([{ type: "text", text: "hello world" }]);
    expect(json.stop_reason).toBe("end_turn");
    expect(json.usage).toMatchObject({ input_tokens: 12, output_tokens: 3 });
  });

  it("emits a tool_use block for a tool call", async () => {
    const result = await handleForcedSSEToJson(sseCtx({ sourceFormat: FORMATS.CLAUDE, chunks: TOOL_CHUNKS }));
    const json = await result.response.json();
    expect(json.type).toBe("message");
    const toolUse = json.content.find((b) => b.type === "tool_use");
    expect(toolUse).toBeTruthy();
    expect(toolUse.id).toBe("call_9");
    expect(toolUse.name).toBe("Bash");
    expect(toolUse.input).toEqual({ command: "ls" });
    expect(json.stop_reason).toBe("tool_use");
  });

  it("still returns chat.completion for a plain OpenAI client (no regression)", async () => {
    const result = await handleForcedSSEToJson(sseCtx({ sourceFormat: FORMATS.OPENAI, chunks: TEXT_CHUNKS }));
    const json = await result.response.json();
    expect(json.object).toBe("chat.completion");
    expect(json.choices[0].message.content).toBe("hello world");
  });
});
