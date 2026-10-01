// Regression: reported usage must be the provider's REAL numbers, with no
// headroom buffer.
//
// addBufferToUsage() used to add a fixed 2000 tokens to input/prompt/total
// before returning usage to the client. That inflated a client's context meter
// (Claude Code's "% context used") and made the reported value disagree with the
// value recorded for accounting. It is now a pass-through.
import { describe, expect, it } from "vitest";
import { addBufferToUsage, formatUsage, estimateUsage } from "../../open-sse/utils/usageTracking.js";
import { FORMATS } from "../../open-sse/translator/formats.js";

describe("usage is reported verbatim (no headroom buffer)", () => {
  it("returns real usage unchanged (Claude shape)", () => {
    expect(addBufferToUsage({ input_tokens: 554000, output_tokens: 42 })).toEqual({
      input_tokens: 554000,
      output_tokens: 42,
    });
  });

  it("returns real usage unchanged (OpenAI shape)", () => {
    expect(addBufferToUsage({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 })).toEqual({
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
    });
  });

  it("formatUsage (estimated) does not add phantom tokens", () => {
    expect(formatUsage(100, 10, FORMATS.CLAUDE)).toMatchObject({ input_tokens: 100, output_tokens: 10 });
    expect(formatUsage(100, 10, FORMATS.OPENAI)).toMatchObject({ prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 });
  });

  it("estimateUsage reports its estimate without a buffer", () => {
    // A tiny body estimates a small positive count. Before the fix this came
    // back as estimate + 2000 (e.g. ~2006 for a 6-token body); assert it does not.
    const est = estimateUsage({ model: "m", messages: [{ role: "user", content: "hi" }] }, 4, FORMATS.OPENAI);
    expect(est.prompt_tokens).toBeGreaterThan(0);
    expect(est.prompt_tokens).toBeLessThan(100); // not bumped by the old +2000
  });
});
