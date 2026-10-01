// Regression: the `[1m]` context marker must be auto-applied to the PRIMARY
// model slot (ANTHROPIC_MODEL), not only the per-tier ANTHROPIC_DEFAULT_* slots.
//
// Claude Code assumes a 200K window unless the model name carries `[1m]`. A 1M
// CodeBuddy model mapped to ANTHROPIC_MODEL (the default slot the dashboard
// writes) was therefore treated as 200K — so ~554K real tokens read as
// "100% context used" even though the model's window is 1M.
//
// The route is a Next.js "use server" file (not importable in a unit test), so
// assert the source wires ANTHROPIC_MODEL into the auto-mark key list, alongside
// the behavioral checks of the pure helpers.
import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";
import { shouldMarkOneMContext, withContextMarker } from "../../open-sse/utils/modelMarkers.js";
import { getCapabilitiesForModel } from "../../open-sse/providers/capabilities.js";

const routeSrc = fs.readFileSync(
  path.resolve("../src/app/api/cli-tools/claude-settings/route.js"),
  "utf-8"
);

describe("claude-settings auto-marks ANTHROPIC_MODEL for 1M models", () => {
  it("lists ANTHROPIC_MODEL in MODEL_ENV_KEYS", () => {
    const block = routeSrc.match(/const MODEL_ENV_KEYS\s*=\s*\[([\s\S]*?)\]/);
    expect(block).toBeTruthy();
    expect(block[1]).toContain('"ANTHROPIC_MODEL"');
  });

  it("marks a 1M model and leaves a sub-1M model alone", () => {
    const aliases = { cbai: "codebuddy-intl", cbcn: "codebuddy-cn" };
    const resolve = (prefix, model) =>
      getCapabilitiesForModel(prefix ? (aliases[prefix] || prefix) : null, model)?.contextWindow;

    // deepseek-v4.1-flash = 1M → marked; glm-5.1 = 200K → untouched.
    expect(shouldMarkOneMContext("cbai/deepseek-v4.1-flash", resolve)).toBe(true);
    expect(withContextMarker("cbai/deepseek-v4.1-flash", true)).toBe("cbai/deepseek-v4.1-flash[1m]");
    expect(shouldMarkOneMContext("cbai/glm-5.1", resolve)).toBe(false);
  });

  it("is idempotent (never stacks [1m][1m])", () => {
    expect(withContextMarker("cbai/deepseek-v4.1-flash[1m]", true)).toBe("cbai/deepseek-v4.1-flash[1m]");
  });
});
