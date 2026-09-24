import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadNodeBundle } from "../src/providers.js";
import { DEFAULT_MODELS, Router } from "../src/router.js";

describe("model integrity", () => {
  it("pins published model revisions and propagates them to loaders", async () => {
    expect(DEFAULT_MODELS.english.revision).toMatch(/^[0-9a-f]{40}$/);
    let received: unknown;
    const router = new Router({
      loader: (_name, spec) => {
        received = spec;
        return {};
      },
    });
    await router.load("multilingual");
    expect((received as { revision?: string }).revision).toBe(DEFAULT_MODELS.multilingual.revision);
  });

  it("verifies local config and tokenizer digests", async () => {
    const root = await mkdtemp(join(tmpdir(), "laya-integrity-"));
    try {
      await mkdir(join(root, "tokenizer"));
      const config = JSON.stringify({ max_len: 512 });
      const tokenizer = JSON.stringify({ model: { vocab: {}, merges: [] } });
      await writeFile(join(root, "rl_agent_config.json"), config);
      await writeFile(join(root, "tokenizer", "tokenizer.json"), tokenizer);
      const digest = (value: string) => createHash("sha256").update(value).digest("hex");

      const bundle = await loadNodeBundle(root, {
        digests: { config: digest(config), tokenizer: digest(tokenizer) },
      });
      expect(bundle.cfg.max_len).toBe(512);
      expect(bundle.tokenizerJson).not.toBeNull();

      await expect(loadNodeBundle(root, { digests: { config: "0".repeat(64) } })).rejects.toThrow("SHA-256 mismatch");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
