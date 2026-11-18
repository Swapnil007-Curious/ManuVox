// Teach Dex storage when IndexedDB is missing, blocked or failing: the app must keep working with
// examples in memory and say why (src/teachStore.ts). The IndexedDB path itself runs in the
// browser tests (tests/e2e/teach.spec.ts).

import { describe, expect, it } from "vitest";
import { normalize, type TaughtExample } from "../../src/personalization";
import { MemoryTeachStore, forModel, openTeachStore } from "../../src/teachStore";

function example(label: string, sha = "a".repeat(64), id = `${label}-${sha[0]}`): TaughtExample {
  return { id, label, embedding: normalize([1, 2, 3]), createdAt: 1, modelSha256: sha };
}

/** An IDBFactory whose open() request fires the given event (or throws). */
function fakeFactory(behaviour: "throws" | "error" | "blocked" | "silent"): IDBFactory {
  return {
    open() {
      if (behaviour === "throws") throw new DOMException("The operation is insecure.", "SecurityError");
      const request = { result: null, error: new DOMException("denied", "UnknownError") } as unknown as IDBOpenDBRequest;
      setTimeout(() => {
        if (behaviour === "error") request.onerror?.(new Event("error"));
        if (behaviour === "blocked") request.onblocked?.(new Event("blocked") as IDBVersionChangeEvent);
      }, 0);
      return request;
    },
  } as unknown as IDBFactory;
}

describe("openTeachStore falls back to memory", () => {
  it("without IndexedDB", async () => {
    const store = await openTeachStore(undefined);
    expect(store.mode).toBe("memory");
    expect(store.reason).toMatch(/no IndexedDB/);
  });

  it.each([
    ["throws", /insecure/],
    ["error", /denied/],
    ["blocked", /blocked/],
  ] as const)("when open() %s", async (behaviour, reason) => {
    const store = await openTeachStore(fakeFactory(behaviour));
    expect(store.mode).toBe("memory");
    expect(store.reason).toMatch(reason);
    await store.add(example("mom"));
    expect((await store.all()).map((item) => item.label)).toEqual(["mom"]);
  });

  it("when open() never answers", async () => {
    const store = await openTeachStore(fakeFactory("silent"));
    expect(store.mode).toBe("memory");
    expect(store.reason).toMatch(/in time/);
  }, 10_000);
});

describe("MemoryTeachStore", () => {
  it("adds, removes one sign and clears", async () => {
    const store = new MemoryTeachStore("test");
    await store.add(example("mom"));
    await store.add(example("dad"));
    await store.add({ ...example("mom"), id: "mom-2" });
    expect((await store.all()).length).toBe(3);
    await store.removeLabel("mom");
    expect((await store.all()).map((item) => item.label)).toEqual(["dad"]);
    await store.clear();
    expect(await store.all()).toEqual([]);
  });
});

it("keeps only the examples recorded with the loaded model", () => {
  const examples = [example("mom", "a".repeat(64)), example("dad", "b".repeat(64))];
  expect(forModel(examples, "a".repeat(64)).map((item) => item.label)).toEqual(["mom"]);
});
