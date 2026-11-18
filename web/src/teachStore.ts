// Teach Dex storage: the user's examples (label + embedding) in IndexedDB on this device. Nothing
// here touches the network. When IndexedDB is missing, blocked or failing (private windows,
// storage blocked by the browser, a test page), the store keeps the examples in memory for the
// session and says so through `mode` and `reason`.

import type { TaughtExample } from "./personalization";

export const DB_NAME = "edge-vision-teach";
export const DB_VERSION = 1;
export const STORE_NAME = "examples";
const OPEN_TIMEOUT_MS = 3000;

export interface TeachStore {
  readonly mode: "indexeddb" | "memory";
  /** Why the store fell back to memory; null for IndexedDB. */
  readonly reason: string | null;
  /** Every stored example, of any model. */
  all(): Promise<TaughtExample[]>;
  add(example: TaughtExample): Promise<void>;
  removeLabel(label: string): Promise<void>;
  clear(): Promise<void>;
}

interface StoredExample {
  id: string;
  label: string;
  embedding: number[];
  createdAt: number;
  modelSha256: string;
}

function toStored(example: TaughtExample): StoredExample {
  return { ...example, embedding: Array.from(example.embedding) };
}

function fromStored(stored: StoredExample): TaughtExample | null {
  if (typeof stored?.label !== "string" || !Array.isArray(stored.embedding) || typeof stored.modelSha256 !== "string") return null;
  return { ...stored, embedding: Float32Array.from(stored.embedding) };
}

export class MemoryTeachStore implements TeachStore {
  readonly mode = "memory" as const;
  private readonly examples = new Map<string, TaughtExample>();

  constructor(readonly reason: string) {}

  async all(): Promise<TaughtExample[]> {
    return [...this.examples.values()];
  }

  async add(example: TaughtExample): Promise<void> {
    this.examples.set(example.id, example);
  }

  async removeLabel(label: string): Promise<void> {
    for (const [id, example] of this.examples) if (example.label === label) this.examples.delete(id);
  }

  async clear(): Promise<void> {
    this.examples.clear();
  }
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
  });
}

class IndexedDbTeachStore implements TeachStore {
  readonly mode = "indexeddb" as const;
  readonly reason = null;

  constructor(private readonly db: IDBDatabase) {}

  private async run<T>(mode: IDBTransactionMode, body: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const transaction = this.db.transaction(STORE_NAME, mode);
    const done = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
      transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    });
    const result = await request(body(transaction.objectStore(STORE_NAME)));
    await done;
    return result;
  }

  async all(): Promise<TaughtExample[]> {
    const rows = await this.run("readonly", (store) => store.getAll() as IDBRequest<StoredExample[]>);
    return rows.map(fromStored).filter((example): example is TaughtExample => example !== null);
  }

  async add(example: TaughtExample): Promise<void> {
    await this.run("readwrite", (store) => store.put(toStored(example)));
  }

  async removeLabel(label: string): Promise<void> {
    const ids = (await this.all()).filter((example) => example.label === label).map((example) => example.id);
    for (const id of ids) await this.run("readwrite", (store) => store.delete(id));
  }

  async clear(): Promise<void> {
    await this.run("readwrite", (store) => store.clear());
  }
}

function openDatabase(factory: IDBFactory): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("IndexedDB did not open in time")), OPEN_TIMEOUT_MS);
    const finish = (action: () => void): void => {
      clearTimeout(timer);
      action();
    };
    let req: IDBOpenDBRequest;
    try {
      req = factory.open(DB_NAME, DB_VERSION);
    } catch (error) {
      finish(() => reject(error instanceof Error ? error : new Error(String(error))));
      return;
    }
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE_NAME)) req.result.createObjectStore(STORE_NAME, { keyPath: "id" });
    };
    req.onsuccess = () => finish(() => resolve(req.result));
    req.onerror = () => finish(() => reject(req.error ?? new Error("IndexedDB could not be opened")));
    req.onblocked = () => finish(() => reject(new Error("IndexedDB is blocked by another tab")));
  });
}

/** The page's IndexedDB; reading it can itself throw where storage is blocked. */
function pageIndexedDb(): IDBFactory | undefined {
  try {
    return globalThis.indexedDB;
  } catch {
    return undefined;
  }
}

/**
 * IndexedDB when it works, else a memory store that explains why. Never throws. ``factory``
 * defaults to the page's IndexedDB; pass null to force memory (tests).
 */
export async function openTeachStore(factory: IDBFactory | null | undefined = pageIndexedDb()): Promise<TeachStore> {
  if (!factory) return new MemoryTeachStore("this browser has no IndexedDB, or blocks it");
  try {
    const db = await openDatabase(factory);
    const store = new IndexedDbTeachStore(db);
    await store.all(); // A store that cannot be read is no better than none.
    return store;
  } catch (error) {
    return new MemoryTeachStore(error instanceof Error ? error.message : String(error));
  }
}

/** Examples recorded with ``modelSha256``; the others cannot be compared with this model. */
export function forModel(examples: readonly TaughtExample[], modelSha256: string): TaughtExample[] {
  return examples.filter((example) => example.modelSha256 === modelSha256);
}
