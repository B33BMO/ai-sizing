// Long-lived JSONL tool server over Spero's real retrieval code, pointed at a DB snapshot.
// Lives outside the Spero repo; run from the repo root so tsconfig/.env resolve:
//   cd /mnt/d/spero/365helper && SAGE_DB_PATH=~/memlm-spero/sage.db npx tsx ~/memlm-spero/kb-serve.ts
// stdin : {"id":1,"op":"search","q":"...","k":6} | {"op":"graph","entity":"..."} | {"op":"path","from":"..","to":".."}
// stdout: {"id":1,"ok":true,"text":"..."}  (exactly the text Spero's own model sees)
import "/mnt/d/spero/365helper/scripts/_chdir-root";
import "/mnt/d/spero/365helper/scripts/_env";
import * as readline from "node:readline";
import { formatHitsForModel, searchDocs } from "/mnt/d/spero/365helper/lib/rag/search";
import { exploreGraph } from "/mnt/d/spero/365helper/lib/tools/explore-graph";

if (!process.env.SAGE_DB_PATH) throw new Error("refusing to run without SAGE_DB_PATH (would open the live DB)");

async function handle(req: any): Promise<string> {
  if (req.op === "search") return formatHitsForModel(await searchDocs(String(req.q), Math.min(20, req.k ?? 6)));
  if (req.op === "graph") return String(await exploreGraph({ action: "entity", entity: String(req.entity) }));
  if (req.op === "path") return String(await exploreGraph({ action: "path", from: String(req.from), to: String(req.to) }));
  throw new Error(`unknown op ${req.op}`);
}

const rl = readline.createInterface({ input: process.stdin });
const queue: Promise<void>[] = [];
rl.on("line", (line) => {
  if (!line.trim()) return;
  let req: any;
  try { req = JSON.parse(line); } catch { return; }
  // serialize requests: one embedding model, one DB handle
  const prev = queue.length ? queue[queue.length - 1] : Promise.resolve();
  queue.push(prev.then(async () => {
    try {
      process.stdout.write(JSON.stringify({ id: req.id, ok: true, text: await handle(req) }) + "\n");
    } catch (e) {
      process.stdout.write(JSON.stringify({ id: req.id, ok: false, text: String((e as Error).message) }) + "\n");
    }
  }));
});
rl.on("close", () => Promise.all(queue).then(() => process.exit(0)));
process.stderr.write("kb-serve ready\n");
