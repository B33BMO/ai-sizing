// Build a search-only copy of the Spero DB snapshot for the job pack: keep the corpus, its FTS/vector
// indexes and the knowledge graph; drop everything personal (conversations, memories, users, sessions,
// secrets, runs, usage...). Operates on a COPY; never opens the live DB.
const fs = require("fs");
const Database = require("/mnt/d/spero/365helper/node_modules/better-sqlite3");
const sqliteVec = require("/mnt/d/spero/365helper/node_modules/sqlite-vec");
const src = process.env.HOME + "/memlm-spero/sage.db";
const dst = process.env.HOME + "/memlm-spero/jobpack-kb/data/sage.db";
fs.mkdirSync(require("path").dirname(dst), { recursive: true });
fs.copyFileSync(src, dst);
const db = new Database(dst);
sqliteVec.load(db);
const KEEP = /^(docs|docs_fts.*|docs_vec.*|entities|entities_vec.*|edges|mentions|graph_status|sqlite_sequence)$/;
const tables = db.prepare("SELECT name, sql FROM sqlite_master WHERE type='table'").all();
// drop virtual tables first (their shadow tables go with them), then the rest
const virt = tables.filter(t => /^CREATE VIRTUAL/i.test(t.sql || "") && !KEEP.test(t.name));
for (const t of virt) db.exec(`DROP TABLE IF EXISTS "${t.name}"`);
for (const t of db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all())
  if (!KEEP.test(t.name)) db.exec(`DROP TABLE IF EXISTS "${t.name}"`);
db.exec("VACUUM");
const left = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all().map(t => t.name);
console.log("kept tables:", left.join(", "));
console.log("docs", db.prepare("select count(*) n from docs").get().n, "entities", db.prepare("select count(*) n from entities").get().n);
db.close();
console.log("size MB", (fs.statSync(dst).size / 1e6).toFixed(1));
