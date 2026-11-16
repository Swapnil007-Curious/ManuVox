// WCAG AA check of the Duet palette in src/styles.css (both themes, glass over the video checked
// against black and white frames). Exits 1 when a pair fails. Node 22.6+ runs the TypeScript
// module directly (type stripping).
//
//   npm run contrast

import { readFileSync } from "node:fs";
import { auditStylesheet } from "../src/ui/contrast.ts";

const css = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const { results, problems } = auditStylesheet(css);
for (const result of results) console.log(`${result.pass ? "ok  " : "FAIL"} ${result.detail}`);
if (problems.length) {
  console.error(`\n${problems.length} problem(s):\n${problems.join("\n")}`);
  process.exit(1);
}
console.log(`\nall ${results.length} pairs pass AA`);
