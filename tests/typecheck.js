// Type-check the front end with the TypeScript compiler, as VS Code does
// with the same jsconfig. usage: node typecheck.js <typescript.js> <jsconfig.json>
const ts = require(process.argv[2]);
const path = require("path");
const configPath = path.resolve(process.argv[3]);
const read = ts.readConfigFile(configPath, ts.sys.readFile);
if (read.error) { console.error(ts.flattenDiagnosticMessageText(read.error.messageText, "\n")); process.exit(2); }
const parsed = ts.parseJsonConfigFileContent(read.config, ts.sys, path.dirname(configPath));
const program = ts.createProgram(parsed.fileNames, parsed.options);
const diags = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
for (const d of diags) {
  const msg = ts.flattenDiagnosticMessageText(d.messageText, "\n");
  if (d.file) {
    const { line, character } = d.file.getLineAndCharacterOfPosition(d.start);
    console.log(`${path.relative(process.cwd(), d.file.fileName)}:${line + 1}:${character + 1}  TS${d.code}  ${msg}`);
  } else {
    console.log(`TS${d.code}  ${msg}`);
  }
}
console.log(`${parsed.fileNames.length} files, ${diags.length} diagnostics (TypeScript ${ts.version})`);
process.exit(diags.length ? 1 : 0);
