// Reads a JSON array of strings from stdin, embeds each with the real,
// published @ternlight/base package (generic, English, general-purpose
// distillation), writes a JSON array of 384-dim vectors to stdout.
// This is the actual shipped model, not a reimplementation.
import { embed } from '@ternlight/base';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const texts = JSON.parse(Buffer.concat(chunks).toString('utf8'));

const vecs = texts.map((t) => Array.from(embed(t)));
process.stdout.write(JSON.stringify(vecs));
