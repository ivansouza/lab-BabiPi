// bev-web.js — bev-decider-0.4B rodando no navegador (onnxruntime-web + tokenizers.js).
// Reimplementa bev_decider/encode.py e decider.py (pergunta "choice" / "noul" / "score").
const TASK_TYPES = { choice: 0, noul: 1, score: 2 };
const TASK_PROMPTS = { choice: "Choose the best option.", score: "Rate it on the given scale.", noul: "Answer yes or no." };
const ANSWER_PROMPT = "The answer is:";
const HIDDEN = 1024;

const asText = (x) => (typeof x === "string" ? x : JSON.stringify(x, null, 2));

function questionOptions(q) {
  const kind = q.type, crit = q.criteria;
  if (kind === "choice") {
    if (!crit || !Object.keys(crit).length) throw new Error("a choice question needs criteria");
    const keys = Object.keys(crit);
    return { keys, options: keys.map((k) => (crit[k] ? `${k}: ${asText(crit[k])}` : k)) };
  }
  if (kind === "score") {
    if (!crit || !crit.length) throw new Error("a score question needs criteria");
    return { keys: crit.map((_, i) => String(i)), options: crit.map((l, i) => `${i}: ${asText(l)}`) };
  }
  if (kind === "noul") {
    const options = crit ? [`No: ${asText(crit.false)}`, `Yes: ${asText(crit.true)}`] : ["No", "Yes"];
    return { keys: ["false", "true"], options };
  }
  throw new Error(`unknown question type ${kind}`);
}

// Baixa um arquivo (possivelmente em pedaços) com progresso e Cache API
async function fetchCached(urls, cacheName, onBytes) {
  const cache = typeof caches !== "undefined" ? await caches.open(cacheName) : null;
  const parts = [];
  for (const url of urls) {
    let resp = cache ? await cache.match(url) : null;
    const fromCache = !!resp;
    if (!resp) {
      const net = await fetch(url);
      if (!net.ok) throw new Error(`HTTP ${net.status} em ${url}`);
      if (cache) { await cache.put(url, net.clone()); }
      resp = net;
    }
    const reader = resp.body.getReader();
    const chunks = []; let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); n += value.length; onBytes && onBytes(value.length, fromCache);
    }
    const buf = new Uint8Array(n); let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    parts.push(buf);
  }
  if (parts.length === 1) return parts[0];
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total); let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export class BevWeb {
  // Verifica se todos os arquivos do modelo já estão na Cache API
  static async cacheStatus({ base, cacheName = "bev-web-v1" }) {
    base = base.replace(/\/+$/, "") + "/";
    const manifest = await (await fetch(base + "manifest.json", { cache: "no-cache" })).json();
    const total = manifest.files.reduce((a, f) => a + f.size, 0);
    if (typeof caches === "undefined") return { cached: false, total };
    const cache = await caches.open(cacheName);
    for (const f of manifest.files) {
      const urls = f.parts ? f.parts.map((p) => base + p) : [base + f.name];
      for (const u of urls) if (!(await cache.match(u + "?v=" + f.sha.slice(0, 12)))) return { cached: false, total };
    }
    return { cached: true, total };
  }

  // base: URL da pasta com manifest.json; ort/Tokenizer: módulos já importados
  static async load({ base, ort, Tokenizer, onProgress, executionProviders, cacheName = "bev-web-v1" }) {
    base = base.replace(/\/+$/, "") + "/";
    const manifest = await (await fetch(base + "manifest.json", { cache: "no-cache" })).json();
    const total = manifest.files.reduce((a, f) => a + f.size, 0);
    let got = 0;
    const tick = (n, fromCache) => { got += n; onProgress && onProgress(got, total, fromCache); };
    const file = async (name) => {
      const f = manifest.files.find((x) => x.name === name);
      const urls = f.parts ? f.parts.map((p) => base + p) : [base + name];
      return fetchCached(urls.map((u) => u + "?v=" + f.sha.slice(0, 12)), cacheName, tick);
    };
    const dec = new TextDecoder();
    const tokJson = JSON.parse(dec.decode(await file("tokenizer.json")));
    const tokCfg = JSON.parse(dec.decode(await file("tokenizer_config.json")));
    const config = JSON.parse(dec.decode(await file("config.json")));
    const embQ = await file("embed_int8.bin");
    const embS = await file("embed_scale_f32.bin");
    const model = await file("bev_q8.onnx");
    const eps = executionProviders || ["wasm"];
    const session = await ort.InferenceSession.create(model, { executionProviders: eps, graphOptimizationLevel: "all" });
    const b = new BevWeb();
    Object.assign(b, {
      ort, session, config, tok: new Tokenizer(tokJson, tokCfg),
      emb: new Int8Array(embQ.buffer, embQ.byteOffset, embQ.byteLength),
      scale: new Float32Array(embS.buffer.slice(embS.byteOffset, embS.byteOffset + embS.byteLength)),
      maxState: config.max_state_tokens, maxChoice: config.max_choice_tokens, eps,
    });
    b.optBegin = b.ids("<option>"); b.optEnd = b.ids("</option>"); b.answer = b.ids(ANSWER_PROMPT);
    return b;
  }

  ids(text) { return this.tok.encode(text, { add_special_tokens: false }).ids; }
  truncate(text, max) {
    const t = this.ids(text);
    return t.length <= max ? text : this.tok.decode(t.slice(0, max), { skip_special_tokens: false });
  }

  encode(state, question) {
    const { options, keys } = questionOptions(question);
    const prompt = `${TASK_PROMPTS[question.type]}\nQuestion: ${asText(question.instructions)}\n\n` +
      `${this.truncate(asText(state), this.maxState)}\n`;
    const P_ids = this.ids(prompt);
    const optTok = options.map((o) => [...this.optBegin, ...this.ids(this.truncate(o, this.maxChoice)), ...this.optEnd]);
    const input_ids = [...P_ids], position_ids = P_ids.map((_, i) => i), spans = [];
    for (const t of optTok) {
      spans.push([input_ids.length, input_ids.length + t.length]);
      input_ids.push(...t);
      for (let i = 0; i < t.length; i++) position_ids.push(P_ids.length + i);
    }
    const answerStart = input_ids.length;
    let mx = Math.max(...position_ids);
    input_ids.push(...this.answer);
    for (let i = 0; i < this.answer.length; i++) position_ids.push(mx + 1 + i);
    const L = input_ids.length, P = P_ids.length;
    const mask = new Float32Array(L * L).fill(-Infinity);
    const allow = (r, c) => { mask[r * L + c] = 0; };
    for (let r = 0; r < P; r++) for (let c = 0; c <= r; c++) allow(r, c);
    for (const [bg, en] of spans) for (let r = bg; r < en; r++) {
      for (let c = 0; c < P; c++) allow(r, c);
      for (let c = bg; c <= r; c++) allow(r, c);
    }
    for (let r = answerStart; r < L; r++) for (let c = 0; c <= r; c++) allow(r, c);
    return {
      input_ids, position_ids, mask, L, keys,
      choice_read_idx: spans.map(([, en]) => en - 1 - this.optEnd.length),
      answer_end_idx: L - 1, task_type: TASK_TYPES[question.type],
    };
  }

  async probabilities(ex) {
    const { ort } = this; const L = ex.L, C = ex.choice_read_idx.length;
    const embeds = new Float32Array(L * HIDDEN);
    ex.input_ids.forEach((id, t) => {
      const s = this.scale[id], off = id * HIDDEN, o = t * HIDDEN;
      for (let h = 0; h < HIDDEN; h++) embeds[o + h] = this.emb[off + h] * s;
    });
    const big = (a) => BigInt64Array.from(a.map((x) => BigInt(x)));
    const feeds = {
      inputs_embeds: new ort.Tensor("float32", embeds, [1, L, HIDDEN]),
      position_ids: new ort.Tensor("int64", big(ex.position_ids), [1, L]),
      attention_mask: new ort.Tensor("float32", ex.mask, [1, 1, L, L]),
      choice_read_idx: new ort.Tensor("int64", big(ex.choice_read_idx), [1, C]),
      choice_mask: new ort.Tensor("bool", new Uint8Array(C).fill(1), [1, C]),
      answer_end_idx: new ort.Tensor("int64", big([ex.answer_end_idx]), [1]),
      task_type: new ort.Tensor("int64", big([ex.task_type]), [1]),
    };
    const out = await this.session.run(feeds);
    const lg = Array.from(out.logits.data);
    const m = Math.max(...lg); const e = lg.map((x) => Math.exp(x - m)); const z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  }

  // Mesmo formato do servidor: decide(state, {id: question}) -> {id: answer}
  async decide(state, questions) {
    const answers = {};
    for (const [qid, q] of Object.entries(questions)) {
      const ex = this.encode(state, q);
      const p = await this.probabilities(ex);
      const probs = Object.fromEntries(ex.keys.map((k, i) => [k, p[i]]));
      if (q.type === "noul") answers[qid] = { type: "noul", noul: p[1] };
      else if (q.type === "choice") {
        let best = 0; p.forEach((x, i) => { if (x > p[best]) best = i; });
        answers[qid] = { type: "choice", choice: ex.keys[best], probabilities: probs };
      } else answers[qid] = { type: "score", score: p.reduce((a, x, i) => a + i * x, 0), probabilities: probs };
    }
    return answers;
  }
}
