# bev-web

[avbiswas/bev-decider-0.4B](https://huggingface.co/avbiswas/bev-decider-0.4B) exportado para ONNX para rodar no navegador
(onnxruntime-web, WASM). Usado pelo `campo-minado.html` no modo "IA no navegador".

- `bev_q8.onnx.partNN`: backbone Qwen3 truncado (20 camadas) + cabeça de decisão, pesos MatMulNBits 8 bits (bloco 32).
- `embed_int8.bin.partNN` + `embed_scale_f32.bin`: tabela de embeddings int8 por linha (lookup feito em JS).
- `bev-web.js`: reimplementação do encode.py/decider.py (máscara de atenção das opções, position ids, softmax).
- Arquivos divididos em pedaços < 100 MB por causa do limite do GitHub; `manifest.json` lista tamanhos e sha256.

Paridade com o servidor Python (fp32), 25 prompts: mesma resposta em 25/25, diferença máx. de probabilidade 0,03.
4 bits foi testado e descartado (só 41-43/50 respostas iguais).
Licença do modelo: ver o repositório original no Hugging Face (base Qwen3-0.6B, Apache-2.0).
