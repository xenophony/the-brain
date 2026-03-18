=== VAST.AI QUICKSTART FOR THE-BRAIN PROJECT ===

1. RENT INSTANCE
   - GPU: RTX 5090 (32GB VRAM)
   - Container: 20GB
   - Volume: 100GB
   - Template: PyTorch (vast.ai)
   - Avoid hosts: 296571, 344939 (bad HuggingFace routing)
   - Check download speed before committing:

```bash
curl -o /dev/null -s -w "Speed: %{speed_download} bytes/sec\n" https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/README.md
```
   Anything below 10000 bytes/sec → destroy and try another host


2. BOOTSTRAP (run in Jupyter terminal)

```bash
cd /workspace && git clone https://github.com/xenophony/the-brain.git brain && cd brain && bash scripts/bootstrap_cloud.sh && pip install tokenizers
```


3. HUGGINGFACE LOGIN

```bash
python3 -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN_HERE'); print('Login successful')"
```


4. DOWNLOAD MODEL (~18GB, takes 1-3 min at good speed)

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('turboderp/Qwen3-30B-A3B-exl2', revision='4.0bpw', local_dir='models/Qwen3-30B-A3B-exl2'); print('Done')"
```


5. VALIDATE ADAPTER

```bash
python3 -c "from sweep.exllama_adapter import ExLlamaV2LayerAdapter; a = ExLlamaV2LayerAdapter('models/Qwen3-30B-A3B-exl2'); print('layers:', a.num_layers); out = a.generate_short('What is 2+2?', max_new_tokens=5); print('output:', out)"
```
   Expected: layers=48, output contains '4'


6. SMOKE TEST (5-10 min)

```bash
python scripts/run_sweep.py --model models/Qwen3-30B-A3B-exl2 --probes causal_logprob logic_logprob sentiment_logprob --mode duplicate --max-layers 6 --max-block 2 --output results/smoke
```


7. START FULL SWEEP (duplicate)

```bash
mkdir -p logs && nohup python scripts/run_sweep.py --model models/Qwen3-30B-A3B-exl2 --probes causal_logprob logic_logprob sentiment_logprob error_logprob routing_logprob judgement_logprob sycophancy_logprob --mode duplicate --output results/tier1_reasoning > logs/tier1_reasoning.log 2>&1 & echo "PID: $!"
```

```bash
tail -f logs/tier1_reasoning.log
```


8. START SKIP SWEEP

```bash
mkdir -p logs && nohup python scripts/run_sweep.py --model models/Qwen3-30B-A3B-exl2 --probes causal_logprob logic_logprob sentiment_logprob error_logprob routing_logprob judgement_logprob sycophancy_logprob --mode skip --output results/tier1_skip > logs/tier1_skip.log 2>&1 & echo "PID: $!"
```

```bash
tail -f logs/tier1_skip.log
```


9. START SKIP SWEEP WITH PSYCH CAPTURE

```bash
mkdir -p logs && nohup python scripts/run_sweep.py --model models/Qwen3-30B-A3B-exl2 --probes causal_logprob logic_logprob sentiment_logprob error_logprob routing_logprob judgement_logprob sycophancy_logprob --mode skip --psych --output results/tier1_skip_psych > logs/tier1_skip_psych.log 2>&1 & echo "PID: $!"
```

```bash
tail -f logs/tier1_skip_psych.log
```


=== MONITORING ===

Check progress:
```bash
tail -f logs/tier1_reasoning.log
```

Check GPU:
```bash
watch -n 2 'nvidia-smi | grep -E "MiB|Util|W"'
```

Check results:
```bash
python3 -c "import json; data = json.load(open('results/tier1_reasoning/sweep_results.json')); print(f'Configs: {len(data)}'); scores = [(r['i'],r['j'],r['probe_scores'].get('causal_logprob',0)) for r in data]; scores.sort(key=lambda x: x[2], reverse=True); [print(f'  ({i},{j}) = {s:.3f}') for i,j,s in scores[:5]]"
```

Check psych profiles (if --psych was used):
```bash
python3 -c "import json; data = json.load(open('results/tier1_skip_psych/psych_profiles.json')); print(f'Profiles captured: {len(data)}')"
```

Run psych analysis:
```bash
python analysis/analyze_psych_profiles.py --results results/tier1_skip_psych
```


=== NOTES ===
- hf-xet protocol used automatically — much faster than standard HTTPS
- Must be logged in to HuggingFace for full speed (authenticated xet)
- If bootstrap slow on pip, just wait — it retries automatically
- CUDA graph caching disabled (no_graphs=True) — required for sweep correctness
- Model loads to ~18GB VRAM, leaves ~14GB for inference
- Keep volume when destroying instance (~$0.10/day) — avoids re-downloading model
- To keep volume: click Destroy → say NO to "Delete volume?"
- Reattach volume on next instance via vast.ai dashboard
- --psych flag adds psycholinguistic capture as parallel dataset, does not affect sweep results