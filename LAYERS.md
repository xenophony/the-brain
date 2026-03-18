 What was built                                                                                                                                                         
                                                                                                                                                                         
  1. Multi-token psycholinguistic scoring fix                                                                                                                          
                                                                                                                                                                         
  probes/psycholinguistics.py — build_psych_token_map() now returns full subword token sequences per word. score_psych_vocab_from_logits() uses geometric mean of subword
   probabilities for multi-token words (e.g., "approximately" → exp(mean(log_p("approx"), log_p("im"), log_p("ately")))). Backward compatible — old token maps still     
  work.                                                                                                                                                                  
                                                                                                                                                                       
  2. Per-layer adapter method                                                                                                                                            
   
  sweep/exllama_adapter.py — New get_layerwise_logprobs(prompt, targets, psych_token_map) method. Uses forward_with_hooks to capture hidden states at every layer,       
  projects each through norm+lm_head to get vocab-space probabilities. Returns per-layer logprobs for answer choices AND psycholinguistic signals.                     
                                                                                                                                                                         
  3. Layerwise probe framework                                                                                                                                         

  probes/layerwise_registry.py — BaseLayerwiseProbe with built-in analysis:                                                                                              
  - Convergence detection — finds layer where p(correct) stabilizes
  - Computation region — steepest p(correct) rise (Kadane's algorithm)                                                                                                   
  - Correct vs incorrect psych comparison — do hedging/confidence signals predict correctness?                                                                         
  - Per-layer entropy — decision certainty across layers                                                                                                                 
  - Surprise detection — dramatic layer-to-layer probability changes                                                                                                     
                                                                                                                                                                         
  4. 20 layerwise probe wrappers                                                                                                                                         
                                                                                                                                                                         
  probes/layerwise/ — One for each existing logprob probe, reusing their ITEMS/CHOICES. Sycophancy probe has custom dual-condition tracing (neutral vs pressure at each  
  layer).                                                                                                                                                                
                                                                                                                                                                         
  5. Analysis module                                                                                                                                                     
   
  analysis/layerwise_analysis.py — Cross-probe analysis: shared computation regions, convergence comparison, psych peak/trough detection, correctness correlation.       
                                                                                                                                                                       
  6. CLI runner                                                                                                                                                          
                                                                                                                                                                       
  scripts/run_layerwise.py — Full CLI with --mock, --probes, --compare i,j, --layer-path i,j, --no-psych.                                                                
   
  7. MockAdapter support                                                                                                                                                 
                                                                                                                                                                       
  sweep/mock_adapter.py — get_layerwise_logprobs() with mode-specific synthetic curves (sigmoidal rise, sycophantic drop, etc.).                                         
                                                                                                                                                                       
  Blockers (require real GPU model)                                                                                                                                      
                                                                                                                                                                       
  1. get_layerwise_logprobs projects layers sequentially through post_modules. If too slow on 48 layers, batched projection (stacking hidden states) should be attempted 
  after reading ExLlamaV2 RMSNorm/HeadModule source.                                                                                                                   
  2. Device handling follows existing project_to_vocab pattern but needs validation on real GPU.                                                                         
  3. Psych token ID resolution needs validation with actual Qwen3 tokenizer.                                                                                             
                                                                                                                                                                         
  Usage                                                                                                                                                                  
                                                                                                                                                                         
  # Mock test                                                                                                                                                            
  python scripts/run_layerwise.py --mock --probes causal,logic,sentiment                                                                                                 
                                                                                                                                                                         
  # Real model                                                                                                                                                           
  python scripts/run_layerwise.py --model models/Qwen3-30B-A3B --probes causal,logic                                                                                     
                                                                                                                                                                         
  # Compare baseline vs specific (i,j) config                                                                                                                            
  python scripts/run_layerwise.py --model models/Qwen3-30B-A3B --compare 2,5               