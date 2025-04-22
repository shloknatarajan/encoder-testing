## Proposed Baselines
- Martian Router
- FrugalGPT
- Oracle Router
- Single Model Alone

## Speed up testing
Run all of the models in your ensemble against the benchmark and save all the model generations / outputs.
After that the router doesn't need to actually generate new answers, just use the pre-existing ones. Can also generate pass@k and then randomly choose one of those when doing tests to simulate some variance.
