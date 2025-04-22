# Encoder Testing
Playground for good ways to load data and test out an encoder based LLM router

## Updates
### 3/31
Goals:
1. Working MVP for getting an encoder with MLP to predict the class
2. Come up with 10 possible experiments based on literature review
3. Integrate with running on a server (maybe personal AWS)

### 4/9
Overarching mission is the idea of the conformal router. I think it creates significant uniqueness to other approaches and handles uncertainty in some way which is good. Another approach is the routing recommender approach but then try all LLMs and choose one (or use an LLM Cascade approach) when a new embedding doesn't have enough near neighbors.

Let's focus on getting the MVP running for training and testing an encoder then work on pipelines

| Task | Status |
| ---- | ------ |
| Pipeline for embeddings | ✅ |
| Fake agent/model performance | ✅ |
| Working MVP for getting an encoder with MLP to predict the class | |
| Integrate with running on a server (maybe personal AWS) |  |
| Conformal Layer | |
| Come up with 10 possible experiments based on literature review |  |
| Decide on model list | |
| Dataset list | |
| Pipeline for generating pass@k results to data for models | |
