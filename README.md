# recommendation_engine
Repo for the recommendation engine that was part of the DRP project


## Recommender Pipeline

1. Generate Reaction Features
   - Get triples of reaction (A "triple" is a set of 3 chemicals that constitute a reaction)
   - Generate descriptors for the reaction (i.e. for the 3 chemicals and reaction parameters)

2. Run trained models / Reaction Sieve
   - ReactionSieve (model, whitelist)
   - plausible_reactions = ReactionSieve.filter(reactions from Step 1)

3. Recommend Reactions
   - Filter plausible_reactions through Mutual Information hashing?
   - Get top 'k' reactions from previous step
   - Recommend selected reactions

### Progress

- [ ] Generate Reaction features
- [ ] Reaction Sieve
- [ ] Reaction Recommender