# recommendation_engine
Repo for the recommendation engine that was part of the DRP project


## Recommender Pipeline

Steps to implement a recommender pipeline
(Specific implementation of this pipeline is available in ./recommender/recommender_pipeline.py)

1. Generate reaction features
    - Get the chemicals in a reaction. For DRP these are referred to as triples
    - Generate descriptors for each of the chemicals in the reaction
    - Generate a sampling grid of reaction parameters
    - Expand grid by associating descriptors with each point on the grid

2. Run trained models with the reaction Sieve
    - Get a trained machine learning model
    - Filter sampling grid by running it through the ML model
    - Make a list of all the potentially successful reactions
      as predicted by the ML model

3. Recommend reactions
    - Calculate the mutual information of the potential reactions
      as compared to the already completed reactions
    - Select the top 'k' reactions with the highest MI


### Progress

- [x] Generate Reaction features
- [x] Reaction Sieve
- [x] Reaction Recommender
- [ ] Test and evaluate against Nature paper