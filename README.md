# Prolog Evaluation Metrics

This repository showcases the metrics implemented to evaluate a Prolog code.

Every experiment is reproducible and allows you to use different models to test yourself. Feel free to fork the project or raise issues.

## Organization

This repository is organized in several folders, each representing a step of our work.

- `1-prolog-generation`: includes all code realized to generate Prolog programs in the reasoning traces using an LRM.

- `2-metrics-discovery`: includes the code we wrote to discover issues and then create metrics over these.

- `3-metrics-evaluation`: includes all the stuff required to assess that our metrics are relevant (i.e. well-designed enough to cover the issues discovered in the previous step).

Additional files can help you edit key variables to change the experiments parameters with minimal effort:

- `config.py`: contains every key variable.

- `.env`: you should duplicate `.env.example` into a `.env` file to fill in your API keys. We are using OpenRouter in our case.

## Citation

Cite our work using:

> TODO
