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

## Prompts

Every prompt we designed is stored in a .j2 file.

## Citation

Cite our work using:

@inproceedings{haurel2026evaluation, 
  author = {Haurel, Maxime and Brun, Armelle and d'Aquin, Mathieu}, 
  title = {Towards an Evaluation Framework for Generated Formal Knowledge}, 
  booktitle = {LLMS4KGOE 2026 and ELMKE 2026: Joint Proceedings of the First Workshop on LLM-driven Knowledge Graph and Ontology Engineering and the Third Workshop on Evaluation of Language Models in Knowledge Engineering, co-located with ESWC 2026}, 
  editor = {Dalal, Aryan Singh and Jagodnik, Kathleen and Maleshkova, Maria and Shimizu, Cogan and Lippolis, Anna Sofia and Alharbi, Reham and Zhang, Bohui and He, Yuan and K{\"u}ç{\"u}k McGinty, Hande}, 
  series = {CEUR Workshop Proceedings}, 
  volume = {4246}, 
  pages = {177--191}, 
  year = {2026}, 
  month = may, 
  address = {Dubrovnik, Croatia}, 
  publisher = {CEUR-WS.org}, 
  url = {https://ceur-ws.org/Vol-4246/elmke-5.pdf} 
}

