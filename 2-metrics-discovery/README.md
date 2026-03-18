# vx-discovering-issues-claude

Discovering potential issues in Prolog programs by prompting Claude Sonnet 4.6 to identify it.

The process has two steps:

1. collect the issues
2. re-group them

After, we will design measures and test them to validate their relevance and utility.

## Analysis

Several files are important here.

- `analyze_fixes.ipynb` analyzes the fixes by computing the clusters of each fix (name + description) using similarity over their semantic embeddings
- `classify_fixes.ipynb` calls an LLM to produce a short label summarizing the fix. It clusterizes then to produce different groups.
- `merge_computed_fixes_labels.ipynb` merges the fixes obtained in the collection of the fixes with the tags produced by the LLM in `classify_fixes.ipynb`

## What is left to be done

- Actually analyze the fixes to produce a taxonomy of what we need to precisely check in a Prolog program.
- Test the measures we created to fill the gaps.
- Write one or more parts about this in the paper.
