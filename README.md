## Does AI Have a Theory of Mind (ToM)? Evaluating LLM performance on video-based ToM tasks. 

Authors: Edward Bullock 

## Abstract 
We evaluate five large language models (Gemini 3 Pro, Gemini 3 Flash, GPT-5, GPT-5 Mini, Vlaude Opus 4.5) on two mental state recofnition datasets, EU-Emotion (27 mental states) and Mindreading (357 mental states), using a four alternative forced-choice paradigm. 

## Repository structure 
- **'models/'**: Model wrappers/adapters and share inference utilities for eeach LLM provider 
- **'experiments/'**: Experiment runners and configuration for the forced choice evaluation pipeline 
- **'analysis/'**: Statistical analyses, figures table generation scripts, and any post-processing code. 
- **'prompts/'**: Prompt templates and prompt variants used for each model and condition 
- **'data/'**: Local dataset mount point (not included in this repository); expexted input file layouts and helper scritps
- **'results/'**: Generated outputs (metics, logs, intermediate artifacts). The ddirectory is tracked, but large files are ignored by git. 
- **'cache/'**: Local cahces (e.g., downloaded assets, oreoricessed featyres, API response cachces). Not intended for version control. 

## Installation 
reate a Python enviroment and install dependencies

'''bash 
pip instal -r requirements.txt 
'''

create a '.env' file in the repository root with your API keys, for example:

'''Bash 
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
'''

## Usage 
Run the evaluation entrypoint:

'''bash 
python experiments/run_evaluation.py
'''

## Data availability 
The EU-Emotion and Mindreading stimuli cannot be included in this repository due to library size and liscencing restrictions. Please reach out to the cambridge autismresearchcentre.com for access. This is free for research purposes. 

- EU_Emotion Dataset: O'Reilly, H., Pigat, D., Fridenson, S., Berggren, S., Tal, S., Golan, O., Bölte, S., Baron-Cohen, S., & Lundqvist, D. (2016). The EU-Emotion Stimulus Set: A validation study. Behavior Research Methods, 48(2), 567-576. https://doi.org/10.3758/s13428-015-0601-4
- Mindreading library: Baron-Cohen, S., Golan, O., Wheelwright, S., & Hill, J. J. (2004). Mind Reading: The Interactive Guide to Emotions [DVD-ROM]. Jessica Kingsley Publishers.

'''bibtex
@article{
    title = {Evaluating Large Language Models on Video-Based Theory-of-Mind Tasks: Comparison to Autistic and Non-Autistic People}
    author = {Edward Bullock}
    Journal = {TBD, will be updated when available}
    year = {TBD, will be updated when available}
}
'''

## Liscence 
MIT