# Modeling Inequities in EV Charging Access with Probabilistic Graphical Models

## Abstract

This project uses probabilistic graphical models to study uneven access to public electric vehicle charging infrastructure across the United States. I combine statewide charging port counts from the U.S. Department of Energy Alternative Fuels Data Center, 2023 EV registration counts, Census population estimates, Census SAIPE income and poverty estimates, and Census TIGERweb land area measures for the 50 states and the District of Columbia. A Gaussian mixture model identifies three latent state context profiles, and a Bayesian network represents probabilistic dependencies among region, latent context, income, poverty, population density, EV adoption, and charging access. The results show that high EV adoption is strongly associated with high public charging access: the fitted network estimates P(high charging access | high EV adoption) = 0.641, compared with 0.100 under low EV adoption. Leave one out prediction of charging access reaches 64.7 percent accuracy, above a 33.3 percent chance baseline. The model is descriptive rather than causal, but it shows how PGMs can organize evidence about infrastructure inequality and reveal interpretable state profiles.

## Repository Contents

- `index.qmd`: Final project manuscript.
- `scripts/build_analysis.py`: Reproducible data cleaning, modeling, evaluation, and figure generation script.
- `data/raw/`: Downloaded source files from AFDC and the U.S. Census Bureau.
- `data/processed/`: Cleaned modeling data, conditional probability tables, model metrics, and predictions.
- `images/`: Figures used in the manuscript.
- `docs/`: Static rendered report files for GitHub Pages.

## Reproduce

```bash
/Users/fanlanbin/miniforge3-codex/bin/python scripts/build_analysis.py
```
