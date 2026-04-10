# Temporal Network Analysis of Passing Dynamics in Association Football

SYDE 532: Introduction to Complex Systems — Final Project  
Author: Daud Khan, University of Waterloo

## AI Disclosure: 
The final analysis script (`analysis_main.py`) was developed with assistance from Claude (Anthropic) for code refactoring, statistical test implementation, and figure generation improvements. The original analysis pipeline was developed independently. All AI-assisted code was reviewed and validated by the author. The original `.ipynb` file written by the author is attached as well.

## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/daud-khan-96/SYDE-532-Project.git
cd SYDE-532-Project
```

### 2. Install Python dependencies

Requires Python 3.8+.

```bash
pip install -r requirements.txt
```

### 3. Download the StatsBomb open data

The analysis uses publicly available event data from the [StatsBomb Open Data](https://github.com/statsbomb/open-data) repository. Clone it into the project root:

```bash
git clone https://github.com/statsbomb/open-data.git
```

This places the data at `open-data/data/events/` and `open-data/data/matches/`, which is where the scripts expect it.

### 4. Create output directories

```bash
mkdir -p figures results tables
```

### 5. Run the analysis

Run the main analysis first (processes all 380 La Liga 2015/16 matches, generates season-wide figures and statistics, and produces the El Clasico case study figures):

```bash
python code/analysis_main.py
```

This takes a few minutes on the first run. It caches season-wide results to `results/season_results.csv` so subsequent runs are faster.

Then run the figure correction script (regenerates El Clasico and community figures with proper player name labels):

```bash
python code/fix_and_regenerate.py
```

All figures are saved to the `figures/` directory.

## Data

This project uses La Liga 2015/16 event data (380 matches, 760 team-match observations) from the StatsBomb Open Data repository, available under the StatsBomb Public Data license at https://github.com/statsbomb/open-data.

## License

The analysis code in this repository is provided for academic purposes. The underlying match data is subject to the [StatsBomb Open Data license](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
