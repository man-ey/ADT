# Context-Based Attentive Decision Tree Extraction for Atari Games ADT

## Installation And Requirements

This repository requires Python 3.8.x and specific dependencies. The setup process involves creating a virtual environment and installing the required packages.

```bash
# Create Python 3.8 virtual environment
python3.8 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Downloading Required Resources

The repository requires two sets of pre-trained models and data files. Use the following commands to download and extract them:

### Download and Extract Models
```bash
# Download first resource package
wget https://next.hessenbox.de/public.php/dav/files/tdyaS3Jes7friQm -O resource_package_1.zip
unzip resource_package_1.zip
rm resource_package_1.zip

# Download second resource package  
wget https://next.hessenbox.de/public.php/dav/files/oBCxeDTqecLeJSy -O resource_package_2.zip
unzip resource_package_2.zip
rm resource_package_2.zip
```

## How To Use

There are two main Python scripts that can be run directly. Each has a `-h` help flag for detailed parameter information.

### Extracting Decision Trees (Extractor)

The extractor script trains context-based decision trees using the DAGGER algorithm. It supports both Freeway and Tennis Atari games.

#### Basic Usage:
```bash
python extractor.py -game freeway -seed 0 -depth 5 -k 3
```

#### Full Options Extraction Example:
```bash
python extractor.py -game tennis -seed 42 -depth 8 -k 5 -oblique -batch 50000 -iterations 5 -distance l2 -output custom_trees/ -eval-episodes 20
```

#### Key Parameters:
- `-game`: Choose between "freeway" or "tennis"
- `-seed`: Random seed for reproducibility (default: 0)
- `-depth`: Maximum depth of decision trees (default: 5)
- `-k`: Number of objects per context (default: 3)
- `-oblique`: Use oblique decision boundaries
- `-batch`: DAGGER batch size of sample collection
- `-iterations`: Number of refinement iterations
- `-distance`: Distance function if a special one is desired
- `-output`: Output directory for saved trees (default: "context_adt_nearest/")
- `-pruned`: Use pruned environment

### Loading and Evaluating Decision Trees (Loader)

The loader script loads pre-trained decision trees and evaluates their performance in the specified game environment.

#### Basic Usage:
```bash
python loader.py -dir path/to/trees -game freeway -episodes 10
```

#### Advanced Evaluation Example:
```bash
python loader.py -dir context_adt_nearest/ -game tennis -episodes 50 -video -seed 42 -pruned -verbose -k 5 -pruned
```

#### Key Parameters:
- `-dir`: Path to directory containing tree files (adt_*.joblib) - **Required**
- `-game`: Choose between "freeway" or "tennis" - **Required**
- `-episodes`: Number of episodes to evaluate (default: 10)
- `-video`: Generate visualization video
- `-seed`: Seed for the game environment (default: 0)
- `-pruned`: Use pruned environment
- `-verbose`: Debugging prints enabler
- `-k`: Objects per context accounted for during tree extractions

## Workflow Example

Here's a complete workflow for training and evaluating decision trees:

### 1. Extract Decision Trees
```bash
# Extract trees for Freeway game
python extractor.py -game freeway -seed 0 -depth 5 -k 3 -oblique -iterations 10 -output freeway_trees/ -pruned

# Extract trees for Tennis game
python extractor.py -game tennis -seed 0 -depth 10 -k 4 -batch 6000 -output tennis_trees/ -pruned
```

### 2. Evaluate Trained Trees
```bash
# Evaluate Freeway trees
python loader.py -dir freeway_trees/ -game freeway -episodes 10 -verbose -pruned

# Evaluate Tennis trees with video generation
python loader.py -dir tennis_trees/ -game tennis -episodes 5 -video -verbose -pruned
```

## Output Structure

### Training Output
- Trees are saved as `.joblib` files with naming convention `adt_*.joblib`
- Default output directory: `context_adt_nearest/`
- Training logs include mean returns and standard deviations

### Evaluation Output
- Performance metrics printed to console
- Optional video generation showing agent behavior
- Visualization of decision tree performance over episodes

## Game-Specific Contexts
Pre-delivered and currently working
### Freeway
- **Top context**: Cars 6-10 (`Car6` through `Car10`)
- **Bottom context**: Cars 1-5 (`Car1` through `Car5`)
- Unplayable character `Chicken2` is automatically removed

### Tennis
- **Top/Bottom contexts**: Ball, Enemy, and Ball Shadow (`Ball1`, `Enemy1`, `BallShadow1`)

## Technical Details

- Uses **ContextVIPER** for context-based tree extraction
- Supports both axis-aligned and oblique decision boundaries
- Implements DAGGER (Dataset Aggregation) for iterative training
- Feature extraction based on object-centric representations
- Configurable attention mechanisms with multiple distance functions

## Troubleshooting

If you encounter issues:

1. **Missing tree files**: Ensure the directory contains `adt_*.joblib` files
2. **Environment errors**: Verify all dependencies are installed in the Python 3.8 environment  
3. **Memory issues**: Reduce batch size or number of episodes
4. **Game-specific errors**: Check that the game name matches exactly ("freeway" or "tennis")

For more detailed error information, use the `-verbose` flag when running the loader.
