# CAID

CAID is a modular deep learning framework designed for incremental learning and research in computer vision. It provides flexible components for data processing, model definition, training, and evaluation, making it easy to experiment with new ideas and configurations.

## Features

- Modular architecture for easy extension and customization
- Support for incremental learning scenarios
- Configurable training pipelines
- Utilities for data augmentation, validation, and checkpointing

Support methods

Sequense Finetune


Conditioned Prompt-Optimization for Continual Deepfake Detection

Exemplar-Free Incremental Deepfake Detection



Datasets:
CDDB

## Project Structure

```
CAID/
├── cfgs/                # Configuration files (YAML)
├── data/                # Data processing and augmentation scripts
├── docs/                # Documentation (if any)
├── engine/              # Training and evaluation engine
├── networks/            # Model/network definitions
├── utils/               # Utility functions and helpers
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── LICENSE              # License file
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CAID.git
   cd CAID
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All experiment and model settings are managed via YAML files in the `cfgs/` directory. You can modify or create new configuration files to suit your experiments.

Example: `cfgs/incremental_rine.yaml`

## Usage

### Training

To start training with a specific configuration file:

```bash
python train.py --config cfgs/incremental_rine.yaml
```

You can customize the configuration file or add new ones for different experiments.

### Data Preparation

Ensure your datasets are prepared as expected by the scripts in the `data/` directory. Refer to the code and comments for dataset format and augmentation options.

## Extending the Framework

- **Add new models:** Implement your model in `networks/` and register it in the factory.
- **Custom data processing:** Add or modify scripts in `data/`.
- **Training logic:** Extend or modify the base trainer in `engine/`.

## License

This project is licensed under the terms of the LICENSE file provided in the repository.

## Acknowledgements

If you use this framework in your research, please consider citing or acknowledging the project.
