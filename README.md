# High-Throughput Photocatalysis Project

This project aims to support the analysis of photocatalysis data through a high-throughput approach, generating datasets that can be analyzed using machine learning methods. The project’s workflow is inspired by the paper **"High-Throughput Photocatalysis for Generating Reliable Datasets Analyzed by Machine Learning"** and is organized into modules for data management, machine learning analysis, and visualization.

## Project Structure

The project is structured into the following main directories and files:

- **`Database_generator/`**: This directory handles data generation and machine learning processes.

  - **`Data/`**: Stores the experimental data generated through high-throughput photocatalysis experiments.
  - **`experiment.ipynb`**: The main Jupyter notebook for running machine learning analyses on the experimental data. This notebook is designed to preprocess data, train models, and evaluate performance based on the datasets generated.
- **`Draw_Heatmaps/`**: Contains resources for visual analysis of results through heatmaps.

  - **`draw_heatmaps.ipynb`**: A Jupyter notebook dedicated to creating heatmaps to visualize trends and insights from the machine learning analysis results.

## Getting Started

### Prerequisites

1. **Python 3.9/3.10** and necessary packages
2. Packages requirements are listed in a `requirements.txt` file

### Installation

1. Clone the repository.
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### 1. Data Generation and Machine Learning Analysis

- Open and run **`Database_generator/experiment.ipynb`** to process the high-throughput photocatalysis data stored in `Database_generator/Data/`. This notebook will load the data, preprocess it, and run machine learning analyses to derive insights from the photocatalysis experiments.

### 2. Visualizing Results with Heatmaps

- After running the machine learning analysis, open **`Draw_Heatmaps/draw_heatmaps.ipynb`**. This notebook will create heatmaps based on the generated data and model predictions, allowing for a visual representation of patterns, which aids in further analysis.

## Purpose and Goals

This project supports the generation of reliable photocatalysis datasets that can be effectively analyzed by machine learning algorithms. By integrating high-throughput experimental data with computational methods, this approach aims to facilitate discoveries in photocatalysis, streamline data analysis, and enhance the robustness of data-driven conclusions.
