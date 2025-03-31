# AI Makerspace Midterm: RAG Pipeline with RAGAS Evaluation

A RAG (Retrieval-Augmented Generation) pipeline with an integrated RAGAS evaluation implemented as part of the AI Makerspace Midterm assessment.

## Overview

This project implements a RAG pipeline designed to augment generative models with a retrieval mechanism. The pipeline is evaluated using a custom RAGAS evaluation method. It combines Jupyter Notebook-based exploration with a Python application, providing a versatile environment for both development and demonstration purposes.

## Features

- **Retrieval-Augmented Generation (RAG) Pipeline:** Enhances generative capabilities by integrating document retrieval.
- **RAGAS Evaluation:** Evaluates the performance of the RAG pipeline using custom metrics.
- **Docker Support:** Easily containerize and run the project with Docker.
- **Chainlit Integration:** Includes configuration for Chainlit to support interactive workflows.
- **Modular Structure:** Organized code with notebooks, scripts, and necessary data files.

## Project Structure
```
├── .chainlit/ # Chainlit configuration and assets
├── cache/ # Caching layer for intermediate data/results
├── data/ # Data files used by the pipeline
├── ai_makerspace_midterm_rag.ipynb # Jupyter Notebook with pipeline code and evaluation
├── app.py # Main Python application entry point
├── chainlit.md # Documentation/configuration for Chainlit integration
├── Dockerfile # Docker configuration for containerized deployment
├── requirements.txt # Python dependencies
├── .gitignore # Files and directories to ignore in Git
└── LICENSE # MIT License details
```

## Setup and Installation

### Prerequisites

- **Python 3.8+** (or your preferred Python version)
- **Docker** (optional, for containerized execution)

### Installing Dependencies

1. Clone the repository:
```bash
git clone https://github.com/michael-sd/ai_makerspace_midterm.git
cd ai_makerspace_midterm
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running with Docker
If you prefer to run the project in a container:

Build the Docker image:

```bash
docker build -t ai_makerspace_midterm .
```

Run the Docker container:
```bash
docker run -p 5000:5000 ai_makerspace_midterm
```

## Usage
Running the Pipeline
- Jupyter Notebook: Open ai_makerspace_midterm_rag.ipynb to explore and run the pipeline interactively.
- Python Application:
```bash
Run the main application with:
```
