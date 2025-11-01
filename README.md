# 🛒 Shopify Dropshipping Operations Agent

This repository contains the solution for building a multi-agent hierarchical system to simulate and automate key Shopify dropshipping operations, adhering strictly to the **$0 cost** constraint using local Large Language Models (LLMs).

---

## 🚀 Project Goal

The primary goal of this project is to create an autonomous, multi-agent system capable of simulating a complete dropshipping workflow, including:
1. Product selection from a supplier catalog.
2. Generation of Shopify listing content (titles, descriptions, SEO).
3. Calculating and proposing deterministic pricing and stock synchronization.
4. Simulating order routing.
5. Generating a daily operations report.

---

## 🛠️ Environment Setup ($0 Cost Local LLM)

This project requires a Python environment and **Ollama** to run the local LLMs.

### 1. Prerequisites

* **Python:** Version **3.8+** (recommended)
* **Ollama:** Must be installed and running on your system. It supports macOS, Windows, and Linux.
    * **Tip:** Ensure you have enough RAM (8GB minimum for 7B models, 16GB+ recommended) for smooth operation.

### 2. Ollama Installation and Local LLMs

Ollama is used to host the open-source models **Llama 3** and **Mistral** to satisfy the **Multi-LLM Setup** constraint.

1.  **Install Ollama:**
    * Download and install the application from the official Ollama website.
    * *Alternatively, for Linux, run:*
        ```bash
        curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
        ```

2.  **Download Required Models:** Once Ollama is installed, use your terminal to pull the models. This process ensures the models are locally available for your agents.

    ```bash
    # Download Meta Llama 3 (8B Instruct)
    ollama run llama3

    # Download Mistral (7B Instruct)
    ollama run mistral
    ```
    (Note: You can type `/bye` to exit the interactive chat session after the download is complete.)

### 3. Python Project Setup

It is highly recommended to use a virtual environment to manage dependencies.

1.  **Create and Activate Virtual Environment:**

    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate the environment
    # macOS/Linux:
    source .venv/bin/activate
    # Windows (PowerShell):
    # .venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:** Install the necessary libraries.

    ```bash
    pip install -r requirements.txt
    ```
---

## 🚀 Running the Project

The agent system is executed via a Command Line Interface (CLI) tool.

1.  **Ensure Data Files are Present:** Place the `supplier_catalog.csv` and `orders.csv` files inside a directory named `data/` in your project root.
2.  **Execute the Agent:** Run the main application script with the required file paths.

```bash
python -m app run --catalog data/supplier_catalog.csv \
--orders data/orders.csv --out out/


