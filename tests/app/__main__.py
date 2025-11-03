# app/__main__.py (formerly src/__main__.py)

import argparse
import sys
import os
from typing import Dict, Any, List

# --- Import core components and workflow from the 'app' package ---
# Note the change in import paths from 'src.core' to 'app.core'
from app.core.llm_provider import LLMProvider
from app.core.utils import load_data, save_json_output
from app.workflow.ops_graph import create_ops_workflow # Requires definition in workflow/ops_graph.py

# --- 1. Argument Parsing ---

# Define the CLI command structure
parser = argparse.ArgumentParser(
    description="Shopify Dropshipping Ops Agent: Multi-Agent Workflow CLI."
)
parser.add_argument(
    '--catalog',
    type=str,
    required=True,
    help="Path to the supplier_catalog.csv file."
)
parser.add_argument(
    '--orders',
    type=str,
    required=True,
    help="Path to the orders.csv file."
)
parser.add_argument(
    '--out',
    type=str,
    default="out/",
    help="Output directory where all artifacts (JSON/CSV/MD) will be saved."
)