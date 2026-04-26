import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# DEFENSIVE PATHING: Add the current directory to sys.path
# This allows the "from preprocessing" import to work even when run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# The import you requested
from preprocessing import run_full_preprocessing

def train_titanic_model(data_path='data/train.csv'):
    """
    Loads, Preps, Trains, and Evaluates the Titanic model.
    """
    # Ensure we find the data regardless of where the script is called from
    # Logic: if data/train.csv isn't here, look one level up (if running from src)
    if not os.path.exists(data_path):
        data_path = os.path.join('..', data_path)
