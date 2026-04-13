What it is: A Titanic survival predictor using Random Forest.

How to run it: 1. source venv/bin/activate
2. python3 train_model.py
3. streamlit run app.py

I used RANDOM FOREST because it handles "feature importance" well. You can actually see which of yournew features mattered most by adding these two lines at the end of your script.

The custom engineering i incorporated here were "IsChild and ""Family Size" as a measn to create ever more accurate predictions. 
