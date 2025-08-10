# Expense Splitter
Minimal app to keep track of group expenses and to suggest minimal settlements.

# Installation
1. Clone this repository in your computer.
2. Create a python virtual enviroment.
    ```
    python -m venv .venv
    ```

3. Activate .venv
    - On Windows, run `.venv\Scripts\activate`
    - Otherwise `source .venv/bin/activate`

4. Run `pip install -r requirements.txt`

# How to use
In your terminal, run: `streamlit run group_splitter_app.py`

A browser tab will open where you can create a group, add members, add expenses and check the balances.

The information will be saved in a local sql db file.