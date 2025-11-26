# Glycemic Load Data/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import math
import os # Import the os module

app = Flask(__name__)

# Get the directory of the current script
# basedir = os.path.abspath(os.path.dirname(__file__)) # __file__ is not defined in Jupyter notebooks
# Assuming the CSV is in the current working directory or a known relative path for notebook testing
csv_file_name = "GL GI Prediction Dataset.csv"
# Construct the path. For local testing in Jupyter, current directory is often sufficient.
# If running as a script (app.py), the original basedir method is correct.
# For demonstration in notebook, let's assume CSV is in the current directory or the same as the notebook.
# We'll just use the file name directly, assuming it's in the current working directory.
# If you need to specify a directory relative to the notebook, adjust this.
csv_path = csv_file_name # Simple path for notebook execution

# Load the data into a Pandas DataFrame when the application starts
# Use the determined path to the CSV file
try:
    # Try reading with latin-1 first
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        print("CSV data loaded successfully with 'latin-1' encoding!")
    except Exception as e:
        print(f"Error with 'latin-1' encoding, trying 'cp1252': {e}")
        # If latin-1 fails, try cp1252
        df = pd.read_csv(csv_path, encoding='cp1252')
        print("CSV data loaded successfully with 'cp1252' encoding!")
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("\nColumn names and data types:")
    print(df.info())
    print("\nColumns in the loaded DataFrame:")
    print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found at {csv_path}. Please make sure it is in the same directory as app.py.")
    df = pd.DataFrame() # Create an empty DataFrame to avoid errors later
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    df = pd.DataFrame() # Create an empty DataFrame to avoid errors later


# --- Define the calculation functions ---
# You can copy these directly from your working notebook code

def calculate_test_meal_glycemic_load(test_meal_df):
    """
    Calculates the estimated Glycemic Load (GL) for a test meal.

    GL = (GI * Carbohydrate amount per serving) / 100
    Carbohydrate amount per serving is calculated based on 'Available Carbohydrates per 100g (grams)'
    and 'Quantity Used in gram'.

    Args:
        test_meal_df (pd.DataFrame): DataFrame containing food items in the test meal,
                                      including 'Glycemic Index',
                                      'Available Carbohydrate in gram per 100 gram',
                                      and 'Quantity Used in gram'.

    Returns:
        float: The total Glycemic Load for the test meal, or None if input is invalid.
    """
    if test_meal_df is None or test_meal_df.empty:
        print("Warning: Input DataFrame for GL calculation is empty or None.")
        return None

    # Ensure required columns exist
    # Updated column names to match the loaded DataFrame
    gi_col = 'Glycemic Index'
    carb_col = 'Available Carbohydrate in gram per 100 gram'
    qty_col = 'Quantity Used in gram'
    required_cols = [gi_col, carb_col, qty_col]

    # Check for required columns case-insensitively if needed, but stick to exact match for now
    if not all(col in test_meal_df.columns for col in required_cols):
        # This error should ideally be caught before this function is called if merge is done correctly
        print(f"Error: Missing one or more required columns for GL calculation: {required_cols}")
        print(f"Available columns in input DataFrame: {test_meal_df.columns.tolist()}")
        return None

    # Convert relevant columns to numeric, coercing errors to NaN
    # This is crucial for calculations and prevents TypeErrors if data isn't clean
    for col in [gi_col, carb_col, qty_col]:
         if col in test_meal_df.columns:
            test_meal_df[col] = pd.to_numeric(test_meal_df[col], errors='coerce')

    # Drop rows where essential columns are NaN after coercion
    test_meal_df.dropna(subset=[gi_col, carb_col, qty_col], inplace=True)


    # Calculate carbohydrate amount per serving for each item
    test_meal_df['Carbohydrate Amount (serving)'] = (test_meal_df[carb_col] / 100) * test_meal_df[qty_col]

    # Handle potential division by zero if qty_col is zero for all rows (unlikely but good practice)
    test_meal_df.loc[test_meal_df[qty_col] == 0, 'Carbohydrate Amount (serving)'] = 0


    # Calculate GL for each item and sum them up
    test_meal_df['Item GL'] = (test_meal_df[gi_col] * test_meal_df['Carbohydrate Amount (serving)']) / 100

    total_glycemic_load = test_meal_df['Item GL'].sum()

    # Handle potential NaN or infinite values after sum
    if pd.isna(total_glycemic_load) or math.isinf(total_glycemic_load):
         print("Warning: GL calculation resulted in NaN or infinite value. Check input data.")
         return None

    return total_glycemic_load

def calculate_test_meal_glycemic_index(test_meal_df):
    """
    Calculates the estimated Glycemic Index (GI) for a test meal using a weighted average
    based on the carbohydrate content of each item.

    GI_meal = Sum(GI_item * Carbohydrate_item) / Sum(Carbohydrate_item)

    Args:
        test_meal_df (pd.DataFrame): DataFrame containing food items in the test meal,
                                      including 'Glycemic Index',
                                      'Available Carbohydrate in gram per 100 gram',
                                      and 'Quantity Used in gram'.

    Returns:
        float: The estimated Glycemic Index for the test meal, or None if input is invalid
               or total carbohydrates are zero.
    """
    if test_meal_df is None or test_meal_df.empty:
        print("Warning: Input DataFrame for GI calculation is empty or None.")
        return None

    # Ensure required columns exist
    # Updated column names to match the loaded DataFrame
    gi_col = 'Glycemic Index'
    carb_col = 'Available Carbohydrate in gram per 100 gram'
    qty_col = 'Quantity Used in gram'
    required_cols = [gi_col, carb_col, qty_col]

    # Check for required columns case-insensitively if needed, but stick to exact match for now
    if not all(col in test_meal_df.columns for col in required_cols):
         # This error should ideally be caught before this function is called if merge is done correctly
        print(f"Error: Missing one or more required columns for GI calculation: {required_cols}")
        print(f"Available columns in input DataFrame: {test_meal_df.columns.tolist()}")
        return None

     # Convert relevant columns to numeric, coercing errors to NaN
     # This is crucial for calculations and prevents TypeErrors if data isn't clean
    for col in [gi_col, carb_col, qty_col]:
         if col in test_meal_df.columns:
            test_meal_df[col] = pd.to_numeric(test_meal_df[col], errors='coerce')

    # Drop rows where essential columns are NaN after coercion
    test_meal_df.dropna(subset=[gi_col, carb_col, qty_col], inplace=True)


    # Calculate carbohydrate amount per serving for each item
    test_meal_df['Carbohydrate Amount (serving)'] = (test_meal_df[carb_col] / 100) * test_meal_df[qty_col]

     # Handle potential division by zero if qty_col is zero for all rows (unlikely but good practice)
    test_meal_df.loc[test_meal_df[qty_col] == 0, 'Carbohydrate Amount (serving)'] = 0


    # Calculate (GI * Carbohydrate) for each item
    test_meal_df['GI * Carbohydrate'] = test_meal_df[gi_col] * test_meal_df['Carbohydrate Amount (serving)']

    # Sum the weighted GI and total carbohydrates
    sum_gi_times_carb = test_meal_df['GI * Carbohydrate'].sum()
    total_carbohydrates = test_meal_df['Carbohydrate Amount (serving)'].sum()

     # Handle potential NaN or infinite values before division
    if pd.isna(sum_gi_times_carb) or math.isinf(sum_gi_times_carb) or pd.isna(total_carbohydrates) or math.isinf(total_carbohydrates):
         print("Warning: GI calculation resulted in NaN or infinite values in intermediate steps. Check input data.")
         return None


    # Calculate weighted average GI, avoid division by zero
    if total_carbohydrates > 0:
        weighted_gi = sum_gi_times_carb / total_carbohydrates
         # Handle potential NaN or infinite values after division
        if pd.isna(weighted_gi) or math.isinf(weighted_gi):
            print("Warning: GI calculation resulted in NaN or infinite value. Check input data or total carbohydrates.")
            return None
        return weighted_gi
    else:
        print("Warning: Total carbohydrates in the test meal are zero. Cannot calculate weighted GI.")
        return None


# --- Flask Routes ---

@app.route('/')
def index():
    # Get unique food items for the dropdown, ensuring df is not empty
    food_items = df['Food Item'].unique().tolist() if not df.empty and 'Food Item' in df.columns else []
    # Sort food items alphabetically for better usability
    food_items.sort()
    return render_template('index.html', food_items=food_items)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json # Get JSON data from the frontend

    if not data or 'meal_items' not in data:
        return jsonify({'error': 'Invalid input: Missing meal_items data.'}), 400

    meal_items_list = data.get('meal_items', []) # Get the list, default to empty if not found

    if not meal_items_list:
        return jsonify({'message': 'Meal list is empty. Add items before calculating.'})

    # Convert the list of dictionaries to a DataFrame
    try:
        meal_df_input = pd.DataFrame(meal_items_list)
    except Exception as e:
         return jsonify({'error': f'Invalid meal items data format: {e}'}), 400

    # Ensure the main DataFrame is loaded and has the necessary columns before merging
    if df.empty or not all(col in df.columns for col in ['Food Item', 'Glycemic Index', 'Available Carbohydrate in gram per 100 gram']):
         return jsonify({'error': 'Application data (CSV) not loaded correctly.'}), 500


    # Merge with the main DataFrame to get GI and Carb data
    # Use 'left' merge to keep all items from the input list
    meal_df_merged = pd.merge(meal_df_input, df[['Food Item', 'Glycemic Index', 'Available Carbohydrate in gram per 100 gram']], on='Food Item', how='left')

    # Check if any items from the input list were not found in the main data
    if meal_df_merged['Glycemic Index'].isnull().any() or meal_df_merged['Available Carbohydrate in gram per 100 gram'].isnull().any():
         missing_items = meal_df_merged[meal_df_merged['Glycemic Index'].isnull()]['Food Item'].tolist()
         return jsonify({'error': f'Could not find data for the following items: {", ".join(missing_items)}. Please check the item names.'}), 400

    # Perform calculations
    # Pass the merged DataFrame (which now includes GI and Carb data) to the calculation functions
    glycemic_load = calculate_test_meal_glycemic_load(meal_df_merged.copy()) # Pass a copy to avoid modifying the merged_df in place within functions
    glycemic_index = calculate_test_meal_glycemic_index(meal_df_merged.copy()) # Pass a copy


    results = {}
    if glycemic_load is not None:
        results['glycemic_load'] = f"{glycemic_load:.2f}"
    else:
        results['glycemic_load_error'] = "Could not calculate Glycemic Load. Check data or input values."

    if glycemic_index is not None:
        # Classify GI (reuse your logic)
        if glycemic_index <= 55:
            gi_category = "Low"
        elif 56 <= glycemic_index <= 69:
            gi_category = "Medium"
        else:
            gi_category = "High"
        results['glycemic_index'] = f"{glycemic_index:.2f}"
        results['gi_category'] = gi_category
    else:
        results['glycemic_index_error'] = "Could not calculate Glycemic Index. Check data or input values."

    # If there were issues in calculation functions but no merge errors, add a general warning
    if not results.get('glycemic_load') and not results.get('glycemic_load_error'):
         results['glycemic_load_error'] = "GL calculation returned None. Check data."
    if not results.get('glycemic_index') and not results.get('glycemic_index_error'):
         results['glycemic_index_error'] = "GI calculation returned None. Check data."


    return jsonify(results)


# Add a main block to run the app locally for testing
if __name__ == '__main__':
    # Make sure to put your CSV file in the same folder as app.py for this to work
    # For Render deployment, gunicorn will handle starting the app, this block is for local testing
    print("Running Flask app locally...")
    # Set use_reloader=False to prevent restarts in interactive environments like notebooks
    app.run(debug=True, use_reloader=False)