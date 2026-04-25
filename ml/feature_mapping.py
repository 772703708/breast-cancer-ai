"""
Feature mapping for Breast Cancer Prediction Model
This file contains the complete feature list used during training
You need to update this based on your actual training data columns
"""

# Complete feature list (55 features from your training)
FEATURE_COLUMNS = [
    # Numerical features
    'Age',
    'BMI',
    'TumorSize',
    'InvNodes',
    
    # Menopause (one-hot encoded)
    'Menopause_premeno',
    'Menopause_perimeno',
    'Menopause_postmeno',
    
    # Binary features
    'Metastasis',
    'History',
    'BreastSide_Right',
    
    # Breast Quadrant (one-hot encoded)
    'Quadrant_left_up',
    'Quadrant_left_low',
    'Quadrant_right_up',
    'Quadrant_right_low',
    'Quadrant_central',
    
    # Additional features (from your training data)
    # Add all other features here (41 more features)
    # These are placeholders - update with actual feature names
    'Feature_16', 'Feature_17', 'Feature_18', 'Feature_19', 'Feature_20',
    'Feature_21', 'Feature_22', 'Feature_23', 'Feature_24', 'Feature_25',
    'Feature_26', 'Feature_27', 'Feature_28', 'Feature_29', 'Feature_30',
    'Feature_31', 'Feature_32', 'Feature_33', 'Feature_34', 'Feature_35',
    'Feature_36', 'Feature_37', 'Feature_38', 'Feature_39', 'Feature_40',
    'Feature_41', 'Feature_42', 'Feature_43', 'Feature_44', 'Feature_45',
    'Feature_46', 'Feature_47', 'Feature_48', 'Feature_49', 'Feature_50',
    'Feature_51', 'Feature_52', 'Feature_53', 'Feature_54', 'Feature_55'
]

def get_feature_index(feature_name):
    """Get the index of a feature by its name"""
    try:
        return FEATURE_COLUMNS.index(feature_name)
    except ValueError:
        return -1