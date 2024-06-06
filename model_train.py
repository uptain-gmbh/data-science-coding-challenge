import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
import sys

def main(data_path):
    df = pd.read_csv(data_path)
    le = LabelEncoder()
    df['age'] = le.fit_transform(df['age'])
    df['domain'] = df['domain'].astype('category')
    
    X = df.drop(columns=['email', 'username', 'age'])
    y = df['age']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def objective(trial):
        param = {
            'objective': 'multiclass',
            'num_class': len(le.classes_),
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'class_weight': 'balanced'
        }
        
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best parameters: ", best_params)

    # Train final model with best parameters
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)
    
    # Evaluate final model
    print("Final Model Classification Report:")
    print(classification_report(y_test, final_preds))

    # Save the best model and LabelEncoder
    model_filename = 'best_lightgbm_classification_model.pkl'
    joblib.dump(final_model, model_filename)
    joblib.dump(le, 'label_encoder.pkl')
    print(f"Model and LabelEncoder saved to {model_filename} and label_encoder.pkl")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_path = 'df_cleaned.csv'
    elif len(sys.argv) != 2:
        print("Usage: python3 model_train.py <data.csv>")
        sys.exit(1)
    else:
        data_path = sys.argv[1]
    main(data_path)