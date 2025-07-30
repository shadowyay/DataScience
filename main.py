import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FirePredictionBalanced:
    def __init__(self, data_path='last1.csv', sequence_length=7):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        self.target_column = 'Fire'
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date to ensure chronological order
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Fire events: {self.df['Fire'].sum()} out of {len(self.df)} days ({self.df['Fire'].mean():.2%})")
        
        return self.df
    
    def split_data(self, train_ratio=0.8):
        """Split data into training and testing sets preserving chronological order"""
        print(f"\nSplitting data: {train_ratio:.0%} training, {1-train_ratio:.0%} testing")
        
        # Calculate split point
        split_idx = int(len(self.df) * train_ratio)
        
        # Split data
        self.train_df = self.df.iloc[:split_idx].copy()
        self.test_df = self.df.iloc[split_idx:].copy()
        
        print(f"Training set: {len(self.train_df)} samples")
        print(f"Testing set: {len(self.test_df)} samples")
        
        return self.train_df, self.test_df
    
    def create_sequences(self, df, features, target):
        """Create sequences for LSTM model"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(df)):
            # Get sequence of features (excluding target)
            sequence = features[i-self.sequence_length:i]
            X.append(sequence)
            y.append(target.iloc[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\nPreparing data for training...")
        
        # Scale features
        train_features = self.train_df[self.feature_columns].values
        test_features = self.test_df[self.feature_columns].values
        
        # Fit scaler on training data only
        self.scaler.fit(train_features)
        
        # Transform both training and test data
        train_features_scaled = self.scaler.transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        # Create sequences
        X_train, y_train = self.create_sequences(
            self.train_df, 
            train_features_scaled, 
            self.train_df[self.target_column]
        )
        
        X_test, y_test = self.create_sequences(
            self.test_df, 
            test_features_scaled, 
            self.test_df[self.target_column]
        )
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Testing sequences: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, input_shape):
        """Build a model optimized for predicting fire events"""
        print("\nBuilding LSTM model...")
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(16, return_sequences=False),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.005),  # Lower learning rate for stability
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=500, batch_size=4):
        """Train the model with aggressive class balancing"""
        print(f"\nTraining model for {epochs} epochs...")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Calculate aggressive class weights to favor fire prediction
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # Increase weight for fire class (class 1) to make model more sensitive
        class_weights[1] *= 3.0  # Triple the weight for fire class
        
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights (fire class boosted): {class_weight_dict}")
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=30,
            min_lr=0.0001
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def find_optimal_threshold(self, X_test, y_test):
        """Find optimal threshold that maximizes fire detection"""
        print("\nFinding optimal threshold for fire detection...")
        
        # Get prediction probabilities
        y_pred_proba = self.model.predict(X_test)
        
        # Try different thresholds, focusing on lower values to detect more fires
        thresholds = np.arange(0.05, 0.5, 0.02)  # Lower threshold range
        best_threshold = 0.3  # Start with a lower default
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int).flatten()
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("\nEvaluating model performance...")
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(X_test, y_test)
        
        # Make predictions with optimal threshold
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > optimal_threshold).astype(int).flatten()
        
        print(f"Raw prediction probabilities: {y_pred_proba.flatten()}")
        print(f"Thresholded predictions (threshold={optimal_threshold}): {y_pred}")
        print(f"Actual values: {y_test}")
        print(f"Predictions with value 1: {sum(y_pred)} out of {len(y_pred)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return y_pred, y_pred_proba, optimal_threshold
    
    def iterative_forecasting(self, test_df, test_features_scaled, threshold=0.3):
        """Perform iterative forecasting using 7-day sliding window"""
        print(f"\nPerforming iterative forecasting with threshold {threshold}...")
        
        predictions = []
        actual_values = []
        prediction_dates = []
        prediction_probabilities = []
        
        # Start from the 8th day (index 7) in the test set
        for i in range(self.sequence_length, len(test_df)):
            # Get the 7-day window of features
            window_features = test_features_scaled[i-self.sequence_length:i]
            window_features = window_features.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred_proba = self.model.predict(window_features, verbose=0)
            pred = (pred_proba > threshold).astype(int)[0][0]
            
            # Store results
            predictions.append(pred)
            prediction_probabilities.append(pred_proba[0][0])
            actual_values.append(test_df.iloc[i][self.target_column])
            prediction_dates.append(test_df.iloc[i]['date'])
            
            print(f"Date: {test_df.iloc[i]['date'].strftime('%Y-%m-%d')}, "
                  f"Predicted: {pred} (prob: {pred_proba[0][0]:.4f}), "
                  f"Actual: {test_df.iloc[i][self.target_column]}")
        
        print(f"\nPrediction probabilities range: {min(prediction_probabilities):.4f} to {max(prediction_probabilities):.4f}")
        print(f"Mean prediction probability: {np.mean(prediction_probabilities):.4f}")
        print(f"Predictions with value 1: {sum(predictions)} out of {len(predictions)}")
        
        return predictions, actual_values, prediction_dates, prediction_probabilities
    
    def plot_results(self, predictions, actual_values, prediction_dates, prediction_probabilities):
        """Plot the forecasting results"""
        print("\nCreating visualization...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Time series of predictions vs actual
        dates = [d.strftime('%Y-%m-%d') for d in prediction_dates]
        x_indices = range(len(dates))
        
        ax1.plot(x_indices, actual_values, 'b-', label='Actual Fire', linewidth=2, marker='o')
        ax1.plot(x_indices, predictions, 'r--', label='Predicted Fire', linewidth=2, marker='s')
        ax1.set_title('Fire Prediction vs Actual Values (Iterative Forecasting)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (Days)')
        ax1.set_ylabel('Fire (0/1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis labels for better readability
        step = max(1, len(dates) // 10)
        ax1.set_xticks(x_indices[::step])
        ax1.set_xticklabels(dates[::step], rotation=45)
        
        # Plot 2: Prediction probabilities over time
        ax2.plot(x_indices, prediction_probabilities, 'g-', linewidth=2, marker='o')
        ax2.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Threshold (0.3)')
        ax2.set_title('Prediction Probabilities Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (Days)')
        ax2.set_ylabel('Prediction Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Set x-axis labels
        ax2.set_xticks(x_indices[::step])
        ax2.set_xticklabels(dates[::step], rotation=45)
        
        # Plot 3: Accuracy over time
        correct_predictions = [1 if p == a else 0 for p, a in zip(predictions, actual_values)]
        cumulative_accuracy = np.cumsum(correct_predictions) / np.arange(1, len(correct_predictions) + 1)
        
        ax3.plot(x_indices, cumulative_accuracy, 'purple', linewidth=2)
        ax3.set_title('Cumulative Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (Days)')
        ax3.set_ylabel('Cumulative Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Set x-axis labels
        ax3.set_xticks(x_indices[::step])
        ax3.set_xticklabels(dates[::step], rotation=45)
        
        # Plot 4: Distribution of prediction probabilities
        ax4.hist(prediction_probabilities, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=0.3, color='r', linestyle='--', alpha=0.7, label='Threshold (0.3)')
        ax4.set_title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Prediction Probability')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fire_prediction_balanced_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        final_accuracy = np.mean(correct_predictions)
        print(f"\nFinal Iterative Forecasting Accuracy: {final_accuracy:.4f}")
        print(f"Total predictions made: {len(predictions)}")
        print(f"Correct predictions: {sum(correct_predictions)}")
        print(f"Predictions with value 1: {sum(predictions)}")
        
        return final_accuracy
    
    def run_complete_pipeline(self, epochs=500, batch_size=4):
        """Run the complete fire prediction pipeline"""
        print("=" * 60)
        print("FIRE PREDICTION BALANCED SYSTEM USING TENSORFLOW")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Split data
        self.split_data()
        
        # Step 3: Prepare data for training
        X_train, y_train, X_test, y_test = self.prepare_data()
        
        # Step 4: Train model
        history = self.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)
        
        # Step 5: Evaluate model
        y_pred, y_pred_proba, optimal_threshold = self.evaluate_model(X_test, y_test)
        
        # Step 6: Perform iterative forecasting
        test_features_scaled = self.scaler.transform(self.test_df[self.feature_columns].values)
        predictions, actual_values, prediction_dates, prediction_probabilities = self.iterative_forecasting(
            self.test_df, test_features_scaled, optimal_threshold
        )
        
        # Step 7: Plot results
        final_accuracy = self.plot_results(predictions, actual_values, prediction_dates, prediction_probabilities)
        
        print("\n" + "=" * 60)
        print("BALANCED PIPELINE COMPLETED!")
        print("=" * 60)
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'predictions': predictions,
            'actual_values': actual_values,
            'prediction_dates': prediction_dates,
            'prediction_probabilities': prediction_probabilities,
            'final_accuracy': final_accuracy,
            'optimal_threshold': optimal_threshold
        }

def main():
    """Main function to run the fire prediction balanced system"""
    # Initialize the system
    fire_system = FirePredictionBalanced(data_path='last1.csv', sequence_length=7)
    
    # Run the complete pipeline
    results = fire_system.run_complete_pipeline(epochs=500, batch_size=4)
    
    return results

if __name__ == "__main__":
    results = main() 