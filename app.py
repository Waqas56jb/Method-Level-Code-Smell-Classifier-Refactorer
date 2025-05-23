import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from joblib import dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
sql_file_path = "refactoring50.sql"
table_name = "no"
model_output = "best_model.pth"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
max_value = 1_000_000  # Cap for numerical values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_rows = 500  # Train on 500 samples

# Step 1: Data Extraction from SQL
def extract_columns(sql_file_path, table_name):
    columns = []
    create_start = False
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped = line.strip()
                if stripped.startswith(f"CREATE TABLE") and f"`{table_name}`" in stripped:
                    create_start = True
                    continue
                if create_start:
                    if stripped.startswith(")"):
                        break
                    if stripped.startswith("`"):
                        col_name = stripped.split('`')[1]
                        columns.append(col_name)
    except FileNotFoundError:
        logging.error(f"SQL file not found: {sql_file_path}")
        exit(1)
    return columns

def parse_insert_values(insert_line):
    pattern = re.compile(r'\((.*?)\)(?:,|\);)', re.DOTALL)
    all_values = pattern.findall(insert_line)
    rows = []
    for val_str in all_values:
        values = []
        current = ""
        in_quote = False
        i = 0
        while i < len(val_str):
            ch = val_str[i]
            if ch == "'" and (i == 0 or val_str[i-1] != '\\'):
                in_quote = not in_quote
                current += ch
            elif ch == ',' and not in_quote:
                cleaned = current.strip()
                if cleaned == 'NULL':
                    values.append(None)
                else:
                    if cleaned.startswith("'") and cleaned.endswith("'"):
                        cleaned = cleaned[1:-1].replace("\\'", "'")
                    values.append(cleaned)
                current = ""
            else:
                current += ch
            i += 1
        cleaned = current.strip()
        if cleaned == 'NULL':
            values.append(None)
        else:
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1].replace("\\'", "'")
            values.append(cleaned)
        rows.append(values)
    return rows

# Extract columns
columns = extract_columns(sql_file_path, table_name)
if not columns:
    logging.error(f"No columns found for table `{table_name}` in {sql_file_path}.")
    exit(1)
logging.info(f"Columns found for table `{table_name}` ({len(columns)} columns): {columns}")

# Extract up to 500 rows
insert_rows = []
current_insert = ""
row_count = 0
try:
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith(f"INSERT INTO `{table_name}`"):
                if current_insert:
                    rows = parse_insert_values(current_insert)
                    insert_rows.extend(rows)
                    row_count += len(rows)
                    if row_count >= max_rows:
                        insert_rows = insert_rows[:max_rows]
                        break
                current_insert = line
            elif current_insert and line.endswith(';'):
                current_insert += ' ' + line
                rows = parse_insert_values(current_insert)
                insert_rows.extend(rows)
                row_count += len(rows)
                if row_count >= max_rows:
                    insert_rows = insert_rows[:max_rows]
                    break
            elif current_insert:
                current_insert += ' ' + line
        if current_insert and row_count < max_rows:
            rows = parse_insert_values(current_insert)
            insert_rows.extend(rows[:max_rows - row_count])
except FileNotFoundError:
    logging.error(f"SQL file not found: {sql_file_path}")
    exit(1)

if not insert_rows:
    logging.error(f"No INSERT INTO rows found for table `{table_name}`.")
    exit(1)
logging.info(f"Extracted {len(insert_rows)} rows for training.")

# Create DataFrame
try:
    df = pd.DataFrame(insert_rows, columns=columns)
except ValueError as e:
    logging.error(f"Error creating DataFrame: {e}")
    logging.error(f"Expected {len(columns)} columns, but found rows with value counts: {[len(row) for row in insert_rows[:5]]}")
    exit(1)

# Step 2: Data Preprocessing
# Select numerical features (exactly 65 features)
numerical_features = [
    'classAnonymousClassesQty', 'classAssignmentsQty', 'classCbo', 'classComparisonsQty',
    'classLambdasQty', 'classLcom', 'classLoc', 'classLoopQty', 'classMathOperationsQty',
    'classMaxNestedBlocks', 'classNosi', 'classNumberOfAbstractMethods',
    'classNumberOfDefaultFields', 'classNumberOfDefaultMethods', 'classNumberOfFields',
    'classNumberOfFinalFields', 'classNumberOfFinalMethods', 'classNumberOfMethods',
    'classNumberOfPrivateFields', 'classNumberOfPrivateMethods', 'classNumberOfProtectedFields',
    'classNumberOfProtectedMethods', 'classNumberOfPublicFields', 'classNumberOfPublicMethods',
    'classNumberOfStaticFields', 'classNumberOfStaticMethods', 'classNumberOfSynchronizedFields',
    'classNumberOfSynchronizedMethods', 'classNumbersQty', 'classParenthesizedExpsQty',
    'classReturnQty', 'classRfc', 'classStringLiteralsQty', 'classSubClassesQty',
    'classTryCatchQty', 'classUniqueWordsQty', 'classVariablesQty', 'classWmc',
    'methodAnonymousClassesQty', 'methodAssignmentsQty', 'methodCbo', 'methodComparisonsQty',
    'methodLambdasQty', 'methodLoc', 'methodLoopQty', 'methodMathOperationsQty',
    'methodMaxNestedBlocks', 'methodNumbersQty', 'methodParametersQty',
    'methodParenthesizedExpsQty', 'methodReturnQty', 'methodRfc', 'methodStringLiteralsQty',
    'methodSubClassesQty', 'methodTryCatchQty', 'methodUniqueWordsQty', 'methodVariablesQty',
    'methodWmc', 'bugFixCount', 'linesAdded', 'linesDeleted',
    'qtyMajorAuthors', 'qtyMinorAuthors', 'qtyOfAuthors', 'qtyOfCommits'
]
target = 'type'

# Verify feature count and presence
if len(numerical_features) != 65:
    logging.error(f"Expected 65 numerical features, got {len(numerical_features)}")
    exit(1)
missing_features = [f for f in numerical_features if f not in columns]
if missing_features:
    logging.error(f"Features not found in dataset: {missing_features}")
    exit(1)
logging.info(f"Selected {len(numerical_features)} numerical features.")

# Preprocess numerical features
try:
    X = df[numerical_features].astype(float).clip(upper=max_value).fillna(0).astype(np.float32)
except ValueError as e:
    logging.error(f"Error converting numerical features: {e}")
    exit(1)

# Log max values
max_values = X.max()
logging.info(f"Max values in numerical features:\n{max_values[max_values > 100_000]}")

# Preprocess target labels
y = df[target].astype(str)
logging.info(f"Unique values in `type` column: {y.unique()}")

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
logging.info(f"Number of classes: {num_classes}")
logging.info(f"Encoded labels: {np.unique(y_encoded)}")

# Proceed with actual num_classes (4)
smote = SMOTE(random_state=42, k_neighbors=1)  # Reduced k_neighbors due to small class size
try:
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
except ValueError as e:
    logging.error(f"SMOTE failed: {e}. Proceeding without oversampling.")
    X_resampled, y_resampled = X, y_encoded

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_scaled = pd.DataFrame(X_scaled, columns=numerical_features)  # Ensure feature names
dump(scaler, os.path.join(results_dir, 'scaler.joblib'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Convert to PyTorch tensors (Fixed: Convert DataFrame to NumPy array first)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # [samples, 1, 65]
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

logging.info(f"Training data shape: {X_train_tensor.shape}, Labels shape: {y_train_tensor.shape}")
logging.info(f"Test data shape: {X_test_tensor.shape}, Labels shape: {y_test_tensor.shape}")

# Step 3: Model Definitions
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out.squeeze(1))  # Remove seq_len dim, [batch_size, hidden_size]
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out.squeeze(1))
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out.squeeze(1))
        return out

class BiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out.squeeze(1))
        return out

class DenseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out = x.squeeze(1)  # [batch_size, features]
        out = self.fc(out)
        return out

# Step 4: Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    history = {'loss': [], 'accuracy': []}
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        history['loss'].append(epoch_loss / len(train_loader))
        history['accuracy'].append(correct / total)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {history['loss'][-1]:.4f}, Accuracy: {history['accuracy'][-1]:.4f}")
    return history

# Step 5: Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logging.info(f"Evaluating batch data shape: {data.shape}")
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

# Step 6: Model Training and Evaluation
model_types = ['RNN', 'LSTM', 'GRU', 'Bi-RNN', 'Dense']
histories = {}
metrics = {}
models = {}  # Store trained models and their test loaders for combined accuracy and confusion matrices
best_model = None
best_accuracy = 0
best_model_type = ''

# Define base test dataset for consistency
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
base_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Collect ground truth labels once (to avoid overwriting in the loop)
y_true_all = []
for _, target in base_test_loader:
    y_true_all.extend(target.numpy())
logging.info(f"Total test samples: {len(y_true_all)}")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for model_type in model_types:
    logging.info(f"Training {model_type} model...")
    if model_type == 'Dense':
        model = DenseModel(input_size=X_scaled.shape[1], output_size=num_classes).to(device)
        train_data = X_train_tensor.squeeze(1)
        test_data = X_test_tensor.squeeze(1)
        train_dataset = TensorDataset(train_data, y_train_tensor)
        test_dataset = TensorDataset(test_data, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    else:
        model = {
            'RNN': RNNModel,
            'LSTM': LSTMModel,
            'GRU': GRUModel,
            'Bi-RNN': BiRNNModel
        }[model_type](input_size=X_scaled.shape[1], hidden_size=32, output_size=num_classes).to(device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Class-weighted loss
    class_counts = np.bincount(y_resampled)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        history = train_model(model, train_loader, criterion, optimizer, num_epochs=5)
        histories[model_type] = history
    except Exception as e:
        logging.error(f"Error training {model_type}: {e}")
        continue

    y_true, y_pred = evaluate_model(model, test_loader)
    metrics[model_type] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Generate and save confusion matrix for this model
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))  # Use actual num_classes (4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix ({model_type})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_type.lower()}.png'))
    plt.close()
    logging.info(f"Confusion matrix for {model_type} saved to {results_dir}/confusion_matrix_{model_type.lower()}.png")

    # Cross-validation with 3 folds
    try:
        cv_scores = cross_val_score(RandomForestClassifier(random_state=42), X_scaled, y_resampled, cv=3, scoring='accuracy')
        metrics[model_type]['cv_accuracy'] = cv_scores.mean()
    except ValueError as e:
        logging.warning(f"Cross-validation failed: {e}. Setting cv_accuracy to 0.")
        metrics[model_type]['cv_accuracy'] = 0

    logging.info(f"{model_type} Test Accuracy: {metrics[model_type]['accuracy']:.4f}")
    logging.info(f"{model_type} Cross-Validation Accuracy: {metrics[model_type]['cv_accuracy']:.4f}")

    models[model_type] = (model, test_loader)  # Store model and its test loader for combined accuracy

    if metrics[model_type]['accuracy'] > best_accuracy:
        best_accuracy = metrics[model_type]['accuracy']
        best_model = model
        best_model_type = model_type

# Calculate combined model accuracy
combined_predictions = []
for model_type in model_types:
    if model_type in models:
        model, test_loader = models[model_type]
        _, y_pred = evaluate_model(model, test_loader)
        logging.info(f"Model {model_type} predictions length: {len(y_pred)}")
        combined_predictions.append(y_pred)

# Ensure all predictions have the same length as y_true_all
if not combined_predictions:
    logging.error("No predictions collected for combined accuracy.")
    exit(1)

for i, preds in enumerate(combined_predictions):
    if len(preds) != len(y_true_all):
        logging.error(f"Prediction length mismatch for model {model_types[i]}: {len(preds)} vs {len(y_true_all)}")
        exit(1)

# Convert to numpy array and apply majority voting
combined_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=np.array(combined_predictions).T)
logging.info(f"Combined predictions length: {len(combined_pred)}")
logging.info(f"Ground truth length: {len(y_true_all)}")

# Calculate combined accuracy
combined_accuracy = accuracy_score(y_true_all, combined_pred)
logging.info(f"Combined Model Accuracy: {combined_accuracy:.4f}")

# Save the best model
if best_model:
    torch.save(best_model.state_dict(), model_output)
    logging.info(f"Best model ({best_model_type}) saved to {model_output}")
else:
    logging.error("No model was trained successfully.")
    exit(1)

# Check if accuracy meets the 70% target
if best_accuracy < 0.7:
    logging.warning(f"Best model ({best_model_type}) accuracy {best_accuracy:.4f} is below the 70% target. Consider adjusting hyperparameters or features.")

# Step 7: Evaluation and Visualization
# Plot training curves
plt.figure(figsize=(12, 4))
for model_type in model_types:
    if model_type in histories:
        plt.plot(histories[model_type]['accuracy'], label=f'{model_type} Train')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, 'training_accuracy.png'))
plt.close()

# Plot metrics comparison
metrics_df = pd.DataFrame(metrics).T
plt.figure(figsize=(10, 6))
metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'))
plt.close()

# Cross-validation accuracy comparison
cv_accuracies = [metrics[model_type]['cv_accuracy'] for model_type in model_types]
plt.figure(figsize=(8, 5))
plt.bar(model_types, cv_accuracies)
plt.title('Cross-Validation Accuracy Comparison')
plt.ylabel('Mean CV Accuracy')
plt.savefig(os.path.join(results_dir, 'cv_accuracy_comparison.png'))
plt.close()

# Plot precision, recall, F1-score
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
metrics_df['precision'].plot(kind='bar', ax=ax1, title='Precision Comparison')
metrics_df['recall'].plot(kind='bar', ax=ax2, title='Recall Comparison')
metrics_df['f1'].plot(kind='bar', ax=ax3, title='F1-Score Comparison')
ax1.set_ylabel('Precision')
ax2.set_ylabel('Recall')
ax3.set_ylabel('F1-Score')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'precision_recall_f1_comparison.png'))
plt.close()

# Step 8: Refactoring Logic
def refactor_code(row, predicted_smell):
    method_name = row['fullMethodName']
    if pd.isna(method_name):
        method_name = "public void sampleMethod() { // Complex logic }"
    
    refactoring_techniques = {
        '1': ('Extract Method', 
              f"public void refactoredMethod() {{ extractedLogic(); }}\nprivate void extractedLogic() {{ {method_name} }}"),
        '2': ('Rename Method', 
              f"public void improvedMethodName() {{ {method_name.split('(')[1] if '(' in method_name else '// Logic'} }}"),
        '3': ('Inline Method', 
              f"public void inlinedMethod() {{ // Inlined logic from {method_name} }}"),
        '4': ('Replace Temp with Query', 
              f"public int getValue() {{ return computeValue(); }}\nprivate int computeValue() {{ return {method_name.split('(')[1] if '(' in method_name else '0'}; }}"),
    }
    
    technique, refactored = refactoring_techniques.get(str(predicted_smell), ('No Refactoring', method_name))
    return technique, refactored

# Step 9: Sample Test Example
test_row = df.iloc[0]
test_features = pd.DataFrame([df.iloc[0][numerical_features].astype(float).clip(upper=max_value).fillna(0)], columns=numerical_features)
test_features_scaled = scaler.transform(test_features).astype(np.float32)
test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)
if best_model_type != 'Dense':
    test_tensor = test_tensor.unsqueeze(1)  # [1, 1, 65] for RNN-based models
else:
    test_tensor = test_tensor  # [1, 65] for Dense model
best_model.eval()
with torch.no_grad():
    predicted_probs = best_model(test_tensor)
    predicted_class = torch.argmax(predicted_probs, dim=1).cpu().numpy()[0]
predicted_smell = le.inverse_transform([predicted_class])[0]
technique, refactored_code = refactor_code(test_row, predicted_smell)

logging.info("\nSample Test Example:")
logging.info(f"Input Method (from fullMethodName): {test_row['fullMethodName']}")
logging.info(f"Detected Smell Type: {predicted_smell}")
logging.info(f"Refactoring Technique: {technique}")
logging.info(f"Refactored Code:\n{refactored_code}")

# Step 10: Documentation
architecture_doc = """
# Method-Level Code Smell Classifier and Refactorer

## Project Overview
This project implements a deep learning-based system to classify method-level code smells and apply refactoring, based on the Refactoring-AI repository (https://github.com/refactoring-ai/predicting-refactoring-ml). It uses the `refactoring50.sql` dataset, focusing on the top {max_rows} rows of the `no` table.

## Dataset
- Source: refactoring50.sql, table `no`
- Features: 65 numerical features (e.g., classLoc, methodWmc, bugFixCount)
- Target: `type` (code smell type, categorical)
- Rows Used: Top {max_rows} rows
- Columns: {columns}

## Preprocessing
1. Extracted data directly from SQL using regex-based parsing.
2. Selected 65 numerical features, capped at {max_value}, filled NaNs with 0, converted to float32.
3. Applied SMOTE to handle class imbalance.
4. Encoded target labels (`type`) using LabelEncoder.
5. Scaled features using StandardScaler, converted to float32.
6. Reshaped data for RNN input: [samples, 1, 65].

## Model Architecture
- Models Tested: RNN, LSTM, GRU, Bi-RNN, Dense
- Best Model: {best_model_type}
- RNN/LSTM/GRU/Bi-RNN Architecture:
  - Input: Shape [batch, 1, {len_numerical_features}]
  - RNN Layer: 32 units
  - Dense Layers: 16 units (ReLU), {num_classes} units
- Dense Architecture:
  - Input: Shape [batch, {len_numerical_features}]
  - Dense Layers: 64, 32 units (ReLU), {num_classes} units
- Optimizer: Adam
- Loss: CrossEntropyLoss with class weights
- Device: {device}

## Architecture Diagram (Text-Based)
[SQL Dataset: refactoring50.sql]
|
v
[Data Extraction: Top {max_rows} rows]
|
v
[Preprocessing: 65 features, SMOTE, Scaling]
|
v
[Train-Test Split: 80-20]
|
v
[Models: RNN, LSTM, GRU, Bi-RNN, Dense]
|
v
[Training: 5 epochs, Adam, Weighted Loss]
|
v
[Evaluation: Accuracy, Precision, Recall, F1]
|
v
[Best Model Selection: Highest Accuracy]
|
v
[Refactoring: Extract, Rename, Inline, Replace]
|
v
[Outputs: Model, Plots, Documentation]

## Training
- Epochs: 5
- Batch Size: 2
- Shuffling: Enabled for training
- Cross-Validation: 3-fold

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Visualizations (in {results_dir}):
  - Training accuracy curves (training_accuracy.png)
  - Metrics comparison (metrics_comparison.png)
  - Confusion matrices for all models (confusion_matrix_rnn.png, confusion_matrix_lstm.png, etc.)
  - Cross-validation accuracy comparison (cv_accuracy_comparison.png)
  - Precision, Recall, F1-Score comparison (precision_recall_f1_comparison.png)

## Refactoring
- Techniques: Extract Method, Rename Method, Inline Method, Replace Temp with Query
- Logic: Rule-based mapping from predicted smell type to refactored code, using `fullMethodName`
- Example:
  - Input: {test_row_fullMethodName}
  - Smell: {predicted_smell}
  - Technique: {technique}
  - Refactored: {refactored_code}

## Results
- Best Model: {best_model_type}
- Test Accuracy: {best_accuracy:.4f}
- Combined Model Accuracy: {combined_accuracy:.4f}
- Saved Model: {model_output}
- Visualizations: Saved in {results_dir}

## Notes
- If a visual architecture diagram is needed, use tools like draw.io or Graphviz based on the text-based diagram above.
- If accuracy is below 70%, adjust epochs (e.g., 100) or learning rate (e.g., 0.0001).
"""
with open(os.path.join(results_dir, 'architecture_documentation.txt'), 'w') as f:
    f.write(architecture_doc.format(
        max_rows=max_rows,
        columns=', '.join(columns),
        max_value=max_value,
        best_model_type=best_model_type,
        len_numerical_features=len(numerical_features),
        num_classes=num_classes,
        device=device,
        test_row_fullMethodName=test_row['fullMethodName'],
        predicted_smell=predicted_smell,
        technique=technique,
        refactored_code=refactored_code,
        best_accuracy=best_accuracy,
        combined_accuracy=combined_accuracy,
        model_output=model_output,
        results_dir=results_dir
    ))

# Print final metrics
logging.info("\nFinal Metrics:")
for model_type, metric in metrics.items():
    logging.info(f"{model_type}:")
    for key, value in metric.items():
        logging.info(f"  {key}: {value:.4f}")
logging.info(f"Combined Model Accuracy: {combined_accuracy:.4f}")
logging.info("\nProcessing complete.")
logging.info(f"The best model is: {best_model_type}")