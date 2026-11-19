# NeuroGuard IDS — Neural Network Network-Flow Classifier

A complete, **beginner-friendly local educational demonstration** of an AI-based intrusion detection concept. The model is a scikit-learn **MLPClassifier** (multi-layer perceptron) with 2 hidden layers and a Streamlit dashboard. It does **not** inspect live network traffic, attack hosts, scan ports, or protect a production network.

> **Important limitation:** The bundled records are **fictional synthetic examples**. High test accuracy only means the model learned patterns in those examples. It is **not** evidence of useful real-world intrusion detection. Treat all predictions as classroom demonstrations.

## Files

```text
NeuroGuard-IDS/
├── RUN_WINDOWS.bat         # Double-click to set up and launch (Windows 11)
├── app.py                  # Streamlit interactive dashboard
├── core.py                 # Input schema, preprocessing, MLP model, inference
├── generate_data.py        # Create synthetic labeled training and demo CSV
├── train_model.py          # Train/test model, save report and local joblib
├── predict_csv.py          # CSV classifier using terminal
├── neural_visualizer.py    # Real forward-pass activation + HTML animation
├── neuroguard_animation_preview.html  # Open directly in a browser
├── requirements.txt        # Libraries to install
├── data/
│   ├── synthetic_training.csv # Generated demonstration flow records
│   └── sample_flows.csv       # Generated demo flow records without labels
├── models/                 # Your locally trained files go here
├── tests/
│   ├── test_smoke.py        # Core quick self-check
│   └── test_visualizer.py   # Verify visualization matches the model
├── .github/workflows/
│   └── python-tests.yml     # CI test runner on GitHub
├── GITHUB_SETUP.md          # Steps to publish a separate GitHub repository
└── README.md
```

## Requirements (Windows 11)

- **Python 3.12 or 3.11**, installed from [python.org](https://www.python.org/downloads/). In the installer, select "Add python.exe to PATH" if offered. The Python launcher `py` is useful.
- Internet connection once to install Python libraries.
- 1–2 GB free disk space is a comfortable allowance for Python and dependencies. No dedicated GPU is required.
- VS Code is optional.

### Option A: easiest way

1. Extract the ZIP into a folder, for example `Desktop\NeuroGuard-IDS`.
2. Double-click `RUN_WINDOWS.bat`.
3. A terminal creates a local `.venv`, installs dependencies, generates synthetic data if missing, trains the model if missing, and starts Streamlit.
4. Open the local browser address shown by Streamlit, usually `http://localhost:8501`.
5. To stop, return to the terminal and press **Ctrl+C**.

If Windows warns about opening a downloaded script, inspect `RUN_WINDOWS.bat` in Notepad before you run it. Do not run scripts from sources you do not trust.

### Option B: step-by-step using PowerShell (recommended for learning)

Open PowerShell **in the extracted folder** (File Explorer address bar -> type `powershell` -> Enter). Run each command separately:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you have Python 3.11 instead of 3.12, replace **`-3.12`** with **`-3.11`** in the first two commands. You don't need to activate the virtual environment, so PowerShell's script execution policy is not an issue.

### What you should see

When training completes, the terminal shows accuracy, precision, recall and F1 score **on a synthetic hold-out test set**. The dashboard includes:

- Overview (classifications of demo records)
- Check one flow (manual numerical inputs)
- Analyze a CSV (upload a **local summary CSV**, inspect results and download the predictions)
- Model evaluation (metrics and confusion matrix)
- **Neural network animation** (pause / replay / speed; real neuron activations from selected example)
- How the AI works (short visual explanation and CSV schema)

## Watch the neural network animation

### Quickest way (no installation)

Open `neuroguard_animation_preview.html` with Edge or Chrome. It is a **self-contained, offline HTML preview** generated from one sample row and the saved trained model. Use **Pause**, **Replay**, and **Speed**. It is a snapshot, so changing your trained model does not modify this preview automatically.

### In the Streamlit dashboard (recommended)

1. Start `RUN_WINDOWS.bat`, or use `python -m streamlit run app.py` in your virtual environment.
2. Open `http://localhost:8501` in your browser.
3. Select **🎞️ Neural network animation** in the left sidebar.
4. Choose a sample CSV record. The diagram uses **actual values** calculated from the loaded trained model.
5. Hover over a neuron for its numerical activation and expand **See the actual numbers**.
6. Select a different record to watch the classification change.

The diagram has 11 preprocessed input units → 32 ReLU neurons → 16 ReLU neurons → 1 sigmoid score. Teal connections indicate positive weighted contributions and orange indicates negative ones. It shows **only a few of the strongest connections** to avoid clutter. The progressive lighting and moving dashes represent a **slow-motion teaching replay of one forward pass**, not real inference speed, training updates, backpropagation, or live network traffic.

To **regenerate the standalone preview** after retraining, run this Python snippet from the project folder:

```python
from pathlib import Path
import joblib
from neural_visualizer import write_offline_preview

model = joblib.load('models/model.joblib')['model']
write_offline_preview(model, Path('data/sample_flows.csv'),
                      Path('neuroguard_animation_preview.html'))
```

Only load model files you trained or obtained from a trusted source. Serialized joblib files can execute arbitrary code when loaded.

## Terminal-only CSV prediction

After generating data and training:

```powershell
.\.venv\Scripts\python.exe predict_csv.py
```

This produces `data\predictions.csv` and prints several classifications.

## CSV feature requirements

For **prediction**, CSV must have this header (order may vary):

```csv
protocol,duration_ms,dst_port,src_bytes,dst_bytes,packets,syn_count,failed_logins,connections_per_minute
```

`protocol` must be `TCP`, `UDP` or `ICMP`. All remaining columns must be non-negative numbers. `dst_port` must be between 0 and 65535.

For **training**, add one more column: `label` with `0` (Normal) or `1` (Suspicious), or words `Normal` / `Suspicious`. Supply at least 10 rows for each class, though any meaningful real-world study needs far more well-labeled data and stronger evaluation.

To train on your own **authorized, properly prepared, labeled flow-summary dataset**:

```powershell
.\.venv\Scripts\python.exe train_model.py --data data\my_labeled_flows.csv
```

Your CSV must already use the **nine listed features**; downloading an unrelated dataset does not automatically make its column names compatible. The new training overwrites `models\model.joblib` and `models\metrics.json`. Restart the dashboard after training if necessary.

## Neural network architecture

```
9 input fields (8 numerical + protocol)
                |
                v
StandardScaler (numerics) + OneHotEncoder (protocol)
                |
                v
        11 encoded inputs
                |
                v
       32 hidden neurons (ReLU)
                |
                v
       16 hidden neurons (ReLU)
                |
                v
     Binary classification output
      Normal / Suspicious
```

**Training loop:** give a row of inputs -> produce an output -> compare it to the known label -> calculate error -> backpropagation updates weights -> repeat. The `MLPClassifier` uses the Adam optimizer here. **Evaluation** uses a held-out 25% test split; internal early stopping also reserves a fraction of the training set.

**What the model score means:** `suspicious_score` is the classifier's numerical score for the Suspicious class. It is not guaranteed to be calibrated and must **not** be interpreted as the true probability of an attack.

## Quick test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Common Windows errors

- `py is not recognized`: Install Python (with launcher), close and reopen PowerShell, or use an installed Python executable to create the venv.
- `No suitable Python runtime found`: Check `py --list` and use the Python version you installed.
- `No module named streamlit`: Re-run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- `model.joblib missing`: Run `generate_data.py` and then `train_model.py` before starting the dashboard.
- `Port 8501 is already in use`: Close the other Streamlit process, or run `... -m streamlit run app.py --server.port 8502`.
- `Running scripts is disabled`: You don't have to use `Activate.ps1`; run `.\.venv\Scripts\python.exe` directly as shown above.
- `CSV invalid`: Check the exact nine column names and ensure `protocol` is TCP/UDP/ICMP, with no blanks.

## Limits and responsible use

This project works entirely on existing summary CSV files. It neither captures packets nor performs any network activity. It is **not a firewall**, cannot block an attack, and cannot verify that an event is safe. A suspicious prediction is only a model output, not an established security incident. For coursework using real data, use datasets you are authorized to analyze, check feature definitions and labels, test on independent data, and document false positives and false negatives.

## References

- scikit-learn MLP classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- Streamlit local app installation: https://docs.streamlit.io/get-started/installation/command-line

## Publish on GitHub

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for instructions to create a dedicated repository and push this project. The prepared GitHub version intentionally excludes locally generated `.joblib` models, local evaluation files, and credentials.