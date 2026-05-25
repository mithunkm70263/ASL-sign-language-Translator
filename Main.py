import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Feature engineering helpers
# ──────────────────────────────────────────────

def normalize_landmarks(row_values):
    """
    Wrist-relative normalization:
      1. Subtract wrist (landmark 0) so the hand is origin-centred.
      2. Divide by the Euclidean distance between wrist (0) and middle-finger MCP (9)
         — this makes features scale-invariant.
    Expects 63 values: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    Returns 63 normalised values.
    """
    coords = np.array(row_values, dtype=np.float32).reshape(21, 3)
    wrist = coords[0].copy()
    coords -= wrist                        # translate to wrist origin

    # scale = distance from wrist to middle-finger MCP (landmark 9)
    scale = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale
    return coords.flatten()


def angle_between(v1, v2):
    """Angle in degrees between two 3-D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# Finger joint triplets: (A, B, C) → angle at joint B
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20) # Pinky
]


def compute_angles(coords_3x21):
    """
    coords_3x21: numpy array shape (21, 3) — already wrist-normalised
    Returns a 1-D array of 15 joint angles (degrees).
    """
    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        v1 = coords_3x21[a] - coords_3x21[b]
        v2 = coords_3x21[c] - coords_3x21[b]
        angles.append(angle_between(v1, v2))
    return np.array(angles, dtype=np.float32)


def build_feature_row(raw_63):
    """
    raw_63: flat list/array of 63 raw landmark values (x0,y0,z0,...,x20,y20,z20)
    Returns a 1-D feature vector of 78 values: 63 normalised + 15 angles.
    """
    norm = normalize_landmarks(raw_63)           # 63
    angles = compute_angles(norm.reshape(21, 3)) # 15
    return np.concatenate([norm, angles])        # 78


# ──────────────────────────────────────────────
# Load & transform dataset
# ──────────────────────────────────────────────

DATA_PATH = "/Users/mithunkm/Desktop/Sign recognition IDT/landmarks.csv"
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)

raw_cols = [col for col in df.columns if col != "label"]
labels_series = df["label"]

print("Engineering features (normalisation + angles) …")
feature_rows = []
for _, row in df.iterrows():
    feature_rows.append(build_feature_row(row[raw_cols].values))

X = np.array(feature_rows, dtype=np.float32)
y = labels_series.to_numpy(dtype=str)

print(f"Dataset shape: {X.shape}   Classes: {np.unique(y)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.ensemble import ExtraTreesClassifier

print("Training ExtraTrees (50 trees) …")
model = ExtraTreesClassifier(
    n_estimators=50,
    max_depth=30,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("ExtraTrees model trained.")



train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"\nTrain Accuracy : {train_acc:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test), zero_division=0))



joblib.dump(model, "asl_model.pkl", compress=3)
labels = sorted(np.unique(y).tolist())
joblib.dump(labels, "labels.pkl")
print("\nModel saved → asl_model.pkl")
print("Labels saved → labels.pkl")
