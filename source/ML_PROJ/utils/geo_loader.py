import numpy as np

EXPECTED_GENES = 22277

def load_geo_series_matrix(path):
    """
    Load GEO .txt series_matrix
    Trả về: X shape (n_samples, 22277)
    """
    data = []
    start = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("!series_matrix_table_begin"):
                start = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if start:
                if line.startswith("ID_REF") or line.startswith('"ID_REF"'):
                    continue

                parts = line.split("\t")

                values = [v.strip().strip('"') for v in parts[1:]]

                data.append(values)

    X = np.array(data, dtype=float).T 

    if X.shape[1] != EXPECTED_GENES:
        print(f"[WARN] Feature mismatch: {X.shape[1]} → {EXPECTED_GENES}")
        X = X[:, :EXPECTED_GENES]

    print("GEO loaded:", X.shape)
    return X
