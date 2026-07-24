import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from deap import base, creator, tools, algorithms

from app.config.config import settings
from app.ml.tabular.knn_wrapper import knn_wrapper

logger = logging.getLogger(__name__)

# Cache variables to avoid reloading
_X_val = None
_y_val = None
_img_probs = None

def get_genetic_evaluation_data():
    """
    Prepares validation split of tabular data and dummy/cached image probabilities
    to run genetic optimization.
    """
    global _X_val, _y_val, _img_probs
    if _X_val is not None:
        return _X_val, _y_val, _img_probs
        
    csv_path = Path(settings.DATASET_CSV_PATH)
    if not csv_path.exists():
        alt_path = Path(__file__).resolve().parents[4] / "kidney_multimodal_dataset_FIXED.csv"
        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError("Dataset CSV not found")
            
    df = pd.read_csv(csv_path)
    
    # Target and Features
    y = df["classification"].values
    X = df.drop(columns=["id", "image_path", "classification"], errors="ignore")
    X = X[knn_wrapper.features]
    
    # Split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _X_val = X_val.reset_index(drop=True)
    _y_val = y_val
    
    # Simulate image model probabilities for these validation samples
    # If the ResNet18 model is trained, we could fetch predictions, but to remain self-contained
    # and fast, we map them directly based on classification + sc severity, with small noise.
    # Normal -> Normal (class 0, prob close to 0.0)
    # Stone/Tumor -> Abnormal (class 1, prob close to 1.0)
    img_probs_list = []
    for idx, row in _X_val.iterrows():
        classification_val = _y_val[idx]
        sc_val = row.get("sc", 1.0)
        
        # Simple probability assignment matching dataset rules
        if classification_val == 1:
            if sc_val > 3.0:
                # Tumor (Severe)
                p_tumor = random.uniform(0.7, 0.99)
                p_stone = random.uniform(0.01, 0.2)
                p_cyst = random.uniform(0.01, 0.1)
            else:
                # Stone (Moderate)
                p_tumor = random.uniform(0.01, 0.1)
                p_stone = random.uniform(0.6, 0.95)
                p_cyst = random.uniform(0.01, 0.3)
            p_normal = 1.0 - (p_tumor + p_stone + p_cyst)
        else:
            p_normal = random.uniform(0.8, 0.99)
            p_cyst = random.uniform(0.01, 0.1)
            p_stone = random.uniform(0.00, 0.1)
            p_tumor = 1.0 - (p_normal + p_cyst + p_stone)
            
        # Clinical risk mapper
        img_ckd_prob = (p_cyst * 0.3) + (p_stone * 0.6) + (p_tumor * 0.95)
        img_probs_list.append(min(max(img_ckd_prob, 0.0), 1.0))
        
    _img_probs = np.array(img_probs_list)
    return _X_val, _y_val, _img_probs

def build_dynamic_pipeline(X: pd.DataFrame, feature_subset: list) -> Pipeline:
    X_sub = X[feature_subset]
    categorical_cols = X_sub.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_cols = X_sub.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=3, p=1, weights="uniform")),
    ])
    return pipeline

def optimize_multimodal_weights(pop_size=20, n_gen=5) -> dict:
    """
    Run Genetic Algorithm using DEAP.
    Optimizes a 25-gene individual:
      - Genes 0-23: Binary flags representing tabular feature selection.
      - Gene 24: Float [0.0, 1.0] representing the tabular weight (w_tabular).
    """
    logger.info("Initializing DEAP Genetic Optimization...")
    X_val, y_val, img_probs = get_genetic_evaluation_data()
    features = knn_wrapper.features
    num_features = len(features)
    
    # 1. Define Fitness and Individual classes
    # We must check if creator already contains these to avoid Redefinition errors in hot-reloads
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
    toolbox = base.Toolbox()
    
    # Individual generation: 24 binary integers + 1 float
    def create_ind():
        genes = [random.randint(0, 1) for _ in range(num_features)] # Feature selection
        genes.append(random.uniform(0.1, 0.9)) # Tabular Weight
        return creator.Individual(genes)
        
    toolbox.register("individual", create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 2. Define evaluation function
    def evaluate(individual):
        feature_mask = individual[:num_features]
        tab_weight = individual[num_features]
        
        # Ensure at least 1 feature is selected
        if sum(feature_mask) == 0:
            return (0.0,)
            
        selected_features = [features[i] for i, mask in enumerate(feature_mask) if mask == 1]
        
        try:
            # Retrain KNN on feature subset using 5-fold cross-validation or train/test split
            # For speed, we do a rapid train/test split on X_val
            X_tr, X_te, y_tr, y_te = train_test_split(X_val[selected_features], y_val, test_size=0.3, random_state=42)
            
            # Check if all classes are present
            if len(np.unique(y_tr)) < 2:
                return (0.0,)
                
            clf = build_dynamic_pipeline(X_val, selected_features)
            clf.fit(X_tr, y_tr)
            
            # Get tabular probabilities on test split
            tab_probs_te = clf.predict_proba(X_te)[:, 1]
            
            # Select matching indices of simulated image probabilities
            te_indices = X_te.index.values
            img_probs_te = img_probs[te_indices]
            y_te_arr = y_val[te_indices]
            
            # Late Fusion computation
            w_tab = tab_weight
            w_img = 1.0 - tab_weight
            fused_probs = (w_tab * tab_probs_te) + (w_img * img_probs_te)
            
            preds = (fused_probs >= 0.5).astype(int)
            f1 = f1_score(y_te_arr, preds, average="weighted")
            
            # Return F1 score
            return (f1,)
        except Exception as e:
            logger.error(f"Error evaluating individual in GA: {e}")
            return (0.0,)
            
    toolbox.register("evaluate", evaluate)
    
    # Crossover: Custom crossover (two-point for binary, blend for float)
    def mate(ind1, ind2):
        # Crossover binary features
        tools.cxTwoPoint(ind1[:num_features], ind2[:num_features])
        # Crossover float weight
        w1, w2 = ind1[num_features], ind2[num_features]
        ind1[num_features] = (w1 + w2) / 2
        ind2[num_features] = random.uniform(min(w1, w2), max(w1, w2))
        return ind1, ind2
        
    toolbox.register("mate", mate)
    
    # Mutation: Custom mutation
    def mutate(individual):
        # Mutate binary flags
        for i in range(num_features):
            if random.random() < 0.1:
                individual[i] = 1 - individual[i]
        # Mutate float weight
        if random.random() < 0.2:
            individual[num_features] = min(max(individual[num_features] + random.normalvariate(0, 0.1), 0.0), 1.0)
        return (individual,)
        
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run Genetic Algorithm
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    logger.info("Executing genetic generation loop...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
    
    best_ind = hof[0]
    best_features = [features[i] for i, mask in enumerate(best_ind[:num_features]) if mask == 1]
    best_tab_weight = float(best_ind[num_features])
    best_img_weight = 1.0 - best_tab_weight
    
    result = {
        "best_f1": float(best_ind.fitness.values[0]),
        "tabular_weight": round(best_tab_weight, 4),
        "image_weight": round(best_img_weight, 4),
        "selected_features": best_features,
        "dropped_features": [f for f in features if f not in best_features]
    }
    
    logger.info(f"Genetic Optimization complete: {result}")
    return result
