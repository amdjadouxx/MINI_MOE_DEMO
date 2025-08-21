# 🧠 Mixture of Experts (MoE) Text Classifier Demo

Une démonstration interactive d'un modèle **Mixture of Experts** pour la classification de texte, utilisant le dataset AG News et une interface Gradio.

## 🎯 Aperçu du Projet

Ce projet implémente un système MoE complet qui :

- **Charge et prétraite** le dataset AG News depuis HuggingFace
- **Entraîne un modèle MoE** avec 3 experts spécialisés et un routeur intelligent
- **Fournit une interface interactive** Gradio pour tester le modèle en temps réel
- **Visualise** les probabilités de classes et la distribution des experts

## 🔬 Architecture du Modèle

### Mixture of Experts (MoE)

Le MoE est une architecture qui utilise plusieurs réseaux "experts" spécialisés :

- **3 Experts** : Chacun est un MLP (Multi-Layer Perceptron) spécialisé dans certains types de contenu
- **1 Routeur** : Réseau qui décide dynamiquement quels experts utiliser pour chaque entrée
- **Combinaison pondérée** : Les sorties des experts sont combinées selon les poids du routeur

### Avantages
- **Spécialisation** : Chaque expert peut se spécialiser sur différents types de contenu
- **Efficacité** : Seuls les experts pertinents sont fortement activés
- **Scalabilité** : Facile d'ajouter plus d'experts pour des tâches complexes

## 📊 Dataset et Classes

- **Dataset** : AG News (actualités en anglais)
- **Classes** : 4 catégories
  - 🌍 **World** : Actualités internationales
  - ⚽ **Sports** : Sport et compétitions
  - 💼 **Business** : Économie et finance
  - 🔬 **Sci/Tech** : Science et technologie

## 🚀 Installation et Exécution

### Option 1 : Installation locale

```bash
# Cloner le repository
git clone <repository-url>
cd MINI_MOE_DEMO

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

### Option 2 : Docker

```bash
# Construire l'image Docker
docker build -t moe-demo .

# Lancer le conteneur
docker run -p 7860:7860 moe-demo
```

## 🌐 Interface Gradio

L'interface web sera accessible à l'adresse : `http://localhost:7860`

### Fonctionnalités de l'interface :

1. **Zone de saisie** : Entrez votre texte à classifier
2. **Prédiction** : Affiche la classe prédite avec la confiance
3. **Graphique des probabilités** : Visualise les scores pour chaque classe
4. **Distribution des experts** : Montre quels experts ont été utilisés
5. **Exemples prédéfinis** : Testez rapidement avec des exemples

## 🔧 Détails Techniques

### Prétraitement
- **Vectorisation** : TF-IDF avec 2000 features maximum
- **Normalisation** : Suppression des mots vides anglais

### Architecture des Experts
```
Expert (MLP) :
├── Linear(2000 → 128) + ReLU + Dropout(0.3)
├── Linear(128 → 64) + ReLU + Dropout(0.3)
└── Linear(64 → 4)
```

### Architecture du Routeur
```
Router (MLP) :
├── Linear(2000 → 64) + ReLU + Dropout(0.2)
├── Linear(64 → 32) + ReLU
└── Linear(32 → 3) + Softmax
```

### Entraînement
- **Optimiseur** : Adam (lr=0.01)
- **Loss** : CrossEntropyLoss + diversité du routage
- **Epochs** : 8 (optimisé pour la démo)
- **Batch size** : 64

## 📈 Performance

Le modèle atteint généralement :
- **Précision** : ~85-90% sur le test set
- **Temps d'entraînement** : ~2-3 minutes sur CPU
- **Temps d'inférence** : <100ms par prédiction

## 🔍 Exemples d'Utilisation

Testez avec ces types de textes :

```python
# Science/Tech
"Scientists discover new method to generate clean energy from solar panels."

# Sports  
"The football team won the championship after a spectacular final match."

# Business
"Stock markets rose sharply following the announcement of new economic policies."

# World
"International peace talks continue amid growing tensions between nations."
```

## 📁 Structure du Projet

```
MINI_MOE_DEMO/
├── app.py              # Code principal avec MoE + Gradio
├── requirements.txt    # Dépendances Python
├── README.md          # Documentation (ce fichier)
├── Dockerfile         # Configuration Docker
├── LICENSE            # Licence du projet
└── moe_model.pth      # Modèle sauvegardé (généré après entraînement)
```

## 🧪 Analyse des Experts

Après l'entraînement, vous pouvez observer :

- **Expert 1** : Tend à se spécialiser dans les actualités internationales
- **Expert 2** : Se concentre sur les contenus sports/loisirs  
- **Expert 3** : Gère mieux les contenus business/tech

Le routeur apprend automatiquement à diriger chaque type de contenu vers l'expert le plus approprié.

## 🛠️ Personnalisation

### Modifier le nombre d'experts
```python
classifier = MoETextClassifier(num_experts=5)  # Au lieu de 3
```

### Ajuster les hyperparamètres
```python
classifier.train(epochs=15, learning_rate=0.005, batch_size=32)
```

### Changer le dataset
Remplacez `load_dataset("ag_news")` par votre dataset HuggingFace préféré.

## 📝 Notes Techniques

- **GPU Support** : Détection automatique CUDA si disponible
- **Sauvegarde** : Le modèle est automatiquement sauvegardé après l'entraînement
- **Robustesse** : Gestion d'erreurs et validation des entrées
- **Performance** : Optimisé pour une utilisation interactive

## 🤝 Contribution

Ce projet est conçu comme une démonstration éducative. N'hésitez pas à :
- Expérimenter avec différentes architectures
- Tester sur d'autres datasets
- Améliorer l'interface utilisateur
- Optimiser les performances

## 📄 Licence

Voir le fichier `LICENSE` pour les détails de la licence.
