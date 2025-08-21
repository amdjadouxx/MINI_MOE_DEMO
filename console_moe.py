#!/usr/bin/env python3
"""
Version Console du MoE - Sans Gradio
===================================
Interface simple en ligne de commande
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os

# Importer les classes du fichier app.py
import sys

sys.path.append(".")
from app import MoETextClassifier


def console_interface():
    """Interface console simple"""
    print("🚀 MoE Text Classifier - Console Version")
    print("=" * 50)

    # Initialiser le classifier
    classifier = MoETextClassifier(max_features=500, num_experts=3)

    # Charger ou entraîner le modèle
    if not classifier.load_model():
        print("📚 Training new model...")
        input_dim = classifier.load_and_preprocess_data(
            sample_size=500, use_synthetic=True
        )
        classifier.create_model(input_dim)
        classifier.train(epochs=3, learning_rate=0.01)
        classifier.save_model()

        accuracy = classifier.evaluate()
        print(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    else:
        print("✅ Model loaded successfully!")

    # Menu principal
    while True:
        print("\n" + "=" * 50)
        print("🎯 Menu Principal")
        print("1. Classifier un texte")
        print("2. Exemples prédéfinis")
        print("3. Quitter")
        print("-" * 50)

        choice = input("Votre choix (1-3): ").strip()

        if choice == "1":
            # Classification personnalisée
            text = input("\n📝 Entrez votre texte: ").strip()
            if text:
                classify_and_show(classifier, text)

        elif choice == "2":
            # Exemples prédéfinis
            examples = [
                "Scientists discover new method to generate clean energy from solar panels",
                "The football team won the championship after a spectacular final match",
                "Stock markets rose sharply following new economic policies",
                "International peace talks continue amid growing tensions",
            ]

            print("\n🎮 Tests avec exemples prédéfinis:")
            for i, example in enumerate(examples, 1):
                print(f"\n--- Exemple {i} ---")
                print(f"📝 Texte: {example}")
                classify_and_show(classifier, example)

        elif choice == "3":
            print("\n👋 Au revoir !")
            break

        else:
            print("❌ Choix invalide!")


def classify_and_show(classifier, text):
    """Classifier et afficher les résultats"""
    try:
        predicted_label, class_probs, expert_weights = classifier.predict(text)

        print(f"🎯 Prédiction: {predicted_label}")
        print(f"🔥 Confiance: {max(class_probs):.1%}")

        print("\n📊 Probabilités par classe:")
        for i, (class_name, prob) in enumerate(
            zip(classifier.class_names, class_probs)
        ):
            bar = "█" * int(prob * 30)
            print(f"   {class_name:12s}: {prob:.1%} {bar}")

        print("\n🤖 Contribution des experts:")
        for i, weight in enumerate(expert_weights):
            bar = "█" * int(weight * 30)
            print(f"   Expert {i+1:2d}   : {weight:.1%} {bar}")

    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    try:
        console_interface()
    except KeyboardInterrupt:
        print("\n\n👋 Programme interrompu!")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
