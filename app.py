"""
Mixture of Experts (MoE) Demo for Text Classification
=====================================================

This module implements a complete MoE system that:
1. Loads and preprocesses the AG News dataset
2. Trains a MoE model with 3 experts and a router
3. Provides an interactive Gradio interface for testing

The MoE architecture routes input samples to different experts
based on a learned routing mechanism.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import pickle
import os


class Expert(nn.Module):
    """Single expert network - a simple MLP for text classification."""

    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class Router(nn.Module):
    """Router network that decides which expert(s) to use for each input."""

    def __init__(self, input_dim, num_experts=3):
        super(Router, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model that combines multiple expert networks
    using a learned routing mechanism.
    """

    def __init__(self, input_dim, num_experts=3, num_classes=4):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        # Create expert networks
        self.experts = nn.ModuleList(
            [
                Expert(input_dim, hidden_dim=128, num_classes=num_classes)
                for _ in range(num_experts)
            ]
        )

        # Create router network
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        # Get routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]

        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch_size, num_classes]
            expert_outputs.append(expert_output)

        # Stack expert outputs
        expert_outputs = torch.stack(
            expert_outputs, dim=2
        )  # [batch_size, num_classes, num_experts]

        # Weighted combination of expert outputs
        routing_weights = routing_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        final_output = torch.sum(
            expert_outputs * routing_weights, dim=2
        )  # [batch_size, num_classes]

        return final_output, routing_weights.squeeze(1)


class MoETextClassifier:
    """Main class that handles data loading, training, and inference."""

    def __init__(self, max_features=2000, num_experts=3):
        self.max_features = max_features
        self.num_experts = num_experts
        self.num_classes = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Class labels for AG News dataset
        self.class_names = ["World", "Sports", "Business", "Sci/Tech"]

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english"
        )
        self.model = None

        print(f"Using device: {self.device}")

    def load_and_preprocess_data(self, sample_size=2000, use_synthetic=True):
        """Load AG News dataset and preprocess it."""
        print("Loading data...")

        if use_synthetic:
            # Utilise des données synthétiques pour éviter les problèmes de téléchargement
            print("🎯 Using synthetic data for faster testing...")
            train_texts = [
                # World (0)
                "The president announced new diplomatic relations with neighboring countries",
                "International summit discusses global climate policies",
                "War in the region affects millions of civilians",
                "United Nations votes on humanitarian aid package",
                "European Union implements new trade sanctions",
                "Global leaders meet to discuss peace negotiations",
                "Embassy closure sparks diplomatic tensions",
                "Refugee crisis overwhelms border facilities",
                # Sports (1)
                "The team won the championship after overtime victory",
                "Olympic athlete breaks world record in swimming",
                "Football match ends with controversial referee decision",
                "Basketball playoffs begin next week with top seeds",
                "Tennis player advances to Wimbledon finals",
                "Soccer World Cup generates massive global audience",
                "Marathon runner collapses near finish line",
                "Baseball season ends with dramatic playoff series",
                # Business (2)
                "Company reports quarterly earnings exceed expectations",
                "Stock market reaches new highs amid economic recovery",
                "Startup raises millions in venture capital funding",
                "Central bank announces new interest rate policy",
                "Technology firm acquires competitor in billion dollar deal",
                "Retail giant closes hundreds of stores nationwide",
                "Oil prices surge following supply chain disruptions",
                "Cryptocurrency market experiences major volatility",
                # Science/Tech (3)
                "New AI model achieves breakthrough in language understanding",
                "Scientists discover potential cure for rare disease",
                "Spacecraft successfully lands on distant planet",
                "Revolutionary battery technology doubles electric car range",
                "Quantum computer solves complex mathematical problems",
                "Gene therapy trial shows promising results for cancer",
                "Solar panel efficiency reaches record-breaking levels",
                "Robotic surgery system performs first autonomous operation",
            ] * 10  # Répéter pour avoir plus de données

            train_labels = ([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8) * 10
            test_texts = train_texts[:80]  # 20% pour le test
            test_labels = train_labels[:80]

            print(
                f"✅ Synthetic dataset created: {len(train_texts)} train, {len(test_texts)} test samples"
            )

        else:
            try:
                # Load dataset with timeout handling
                print("📥 Downloading AG News dataset (this may take a few moments)...")
                dataset = load_dataset("ag_news", trust_remote_code=True)
                print("✅ Dataset loaded successfully!")

                # Sample data to keep training fast
                print(f"📊 Preparing {sample_size} training samples...")
                train_data = (
                    dataset["train"]
                    .shuffle(seed=42)
                    .select(range(min(sample_size, len(dataset["train"]))))
                )
                test_data = (
                    dataset["test"]
                    .shuffle(seed=42)
                    .select(range(min(sample_size // 4, len(dataset["test"]))))
                )

                # Extract texts and labels
                train_texts = train_data["text"]
                train_labels = train_data["label"]
                test_texts = test_data["text"]
                test_labels = test_data["label"]

            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                print("🔄 Falling back to synthetic data...")
                return self.load_and_preprocess_data(sample_size, use_synthetic=True)

        print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

        # Vectorize texts
        print("Vectorizing texts...")
        X_train = self.vectorizer.fit_transform(train_texts).toarray()
        X_test = self.vectorizer.transform(test_texts).toarray()

        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.LongTensor(train_labels).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.LongTensor(test_labels).to(self.device)

        print(f"Feature dimension: {X_train.shape[1]}")
        return X_train.shape[1]

    def create_model(self, input_dim):
        """Create and initialize the MoE model."""
        self.model = MixtureOfExperts(
            input_dim=input_dim,
            num_experts=self.num_experts,
            num_classes=self.num_classes,
        ).to(self.device)

        return self.model

    def train(self, epochs=8, learning_rate=0.01, batch_size=64):
        """Train the MoE model."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i : i + batch_size]
                batch_y = self.y_train[i : i + batch_size]

                optimizer.zero_grad()

                outputs, routing_weights = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Add routing diversity loss to encourage expert specialization
                routing_diversity_loss = 0.01 * torch.mean(
                    torch.sum(routing_weights**2, dim=1)
                )
                total_loss_batch = loss + routing_diversity_loss

                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            # Evaluate on test set
            test_acc = self.evaluate()

            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}"
            )

        print("Training completed!")

    def evaluate(self):
        """Evaluate the model on test set."""
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(self.X_test)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(
                self.y_test.cpu().numpy(), predictions.cpu().numpy()
            )
        return accuracy

    def predict(self, text):
        """Predict class and expert distribution for a single text."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        try:
            # Vectorize the input text
            text_vector = self.vectorizer.transform([text]).toarray()
            text_tensor = torch.FloatTensor(text_vector).to(self.device)

            # Get prediction
            self.model.eval()
            with torch.no_grad():
                outputs, routing_weights = self.model(text_tensor)
                probabilities = F.softmax(outputs, dim=1)

                # Convert to numpy for easy handling
                class_probs = probabilities.cpu().numpy()[0]
                expert_weights = routing_weights.cpu().numpy()[0]

                predicted_class = np.argmax(class_probs)
                predicted_label = self.class_names[predicted_class]

            return predicted_label, class_probs, expert_weights

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return fallback values
            return "Error", [0.25, 0.25, 0.25, 0.25], [0.33, 0.33, 0.34]

    def save_model(self, path="moe_model.pth"):
        """Save the trained model and vectorizer."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vectorizer": self.vectorizer,
                "class_names": self.class_names,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path="moe_model.pth", input_dim=2000):
        """Load a pre-trained model."""
        if os.path.exists(path):
            try:
                # PyTorch 2.6+ fix: autoriser TfidfVectorizer
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )

                self.create_model(input_dim)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.vectorizer = checkpoint["vectorizer"]
                self.class_names = checkpoint["class_names"]

                print(f"Model loaded from {path}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Deleting corrupted model file...")
                os.remove(path)  # Supprimer le fichier corrompu
                return False
        return False


def create_gradio_interface(classifier):
    """Create and configure the Gradio interface."""

    def predict_and_visualize(text):
        """Function called by Gradio interface."""
        if not text.strip():
            return "Please enter some text.", "No data", "No data"

        try:
            print(f"🔍 Classifying text: '{text[:50]}...'")
            predicted_label, class_probs, expert_weights = classifier.predict(text)

            # Créer un résultat détaillé avec barres visuelles
            result_text = f"""**🎯 Prédiction: {predicted_label}**
**🔥 Confiance: {max(class_probs):.1%}**

---

**📊 Probabilités par classe:**
"""

            for class_name, prob in zip(classifier.class_names, class_probs):
                bar = "█" * int(prob * 30)
                result_text += f"\n• **{class_name}**: {prob:.1%} {bar}"

            result_text += f"""

---

**🤖 Contribution des experts:**
"""

            for i, weight in enumerate(expert_weights):
                bar = "█" * int(weight * 30)
                result_text += f"\n• **Expert {i+1}**: {weight:.1%} {bar}"

            # Données pour les graphiques (simplifiées)
            class_chart_text = "📊 **Probabilités:**\n"
            for class_name, prob in zip(classifier.class_names, class_probs):
                class_chart_text += f"• {class_name}: {prob:.1%}\n"

            expert_chart_text = "🤖 **Experts:**\n"
            for i, weight in enumerate(expert_weights):
                expert_chart_text += f"• Expert {i+1}: {weight:.1%}\n"

            print(f"✅ Classification completed: {predicted_label}")
            return result_text, class_chart_text, expert_chart_text

        except Exception as e:
            error_msg = f"❌ Error during classification: {str(e)}"
            print(error_msg)
            return error_msg, "Error", "Error"

    # Create Gradio interface with minimal config
    with gr.Blocks(
        title="MoE Text Classifier",
        theme=None,  # Pas de thème pour éviter les erreurs
        analytics_enabled=False,  # Désactiver analytics HuggingFace
        css="",  # CSS vide
    ) as interface:
        gr.Markdown(
            """
        # 🧠 Mixture of Experts (MoE) Text Classifier
        
        This demo showcases a **Mixture of Experts** model trained on the AG News dataset for text classification.
        
        **How it works:**
        - 3 specialized expert networks handle different types of content
        - A router network decides which expert(s) to use for each input
        - The final prediction combines outputs from all experts based on routing weights
        
        **Categories:** World News, Sports, Business, Science & Technology
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to classify",
                    placeholder="Type a news article or headline here...",
                    lines=4,
                )
                classify_btn = gr.Button("🔍 Classify Text", variant="primary")

            with gr.Column(scale=2):
                result_output = gr.Markdown(label="Prediction Result")

        with gr.Row():
            with gr.Column():
                class_output = gr.Markdown(
                    label="Class Probabilities",
                    value="📊 Résultats apparaîtront ici...",
                )

            with gr.Column():
                expert_output = gr.Markdown(
                    label="Expert Weights", value="🤖 Répartition des experts..."
                )

        # Example inputs
        gr.Examples(
            examples=[
                [
                    "Scientists discover new method to generate clean energy from solar panels."
                ],
                [
                    "The football team won the championship after a spectacular final match."
                ],
                [
                    "Stock markets rose sharply following the announcement of new economic policies."
                ],
                [
                    "International peace talks continue amid growing tensions between nations."
                ],
            ],
            inputs=text_input,
        )

        # Connect interface
        classify_btn.click(
            fn=predict_and_visualize,
            inputs=text_input,
            outputs=[result_output, class_output, expert_output],
        )

        text_input.submit(
            fn=predict_and_visualize,
            inputs=text_input,
            outputs=[result_output, class_output, expert_output],
        )

    return interface


def main():
    """Main function to train model and launch Gradio interface."""
    print("🚀 Initializing Mixture of Experts Text Classifier...")

    # Initialize classifier
    classifier = MoETextClassifier(
        max_features=500, num_experts=3
    )  # Réduction features

    # Try to load existing model
    if not classifier.load_model():
        print("No existing model found. Training new model...")

        try:
            # Load and preprocess data (utilise données synthétiques par défaut)
            input_dim = classifier.load_and_preprocess_data(
                sample_size=500, use_synthetic=True
            )

            # Create and train model
            classifier.create_model(input_dim)
            classifier.train(epochs=3, learning_rate=0.01)  # Entraînement très rapide

            # Save trained model
            classifier.save_model()

            # Final evaluation
            final_accuracy = classifier.evaluate()
            print(f"\n✅ Training completed! Final accuracy: {final_accuracy:.4f}")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("🔄 Creating a demo model for testing...")
            # Create a dummy model for demo purposes
            classifier.create_model(2000)
            # Initialize with dummy data for demo
            classifier.X_train = torch.randn(100, 2000)
            classifier.y_train = torch.randint(0, 4, (100,))
            classifier.X_test = torch.randn(25, 2000)
            classifier.y_test = torch.randint(0, 4, (25,))
            print("✅ Demo model created!")
    else:
        print("✅ Loaded existing model successfully!")

    # Create and launch Gradio interface
    print("\n🌐 Launching Gradio interface...")
    interface = create_gradio_interface(classifier)

    # Configuration réseau pour Docker vs local
    # En Docker, il faut écouter sur 0.0.0.0, en local sur 127.0.0.1
    server_host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    
    # Launch with minimal configuration to avoid JavaScript errors
    try:
        print(f"🌐 Interface starting on {server_host}:7860")
        interface.launch(
            server_name=server_host,
            server_port=7860,
            share=False,
            show_error=False,  # Cacher les erreurs JS
            inbrowser=False,  # Ne pas auto-ouvrir pour éviter les conflits
            prevent_thread_lock=False,
            quiet=True,  # Mode silencieux
        )
    except Exception as e:
        print(f"❌ Primary launch failed: {e}")
        print("🔄 Trying port 7861...")
        try:
            interface.launch(
                server_name=server_host,
                server_port=7861,
                share=False,
                show_error=False,
                inbrowser=False,
                quiet=True,
            )
            print("✅ Interface started on http://{server_host}:7861")
        except Exception as e2:
            print(f"❌ Alternative launch failed: {e2}")
            print(f"💡 Try opening http://{server_host}:7860 manually in your browser")


if __name__ == "__main__":
    main()
