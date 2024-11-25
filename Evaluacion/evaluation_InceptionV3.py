import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Define class names
class_names = ['No caries', 'Mild caries', 'Moderate caries', 'Severe caries']

# Create the main Tkinter window
root = tk.Tk()
root.title("InceptionV3 Evaluation - ROC AUC and Metrics")
root.geometry("1200x800")
root.configure(bg="#f0f0f0")

# Title label
label_title = tk.Label(root, text="Evaluation Results - InceptionV3", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333")
label_title.pack(pady=10)

# Text box to display results
text_area = scrolledtext.ScrolledText(root, width=70, height=10, font=("Arial", 12), bg="#e0e0e0", fg="#333")
text_area.pack(pady=10)

# Frame for plots (confusion matrix and ROC curve)
frame_plots = tk.Frame(root, bg="#f0f0f0")
frame_plots.pack(pady=10)

# Configure image preprocessing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Prepare the test set (InceptionV3 expects 299x299 images)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion',
    target_size=(299, 299),  # InceptionV3 expects 299x299 images
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the InceptionV3 model
inceptionv3_model = load_model('C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/ModelosEntrenados/InceptionV3/inceptionv3_model.h5')

# Function to calculate and display metrics
def evaluate_model():
    text_area.insert(tk.END, "Evaluating InceptionV3...\n\n")

    # Predictions
    predictions = inceptionv3_model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=-1)

    # True labels
    true_labels = test_generator.classes

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes, average='weighted')
    recall = recall_score(true_labels, predicted_classes, average='weighted')
    f1 = f1_score(true_labels, predicted_classes, average='weighted')
    confusion_matrix_data = confusion_matrix(true_labels, predicted_classes)

    # Display results in the text box with styling
    text_area.insert(tk.END, f"Accuracy: {accuracy:.4f}\n")
    text_area.insert(tk.END, f"Precision: {precision:.4f}\n")
    text_area.insert(tk.END, f"Recall: {recall:.4f}\n")
    text_area.insert(tk.END, f"F1-score: {f1:.4f}\n")
    text_area.insert(tk.END, "-"*50 + "\n")

    # Display confusion matrix
    display_confusion_matrix(confusion_matrix_data)

    # Binarize true labels for ROC AUC calculation
    binarized_labels = label_binarize(true_labels, classes=[0, 1, 2, 3])

    # Calculate ROC AUC for each class and get ROC curves
    fpr = {}
    tpr = {}
    roc_auc_values = {}

    # For each class, calculate the ROC curve
    for i in range(binarized_labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(binarized_labels[:, i], predictions[:, i])
        roc_auc_values[i] = auc(fpr[i], tpr[i])

    # Display the ROC AUC plot
    display_roc_curve(fpr, tpr, roc_auc_values)

# Function to display the confusion matrix using Matplotlib and Seaborn
def display_confusion_matrix(confusion_matrix_data):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black')
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predictions", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    ax.set_xticklabels(class_names)  # Set class names on x-axis
    ax.set_yticklabels(class_names, rotation=0)  # Set class names on y-axis
    plt.tight_layout()

    # Embed the Matplotlib figure in the Tkinter interface
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame_plots)  # Now within the frame
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)  # Display on the left

# Function to display the ROC curve in Matplotlib
def display_roc_curve(fpr, tpr, roc_auc_values):
    plt.figure(figsize=(6, 5))
    colors = ['blue', 'green', 'red', 'orange']  # Different colors for each class
    for i, color in zip(range(len(fpr)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc_values[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Class', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Embed the Matplotlib figure in the Tkinter interface
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame_plots)  # Now within the frame
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)  # Display on the right

# Button to start evaluation
btn_evaluate = tk.Button(root, text="Evaluate InceptionV3", command=evaluate_model, font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
btn_evaluate.pack(pady=10)

# Run the application
root.mainloop()
