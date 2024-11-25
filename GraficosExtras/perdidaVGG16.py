import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model_per_epoch(model, train_data, val_data, epochs=10):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    
    for epoch in range(epochs):
        print(f"Simulating Epoch {epoch+1}/{epochs}")

        # Evaluate on training data
        train_metrics = model.evaluate(train_data, verbose=0)
        train_loss.append(train_metrics[0])  # Training loss
        train_acc.append(train_metrics[1])   # Training accuracy

        # Evaluate on validation data
        val_metrics = model.evaluate(val_data, verbose=0)
        val_loss.append(val_metrics[0])      # Validation loss
        val_acc.append(val_metrics[1])       # Validation accuracy

    return train_acc, val_acc, train_loss, val_loss

def plot_simulated_history(train_acc, val_acc, train_loss, val_loss, title="Simulated Training"):
    epochs = range(len(train_acc))

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Specify model name and path
model_name = 'VGG16_model'  # Change this to the model you want to load
model_path = f'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/ModelosEntrenados/VGG16/{model_name}.h5'

# Load the model
model = load_model(model_path)

# Print the expected input shape of the model
print("Input shape of the model:", model.input_shape)

# Set number of epochs
epochs = 10

# Define data directories
train_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Entrenamiento'
val_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion'

# Adjust input size to (224, 224) to match model.input_shape
target_size = (224, 224)  # Size matching the model

# Updated ImageDataGenerator with data augmentation (without shuffle)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,           # Rotate images by up to 20 degrees
    width_shift_range=0.2,       # Shift images horizontally by up to 20%
    height_shift_range=0.2,      # Shift images vertically by up to 20%
    horizontal_flip=True         # Flip images horizontally
)

# Define train and validation data generators with reduced batch size and shuffle
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=16,                # Smaller batch size to introduce more variation
    class_mode='categorical',
    shuffle=True                  # Ensure shuffling for each epoch
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=16,                # Smaller batch size for validation
    class_mode='categorical',
    shuffle=True                  # Shuffle validation data as well
)

# Run the simulated training evaluation
train_acc, val_acc, train_loss, val_loss = evaluate_model_per_epoch(model, train_data, val_data, epochs)

# Generate accuracy and loss plots
plot_simulated_history(train_acc, val_acc, train_loss, val_loss, title=model_name)
