# import numpy as np
# import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size = 64
drop_out_prob = 0.1

class MNISTClassifierSimple(torch.nn.Module):
    def __init__(self):
        super(MNISTClassifierSimple, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop_out_prob)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":

    #device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")



    full_ds = datasets.MNIST(
        root="data",
        train=True,  # Use only the training set
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    # Add data augmentation to the training dataset
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Randomly translate images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Update the training dataset to use the augmented transform
    ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform,  # Use augmented transform for training
    )

    ds_train, ds_dev = torch.utils.data.random_split(
        ds, [int(len(ds) * 0.8), int(len(ds) * 0.2)]
    )

    ds_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(ds_train)}")
    print(f"Validation dataset size: {len(ds_dev)}")
    print(f"Test dataset size: {len(ds_test)}")

    # print the shape of the first batch
    for images, labels in dl_train:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        # img = images[0].squeeze().numpy()
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Label: {labels[0].item()}")
        # plt.axis('off')
        # plt.show()
        break


    model = MNISTClassifierSimple().to(device)

    print(f"Model architecture:\n{model}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    def classification_metrics(loader, model, device) -> str:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        return f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"


    epochs = 30
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    # Early stopping parameters
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    best_epoch = 0

    # Training loop with early stopping
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in dl_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(ds_train)
        train_losses.append(epoch_loss)

        # Train metrics
        train_metrics = classification_metrics(dl_train, model, device)
        train_f1 = float(train_metrics.split(", ")[1].split(": ")[1])  # Extract F1 score
        train_f1_scores.append(train_f1)
        print(f"Train Metrics: {train_metrics}")

        # Validation metrics
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in dl_dev:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

        val_loss = val_running_loss / len(ds_dev)
        val_losses.append(val_loss)

        val_metrics = classification_metrics(dl_dev, model, device)
        val_f1 = float(val_metrics.split(", ")[1].split(": ")[1])  # Extract F1 score
        val_f1_scores.append(val_f1)
        print(f"Validation Metrics: {val_metrics}")

        # Check for early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Step scheduler
        scheduler.step()


    # Plot training and validation loss (only up to actual epochs trained)
    actual_epochs = len(train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, actual_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    # Plot training and validation F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_f1_scores, label="Train F1 Score")
    plt.plot(range(1, actual_epochs + 1), val_f1_scores, label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()
    plt.savefig("f1_score_plot.png")
    plt.close()

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = classification_metrics(dl_test, model, device)
    print(f"Test Metrics: {test_metrics}")

    # Train final model from scratch on full dataset with augmentation
    print(f"Training final model from scratch on full dataset for {best_epoch} epochs.")
    
    # Create full dataset with augmentation (consistent with training)
    full_ds_augmented = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform,  # Use the same augmented transform
    )
    
    full_train_loader_augmented = torch.utils.data.DataLoader(
        full_ds_augmented, batch_size=batch_size, shuffle=True
    )

    # Initialize final model from scratch
    final_model = MNISTClassifierSimple().to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
    final_criterion = torch.nn.CrossEntropyLoss()
    final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=20, gamma=0.1)

    for epoch in range(best_epoch):
        final_model.train()
        running_loss = 0.0
        
        for images, labels in full_train_loader_augmented:
            images, labels = images.to(device), labels.to(device)

            final_optimizer.zero_grad()
            outputs = final_model(images)
            loss = final_criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(full_ds_augmented)
        print(f"Final Training - Epoch [{epoch + 1}/{best_epoch}], Loss: {epoch_loss:.4f}")
        
        # Step the scheduler
        final_scheduler.step()

    # Save the final model
    torch.save(final_model.state_dict(), "final_mnist_classifier.pth")
    print("Final model saved as 'final_mnist_classifier.pth'")