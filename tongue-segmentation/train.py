import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from data_loader import Dataset
from loss import DiceBCELoss

def train(train_pth, valid_pth, checkpoint_pth, learning_rate, batch_size, num_epochs, device):
	train_dataset = Dataset(train_pth)
	valid_dataset = Dataset(valid_pth)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

	model = UNet(in_channels=3, num_classes=1).to(device)
	criterion = DiceBCELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	best_val_loss = float('inf')
	train_losses = []
	val_losses = []

	for epoch in tqdm(range(num_epochs)):
		model.train()
		train_running_loss = 0.0

		for i, (images, masks) in enumerate(train_loader):
			images, masks = images.to(device), masks.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks)

			loss.backward()
			optimizer.step()

			train_running_loss += loss.item()

		train_loss = train_running_loss / len(train_loader)
		train_losses.append(train_loss)

		model.eval()
		val_running_loss = 0.0
		with torch.no_grad():
			for i, (images, masks) in enumerate(valid_loader):
				images, masks = images.to(device), masks.to(device)

				outputs = model(images)
				loss = criterion(outputs, masks)

				val_running_loss += loss.item()

		val_loss = val_running_loss / len(valid_loader)
		val_losses.append(val_loss)

		print("-"*30)
		print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
		print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
		print("-"*30)

		if val_loss < best_val_loss:
			print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")

			checkpoint = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': val_loss
			}

			torch.save(checkpoint, f"{checkpoint_pth}/unet_best_model.pth")

			best_val_loss = val_loss

	torch.save({'train_losses': train_losses, 'val_losses': val_losses}, f"{checkpoint_pth}/losses.pth")

	print("Training finished.")
	print(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
	TRAIN_PATH = "./data/train"
	VALID_PATH = "./data/valid"
	CHECKPOINT_PATH = "./checkpoints"

	LEARNING_RATE = 1e-4
	BATCH_SIZE = 16
	NUM_EPOCHS = 10

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train(TRAIN_PATH, VALID_PATH, CHECKPOINT_PATH, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, device)

