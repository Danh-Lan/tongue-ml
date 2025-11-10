import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from data_loader import Dataset
from loss import DiceBCELoss

if __name__ == "__main__":
	TRAIN_PATH = "./data/train"
	VALID_PATH = "./data/valid"
	CHECKPOINT_PATH = "./checkpoints"

	LEARNING_RATE = 1e-4
	BATCH_SIZE = 16
	NUM_EPOCHS = 10

	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_dataset = Dataset(TRAIN_PATH)
	valid_dataset = Dataset(VALID_PATH)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

	model = UNet(in_channels=3, num_classes=1).to(device)
	criterion = DiceBCELoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	best_val_loss = float('inf')
	train_losses = []
	val_losses = []

	for epoch in tqdm(range(NUM_EPOCHS)):
		model.train()
		train_running_loss = 0.0

		for i, (images, masks) in enumerate(train_loader):
			images, masks = images.to(device), masks.to(device)
			masks.unsqueeze_(1).float()

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks)

			loss.backward()
			optimizer.step()

			train_running_loss += loss.item()

		train_loss = train_running_loss / (i + 1)
		train_losses.append(train_loss)

		model.eval()
		val_running_loss = 0.0
		with torch.no_grad():
			for i, (images, masks) in enumerate(valid_loader):
				images, masks = images.to(device), masks.to(device)
				masks.unsqueeze_(1).float()

				outputs = model(images)
				loss = criterion(outputs, masks)

				val_running_loss += loss.item()

		val_loss = val_running_loss / (i + 1)
		val_losses.append(val_loss)

		print("-"*30)
		print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
		print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
		print("-"*30)

		if val_loss < best_val_loss:
			print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
			torch.save(model.state_dict(), f"{CHECKPOINT_PATH}/unet_best_model.pth")
			best_val_loss = val_loss

		if (epoch + 1) % 5 == 0:
			print(f"Saving checkpoint for epoch {epoch+1}...")
			torch.save(model.state_dict(), f"{CHECKPOINT_PATH}/unet_epoch_{epoch+1}.pth")

	torch.save({'train_losses': train_losses, 'val_losses': val_losses}, f"{CHECKPOINT_PATH}/losses.pth")

	print("Training finished.")
	print(f"Best validation loss achieved: {best_val_loss:.4f}")
