import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from data_loader import Dataset

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

	model = UNet(in_channels=1, num_classes=2).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	for epoch in tqdm(range(NUM_EPOCHS)):
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

		train_loss = train_running_loss / (i + 1)

		model.eval()
		val_running_loss = 0.0
		with torch.no_grad():
			for i, (images, masks) in enumerate(valid_loader):
				images, masks = images.to(device), masks.to(device)
				
				outputs = model(images)
				loss = criterion(outputs, masks)

				val_running_loss += loss.item()

		val_loss = val_running_loss / (i + 1)
		
		print("-"*30)
		print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
		print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
		print("-"*30)
	
	torch.save(model.state_dict(), f"{CHECKPOINT_PATH}/unet_epoch_{epoch+1}.pth")
