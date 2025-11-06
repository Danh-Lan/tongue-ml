import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

class Dataset:
	def __init__(self, path):
		# roboflow format has images and masks in the same folder
		self.images = []
		self.masks = []

		for filename in os.listdir(path):
			if filename.endswith(".jpg") and not filename.endswith("_mask.png"):
				image_path = os.path.join(path, filename)
				self.images.append(image_path)

				mask_filename = filename.replace(".jpg", "_mask.png")
				mask_path = os.path.join(path, mask_filename)

				if os.path.exists(mask_path):
					self.masks.append(mask_path)
				else:
					print(f"Warning: Mask not found for image: {filename}")

		if len(self.images) != len(self.masks):
			print("Warning: Mismatch between the number of images and masks.")

		self.image_transform = transforms.Compose([
			transforms.Resize((256, 256)), # divisible by 16
			transforms.ToTensor(),
		])

		self.mask_transform = transforms.Compose([
			transforms.Resize((256, 256), interpolation=Image.NEAREST),
		])

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = Image.open(self.images[idx]).convert("L")
		mask = Image.open(self.masks[idx]).convert("L")

		image = self.image_transform(image)
		mask = self.mask_transform(mask)
		mask = F.pil_to_tensor(mask).squeeze(0).long()
		
		return image, mask


if __name__ == "__main__":
	image_dir = "./data/test"
	
	dataset = Dataset(image_dir)
	print(f"Total image-mask pairs in {image_dir}: {len(dataset)}")

	image, mask = dataset[0]
	print("Image Tensor:", image)
	print("Mask Tensor:", mask)

	