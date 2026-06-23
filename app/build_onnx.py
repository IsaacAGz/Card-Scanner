from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoImageProcessor

model_id = "facebook/dinov2-base"
save_dir = "onnx_dinov2"

# Download and automatically convert the architecture to ONNX format
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
processor = AutoImageProcessor.from_pretrained(model_id)

# Save the assets safely to disk storage
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print("DINOv2 successfully converted and compiled to ONNX format!")