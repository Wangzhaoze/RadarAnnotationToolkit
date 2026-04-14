from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Initialize semantic video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="sam3.pt", half=True, save=True)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Track concepts using text prompts
results = predictor(source="path/to/video.mp4", text=["person", "bicycle"], stream=True)

# Process results
for r in results:
    r.show()  # Display frame with tracked objects

# Alternative: Track with bounding box prompts
results = predictor(
    source="path/to/video.mp4",
    bboxes=[[864, 383, 975, 620], [705, 229, 782, 402]],
    labels=[1, 1],  # Positive labels
    stream=True,
)