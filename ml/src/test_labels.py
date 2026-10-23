from datasets.labels import DISEASE_LABELS, NUM_CLASSES

print("Disease Labels:")
for i, disease in enumerate(DISEASE_LABELS):
    print(f"{i+1}. {disease}")

print(f"\nTotal Classes: {NUM_CLASSES}")
