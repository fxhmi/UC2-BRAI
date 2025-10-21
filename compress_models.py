import joblib
import os

models_dir = 'models'

for filename in os.listdir(models_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(models_dir, filename)
        
        # Load model
        model = joblib.load(filepath)
        
        # Save with compression
        compressed_path = filepath.replace('.pkl', '_compressed.pkl')
        joblib.dump(model, compressed_path, compress=('gzip', 3))
        
        # Check size
        original_size = os.path.getsize(filepath) / (1024 * 1024)
        compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
        
        print(f"{filename}: {original_size:.2f}MB → {compressed_size:.2f}MB")
        
        # Replace original
        os.remove(filepath)
        os.rename(compressed_path, filepath)

print("\n✅ All models compressed!")
