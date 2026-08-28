import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocess.preprocessor_pipeline import PreprocessPipeline

from gui.app import main

if __name__ == "__main__":
    print("🚀 Starting Gene Expression Clustering GUI...")
    print(f"📂 Working directory: {current_dir}")

    print("✅ All files found! Starting GUI...\n")

    main()