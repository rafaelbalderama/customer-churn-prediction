from src import preprocess, train_1, train_2, test, feature_visualization

def main():
    print("Starting ML pipeline:")
    print("\nPreprocessing data...")
    preprocess.run()
    print("\n(1) Training model...")
    train_1.run()
    print("\n(2) Training model...")
    train_2.run()
    print("\nTesting model...")
    test.run()
    print("\nExtracting feature importance...")
    feature_visualization.run()

if __name__ == "__main__":
    main()