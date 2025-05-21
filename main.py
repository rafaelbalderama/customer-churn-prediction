from src import feature_importance_extraction, preprocess, train_1, train_2, test

def main():
    print("Starting ML pipeline:")
    print("\n----- Preprocessing data... -----")
    preprocess.run()
    print("\n----- Training model... -----")
    train_1.run()

    # Skip the tuning phase for simplicity
    '''
    print("\n(2) Training model...")
    train_2.run()
    '''

    print("\n----- Testing model... -----")
    test.run()
    print("\n----- Extracting feature importance... -----")
    feature_importance_extraction.run()

if __name__ == "__main__":
    main()