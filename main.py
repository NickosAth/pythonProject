import os
import random
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = "speech_commands"  # Update with the actual path to the digit dataset
SAMPLE_RATE = 16000
N_MFCC = 13
N_MFCC_COEFF = 7
PITCH_THRESHOLD = 100  # Adjust the threshold based on your specific requirements


# Step 2: Background vs. Foreground Classification
def extract_mfcc_features(segment):
    mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc[:N_MFCC_COEFF, :]


def prepare_dataset():
    X = []
    y = []

    # Traverse the dataset folders
    for label in os.listdir(DATA_PATH):
        if label.isdigit():  # Consider only digit labels
            label_path = os.path.join(DATA_PATH, label)
            if os.path.isdir(label_path):
                # Get a random sample of audio files within each label folder
                audio_files = os.listdir(label_path)
                random.shuffle(audio_files)
                selected_files = audio_files[:random.randint(5, 10)]

                for filename in selected_files:
                    filepath = os.path.join(label_path, filename)
                    if filepath.endswith(".wav"):
                        try:
                            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
                            mfcc_features = extract_mfcc_features(audio)

                            # Pad or truncate the MFCC features to a fixed length
                            if mfcc_features.shape[1] < N_MFCC_COEFF:
                                mfcc_features = np.pad(mfcc_features, ((0, 0), (0, N_MFCC_COEFF - mfcc_features.shape[1])), mode='constant')
                            else:
                                mfcc_features = mfcc_features[:, :N_MFCC_COEFF]

                            X.append(mfcc_features)
                            y.append(label)
                        except Exception as e:
                            print(f"Error processing file: {filepath} - {str(e)}")

    return np.array(X), np.array(y)


# Step 3: Train the Background vs. Foreground Classifier
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(len(X), -1), y, test_size=0.2, random_state=42)
    classifier = SVC()
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier Accuracy: {accuracy}")

    return classifier


# Step 4: Fundamental Frequency Calculation
def estimate_fundamental_frequency(segment):
    # Calculate the autocorrelation of the segment
    autocorr = np.correlate(segment, segment, mode='full')

    # Find the first peak in the autocorrelation that exceeds the threshold
    pitch_period = np.argmax(autocorr > PITCH_THRESHOLD)

    # Calculate the fundamental frequency based on the pitch period
    fundamental_frequency = SAMPLE_RATE / pitch_period

    return fundamental_frequency


# ASR System
def asr_system():
    X, y = prepare_dataset()
    classifier = train_classifier(X, y)

    for audio_features, label in zip(X, y):
        is_foreground = classifier.predict(audio_features.reshape(1, -1))

        if is_foreground:
            fundamental_frequency = estimate_fundamental_frequency(audio_features)
            print(f"Estimated fundamental frequency for digit {label}: {fundamental_frequency}")


# Run the ASR system
asr_system()
