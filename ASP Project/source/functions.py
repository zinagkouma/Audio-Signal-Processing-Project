import os
import csv
import json
import numpy as np
import librosa

from datetime import datetime
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier


"""
     Φόρτωση όλων των αρχείων .wav από έναν φάκελο. Για κάθε αρχείο, γίνεται ανάγνωση
     των δεδομένων ήχου και του sample rate μέσω της librosa και αυτά προστείθονται σε μια λίστα
"""
def load_files(folder_path):
    audio_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".wav"):
            path = os.path.join(folder_path, file)
            audio, sample_rate = librosa.load(path, sr=None)
            audio_files.append((audio, sample_rate))
    return audio_files


"""
     Το ηχητικό σήμα χωρίζεται σε πλαίσια μήκους 25ms με hop 10ms. Επίσης, υπολογίζεται το πλήθος των 
     δειγμάτων που αντιστοιχούν σε αυτά τα χρονικά διαστήματα και μέσω της librosa δημιουργείται ένας πίνακας
     όπου κάθε γραμμή είναι ένα πλαίσιο
"""
def frame_signal(signal, sample_rate, frame_duration=0.025, hop_duration=0.010):
    frame_len = int(sample_rate * frame_duration)
    hop_len = int(sample_rate * hop_duration)
    return librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T


"""
     Εξάγονται τα MFCC χαρακτηριστικά από κάθε πλαίσιο ήχου και επιστρέφεται ένας πίνακας 
     με τα χαρακτηριστικά αυτά
"""
def extract_features(frames, sample_rate):
    features = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=13, n_fft=len(frame), hop_length=len(frame))
        features.append(np.mean(mfcc, axis=1))
    return np.array(features)



"""
     Δημιουργία συνόλου δεδομένων για την εκπαίδευση των ταξινομητών. Κάθε ηχητικό σήμα χωρίζεται σε πλαίσια
     τα οποία ανάλογα με τα χαρακτηριστικά τους μπαίνουν σε λίστα με τις αντίστοιχες ετικέτες (0 για ομιλία,
     1 για υπόβαθρο). Παράλληλα, περιορίζεται ο αριθμός των πλαισίων από κάθε κατηγορία, ώστε ο αριθμός των δειγμάτων 
     να μην είναι πολύ μεγάλος
"""
def create_dataset(speech_data, noise_data, max_speech_frames=30000, max_noise_frames=2700):
    features, labels = [], []
    for signal, sr in speech_data:
        frames = frame_signal(signal, sr)
        if len(frames) > max_speech_frames:
            continue

        features.extend(extract_features(frames, sr))
        labels.extend([1] * len(frames))
    for signal, sr in noise_data:
        frames = frame_signal(signal, sr)
        if len(frames) > max_noise_frames:
            continue
        features.extend(extract_features(frames, sr))
        labels.extend([0] * len(frames))
    return np.array(features), np.array(labels)



"""
     Εκπαιδεύονται τα μοντέλα Least Squares και MLP αντίστοιχα, ώστε να κάνουν τη ταξινόμηση
"""
def LSQ_train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def MLP_train(X, y):
    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500)
    model.fit(X, y)
    return model



"""
     Το ηχητικό σήμα χωρίζεται σε πλαίσια και για το καθένα βγαίνουν προβλέψεις. 
     Αν χρησιμοποιείται το μοντέλο Least Squares, οι τιμές μετατρέπονται  σε 0 ή 1 
     με βάση το όριο 0.5. Στην συνέχεια, οι προβλέψεις εξομαλύνονται
"""
def predict_segments(model, signal, sr, is_lsq=False):
    frames = frame_signal(signal, sr)
    features = extract_features(frames, sr)
    predictions = model.predict(features)
    if is_lsq:
        predictions = (predictions > 0.5).astype(int)
    return predictions

def smooth_predictions(predictions, kernel_size=5):
    return medfilt(predictions, kernel_size=kernel_size)



"""
     Με βάση τη σειρά των προβλέψεων, εντοπίζει τα συνεχόμενα κομμάτια που
    ανήκουν σε μια συγκεκριμένη κατηγορία. Μετατρέπει τις θέσεις των frames σε χρονικά διαστήματα (seconds)
    και βρίσκει πού ξεκινά και πού τελειώνει κάθε διάστημα. Το αποτέλεσμα είναι μια λίστα με χρονικά διαστήματα τύπου (start, end)
"""
def get_segments(predictions, hop_size, sr, label=1):
    segments = []
    segment_start = None
    for index, value in enumerate(predictions):
        timestamp = index * hop_size / sr
        if value == label and segment_start is None:
            segment_start = timestamp
        elif value != label and segment_start is not None:
            segments.append((segment_start, timestamp))
            segment_start = None
    if segment_start is not None:
        segments.append((segment_start, len(predictions) * hop_size / sr))
    return segments




"""
      Διαχειρίζονται τα ground truth διαστήματα ομιλίας και υποβάθρου από ένα αρχείο  μεταγραφής JSON.
      Γίνεται ανάγνωση στο αρχείο και τα χρονικά strings μετατρέπονται σε δευτερόλεπτα. Έπειτα, υπολογίζονται
      τα ενδιάμεσα διαστήματα που δεν περιέχουν ομιλία, άρα είναι υπόβαθρο. 
"""
def get_speech_intervals(json_file):
    with open(json_file, "r") as f:
        annotations = json.load(f)
    speech_intervals = []
    for item in annotations:
        start = timestamp_to_seconds(item["start_time"])
        end = timestamp_to_seconds(item["end_time"])
        speech_intervals.append((start, end))
    return speech_intervals

def get_noise_intervals(speech_intervals, total_duration):
    noise_intervals = []
    previous_end = 0.0
    for start, end in sorted(speech_intervals):
        if start > previous_end:
            noise_intervals.append((previous_end, start))
        previous_end = end
    if previous_end < total_duration:
        noise_intervals.append((previous_end, total_duration))
    return noise_intervals

def timestamp_to_seconds(ts):
    t = datetime.strptime(ts, "%H:%M:%S.%f")
    return t.minute * 60 + t.second + t.microsecond / 1e6



"""
      Δημιουργία πίνακα ετικετών ground truth για κάθε πλαίσιο του ήχου. Λαμβάνονται τα χρονικά διαστήματα της ομιλίας
      και η αρχή και το τέλος κάθε διαστήματος μετατρέπονται σε δείκτες πίνακα. Έτσι, το αντίστοιχο τμήμα του πίνακα γεμίζει με 1 
      (ομιλία), ενώ όλα τα υπόλοιπα πλαίσια παραμένουν στο 0 (υπόβαθρο)
"""
def generate_ground_truth_labels(speech_intervals, num_frames, hop_size, sr):
    labels = np.zeros(num_frames, dtype=int)
    for start, end in speech_intervals:
        start_idx = int(start * sr / hop_size)
        end_idx = int(end * sr / hop_size)
        labels[start_idx:end_idx] = 1
    return labels


"""
     Τα αποτελέσματα της πρόβλεψης αποθηκεύονται σε ένα αρχείο .csv, με την ζητούμενη διάταξη
"""
def save_csv(filename, speech_segments, noise_segments, csv_path="results.csv"):
    combined = [
        [filename, round(start, 2), round(end, 2), label]
        for (start, end), label in zip(speech_segments + noise_segments, ["foreground"] * len(speech_segments) + ["background"] * len(noise_segments))
    ]
    combined.sort(key=lambda row: row[1])
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Audiofile", "start", "end", "class"])
        writer.writerows(combined)
    print(f"\nΤα αποτελέσματα αποθηκεύτηκαν στο {csv_path}")
