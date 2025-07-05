from functions import *

import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
     Το κύριο πρόγραμμα, όπου συνδυάζονται κατάλληλα οι συναρτήσεις από το αρχείο 
     'functions.py' για να παραχθεί το τελικό αποτέλεσμα 
"""

if __name__ == "__main__":
    print("\n=== Ομιλία και υπόβαθρο ===")

    while True:
        print("Διάλεξε ενέργεια:")
        print("1 - Χρήση Least Squares")
        print("2 - Χρήση MLP")
        print("3 - Έξοδος από το πρόγραμμα")

        choice = input("Εισήγαγε επιλογή: ").strip()
        if choice in {"1", "2"}:
            break
        elif choice == "3":
            print("Έξοδος...")
            exit()
        else:
            print("Μη αποδεκτή είσοδος, ξαναπροσπάθησε.\n")

    print("Φόρτωση αρχείων...")
    speech_data = load_files("speech")
    noise_data = load_files("noise")

    print("Δημιουργία συνόλου δεδομένων...")
    X, y = create_dataset(speech_data, noise_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if choice == "1":
        print("Εκπαίδευση LSQ...")
        model = LSQ_train(X_train, y_train)
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        print("Εκπαίδευση MLP...")
        model = MLP_train(X_train, y_train)
        y_pred = model.predict(X_test)

    test_audio = "S01_U04.CH4.wav"
    test_json = "S01.json"

    print(f"\nΠρόβλεψη στο αρχείο: {test_audio}")
    audio_data, sr = librosa.load(test_audio, sr=None)
    total_duration = len(audio_data) / sr

    raw_preds = predict_segments(model, audio_data, sr, is_lsq=(choice == "1"))
    smooth_preds = smooth_predictions(raw_preds)

    hop_length = int(sr * 0.010)

    pred_speech_segments = get_segments(smooth_preds, hop_length, sr, label=1)
    pred_noise_segments = get_segments(smooth_preds, hop_length, sr, label=0)


    gt_speech_intervals = get_speech_intervals(test_json)
    gt_noise_intervals = get_noise_intervals(gt_speech_intervals, total_duration)

    gt_labels = generate_ground_truth_labels(gt_speech_intervals, len(smooth_preds), hop_length, sr)
    accuracy_json = accuracy_score(gt_labels, smooth_preds) * 100
    print(f"Ακρίβεια μοντέλου: {accuracy_json:.1f}%")

    audio_filename = os.path.basename(test_audio)
    save_csv(audio_filename, pred_speech_segments, pred_noise_segments)
