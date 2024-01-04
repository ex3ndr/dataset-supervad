from glob import glob
import requests
import csv

labels_to_ignore = ['/m/09x0r',
                    '/m/05zppz',
                    '/m/02zsn',
                    '/m/0ytgt',
                    '/m/01h8n0', 
                    '/m/02qldy', 
                    '/m/07p6fty', 
                    '/m/07sr1lc', 
                    '/m/02rtxlg', 
                    '/m/015lz1', 
                    '/m/0l14jd', 
                    '/m/0y4f8', 
                    '/m/0z9c', 
                    '/m/07c52', 
                    '/m/06bz3']

def load_dns_noise_with_voice():

    # Download segments index
    print("Downloading unbalanced_train_segments.csv...")
    segments = requests.get("http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv")
    segments = segments.text
    ignored = {}
    for i in segments.split("\n")[3:-1]:
        l = i.split(",", maxsplit=3)
        name = l[0].strip()
        labels = l[3].strip().strip('"').split(',')
        should_ignore = False
        for j in labels:
            if j in labels_to_ignore:
                should_ignore = True
                break
        if should_ignore:
            ignored[name] = True

    # List all files
    files = glob("./dataset/source/source_dns_challenge_4/noise_fullband/*.wav")
    files = [f.split("/")[-1] for f in files]
    files = [f.split(".")[0] for f in files]

    # Filter files
    to_ignore = []
    for i in files:
        if i in ignored:
            to_ignore.append(i)

    # Add extension
    to_ignore = [f + ".wav" for f in to_ignore]

    return to_ignore

if __name__ == "__main__":
    ignored = load_dns_noise_with_voice()
    print(len(ignored))
    print(ignored[0])