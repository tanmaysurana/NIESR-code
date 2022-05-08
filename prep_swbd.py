import os
import librosa
import pickle
import argparse

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--set", required=True)
    parser.add_argument("--ltr", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    dataset_dict = {}

    vocab_list = [    
        '|',
        "'",
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z',
        '<UNK>',
        '<EOS>',
        '<PAD>',
        '<BOS>'
    ]
    vocab_dict = {vocab_list[i] : i for i in range(len(vocab_list))}

    targets = open(args.ltr, "r").readlines()
    speakers = open(args.speaker, "r").readlines()

    with open(args.tsv, "r") as tsv:
        root = next(tsv).strip()
        for index, line in enumerate(tqdm(tsv, total=len(targets))):
            fpath = line.split()[0]
            fname = os.path.basename(fpath)[:-4]
            if fname.startswith('sw02005'): continue
            wav, sr = librosa.load(os.path.join(root, fpath), sr=16_000)
            wav = librosa.util.normalize(wav)
            feat = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=40)
            feat = feat.reshape(feat.shape[::-1])
            target_list = targets[index].split()
            target_list[-1] = '<EOS>'
            tkn_ids = [vocab_dict[t] for t in target_list]
            spk_ids = [int(s) for s in speakers[index].split()]
            env_ids = [0]
            transcript = targets[index]
            dataset_dict[fname] = {
                'feature': feat,
                'token_ids': tkn_ids,
                'speaker_ids': spk_ids,
                'env_ids': env_ids,
                'Transcript': transcript
            }
    
    dataset_dict_file = open(os.path.join(args.output_dir, args.set) + '.p', "wb")
    pickle.dump(dataset_dict, dataset_dict_file)
    if args.set == 'train':
        vocab_dict_file = open(os.path.join(args.output_dir, 'vocab_dict') + '.p', "wb")
        pickle.dump(vocab_dict, vocab_dict_file)
        non_lang_dict_file = open(os.path.join(args.output_dir, 'non_lang_syms') + '.p', "wb")
        pickle.dump({}, non_lang_dict_file)


# train
# python3.8 prep_swbd.py /home/tanmay/swbd/data/train.tsv --set train --ltr /home/tanmay/swbd/data/train2_unk.ltr --speaker /home/tanmay/swbd/data/train.speaker --output-dir /home/tanmay/swbd/niesr

# valid
# python3.8 prep_swbd.py /home/tanmay/swbd/data/valid.tsv --set dev --ltr /home/tanmay/swbd/data/valid2_unk.ltr --speaker /home/tanmay/swbd/data/valid.speaker --output-dir /home/tanmay/swbd/niesr

# test
# python3.8 prep_swbd.py /home/tanmay/swbd/data/valid.tsv --set test --ltr /home/tanmay/swbd/data/valid2_unk.ltr --speaker /home/tanmay/swbd/data/valid.speaker --output-dir /home/tanmay/swbd/niesr
