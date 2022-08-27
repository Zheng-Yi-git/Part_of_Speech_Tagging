#include <bits/stdc++.h>

using namespace std;

string training_path = "D:\\Codefield\\CODE_Cpp\\postagging-main\\training.txt";
string tagged_path = "D:\\Codefield\\CODE_Cpp\\postagging-main\\testdata_tagged.txt";
string untagged_path = "D:\\Codefield\\CODE_Cpp\\postagging-main\\testdata_untagged.txt";

// # of word in training data
int token_count = 0;
// # of different tags in training data
int tag_count = 0;
// maps tag_id to tag
string tags[50];
// maps tag to its id
map<string, int> id;

//frequency of each tag
int fr[50] = {0};

//the most frequent tag
int best_tag = 0;

//w_i->t_j count
map<string, vector<int>> emission;
//most frequent tag for each word
map<string, int> best_emission;

void preprocess_training() {
    fstream training(training_path);
    string word, tag;
    while (training >> word) {
        if (emission.find(word) == emission.end()) {
            emission[word] = vector<int>(50, 0);
        }
        training >> tag;
        training >> tag;
        if (id.find(tag) == id.end()) {
            tags[tag_count] = tag;
            id[tag] = tag_count;
            tag_count++;
        }
        fr[id[tag]]++;
        token_count++;
    }
    for (int i = 0; i < tag_count; ++i) {
        if (fr[best_tag] < fr[i]) best_tag = i;
    }
    training.close();
}

void process_training() {
    fstream training(training_path);
    string word, tag;
    while (training >> word) {
        training >> tag;
        training >> tag;
        //count emission
        emission[word][id[tag]]++;
    }
    for (auto it = emission.begin(); it != emission.end(); it++) {
        int best = 0;
        for (int i = 0; i < it->second.size(); ++i) {
            if (it->second[best] < it->second[i]) best = i;
        }
        best_emission[it->first] = best;
    }
    training.close();
}

void running_test() {
    fstream test(tagged_path);
    string word, tag;
    int total = 0, correct = 0;
    while (test >> word) {
        test >> tag;
        test >> tag;
        if (best_emission.find(word) != best_emission.end()) {
            //word seen before
            correct += (best_emission[word] == id[tag] ? 1 : 0);
        } else {
            //new word
            correct += (best_tag == id[tag] ? 1 : 0);
        }
        total++;
    }
    test.close();
    cout << "total: " << total << endl;
    cout << "correct: " << correct << endl;
    cout << "percent: " << 1.0 * correct / total << endl;
}

int main() {
    preprocess_training();
    cout << "training preprocessed" << endl;
    process_training();
    cout << "training completed" << endl;
    running_test();

    return 0;
}
