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

//initial distribution
double pi[50] = {0.0l};

//transition matrix
double transition[50][50];

//emission matrix
map<string, double> emission[50];

//set of all words
set<string> dictionary;

void preprocess_training() {
    //add all words to dictionary
    //label the tags
    //count tags
    fstream training(training_path);
    string word, tag;
    while (training >> word) {
        dictionary.insert(word);
        training >> tag;
        training >> tag;
        if (id.find(tag) == id.end()) {
            tags[tag_count] = tag;
            id[tag] = tag_count;
            tag_count++;
        }
        token_count++;
    }
    training.close();
}

void process_training() {
    fstream training(training_path);
    bool first = true;
    string word, last_tag, tag;
    while (training >> word) {
        training >> tag;
        training >> tag;
        //count emission
        emission[id[tag]][word]++;
        if (first) {
            //count initial distribution
            pi[id[tag]]++;
            first = false;
        } else {
            //count transition
            transition[id[last_tag]][id[tag]]++;
        }
        last_tag = tag;

        if (word == ".")
            first = true;
    }
    training.close();
}


void normalize_model() {
    //normalize pi
    double denom = 0.0;
    for (int i = 0; i < tag_count; ++i) {
        denom += pi[i];
    }
    for (int i = 0; i < tag_count; ++i) {
        pi[i] /= denom;
    }
    //normalize transition
    for (int i = 0; i < tag_count; ++i) {
        denom = 0.0;
        for (int j = 0; j < tag_count; ++j) {
            denom += transition[i][j];
        }
        for (int j = 0; j < tag_count; ++j) {
            transition[i][j] /= denom;
        }
    }
    //normalize emission
    for (int i = 0; i < tag_count; ++i) {
        denom = 0.0;
        for (auto it = emission[i].begin(); it != emission[i].end(); it++) {
            denom += it->second;
        }
        for (auto it = emission[i].begin(); it != emission[i].end(); it++) {
            it->second /= denom;
        }
    }
}

void preprocess_test() {
    fstream test(untagged_path);
    string word;
    vector<string> new_word;
    while (test >> word) {
        if (dictionary.find(word) == dictionary.end()) {
            //unknown word found
            new_word.push_back(word);
            dictionary.insert(word);
        }
    }
    for (int i = 0; i < tag_count; ++i) {
        for (string str : new_word) {
            emission[i][str] = 0.0;
        }
        for (auto it = emission[i].begin(); it != emission[i].end(); it++) {
            it->second += 0.00001;
        }
    }
    //normalize emission
    for (int i = 0; i < tag_count; ++i) {
        double sum = 0.0;
        for (auto it = emission[i].begin(); it != emission[i].end(); it++) {
            sum += it->second;
        }
        for (auto it = emission[i].begin(); it != emission[i].end(); it++) {
            it->second /= sum;
        }
    }
    test.close();
}

int viterbi(vector<string> &word, vector<string> &tag) {
    int K = tag_count, L = word.size();
    double v[L][K];
    int bt[L][K];
    for (int i = 0; i < K; ++i) {
        v[0][i] = log(pi[i]) + log(emission[i][word[0]]);
    }
    for (int i = 1; i < L; ++i) {
        for (int j = 0; j < K; ++j) {
            v[i][j] = log(emission[j][word[i]]) + v[i - 1][0] + log(transition[0][j]);
            bt[i][j] = 0;
            for (int k = 0; k < K; ++k) {
                if (v[i][j] < log(emission[j][word[i]]) + v[i - 1][k] + log(transition[k][j])) {
                    v[i][j] = log(emission[j][word[i]]) + v[i - 1][k] + log(transition[k][j]);
                    bt[i][j] = k;
                }
            }
        }
    }
    vector<int> z(L);
    z[L - 1] = 0;
    for (int i = 0; i < K; ++i) {
        if (v[L - 1][i] > v[L - 1][z[L - 1]]) z[L - 1] = i;
    }
    for (int i = L - 2; i >= 0; --i) {
        z[i] = bt[i + 1][z[i + 1]];
    }
    int correct = 0;
    for (int i = 0; i < L; ++i) {
        if (tag[i] == tags[z[i]]) correct++;
    }
    return correct;
}

void running_test() {
    fstream test(tagged_path);
    string word, tag;
    vector<string> sword, stag;
    int total=0, correct=0;
    while (test >> word) {
        test >> tag;
        test >> tag;
        //compute emission
        sword.push_back(word);
        stag.push_back(tag);
        if (word == ".") {
            total += sword.size();
            correct += viterbi(sword, stag);
            sword.clear();
            stag.clear();
        }
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
    normalize_model();
    cout << "model normalized" << endl;
    preprocess_test();
    cout << "test preprocessed" << endl;
    running_test();

    return 0;
}
