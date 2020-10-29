// essential
#include <stdio.h>
#include <omp.h>

// random
#include <random>

// data structure
#include <vector>
#include <queue>
#include <set>
#include <bitset>

#include <string.h>
#include <string>

#include <algorithm>
#include <chrono>

#define BATCH_SIZE 10
#define MAX_N 2500000
#define TEST_COUNT 100000

int n, m;
std::vector<int> V[BATCH_SIZE];
std::vector< std::pair<int, double> > E[BATCH_SIZE][MAX_N]; // adjacency list
std::bitset<MAX_N> pi[BATCH_SIZE];

std::queue<int> que[BATCH_SIZE];

std::random_device rd[BATCH_SIZE];
std::mt19937 gen[BATCH_SIZE];
std::uniform_real_distribution<double> unif[BATCH_SIZE];

std::vector<int> seed;

long long mc_result[BATCH_SIZE];

// MC simulation function
int do_task(int i, int maxlen){
    pi[i].reset();
    // set initial seeds
    for(int j=0;j<maxlen;j++){
        pi[i].flip(seed[j]);
        que[i].push(seed[j]);
    }
    int ret = 0;
    int iter = 1;
    while(!que[i].empty()){
        int lft = (int)que[i].size();
        ret += lft;
        while(lft--){
            int now = que[i].front(); que[i].pop();
            for(auto &nxt: E[i][now]){
                if(pi[i].test(nxt.first)) continue;
                // cascading behavior
                if((unif[i])(gen[i]) < nxt.second){
                    pi[i].flip(nxt.first);
                    que[i].push(nxt.first);
                }
            }
        }
        iter++;
    }
    return ret;
}


int main(int argc, char **argv){
    omp_set_num_threads(10);
    for(int i=0;i<BATCH_SIZE;i++){
        gen[i] = std::mt19937((rd[i])());
        unif[i] = std::uniform_real_distribution<>(0.0, 1.0);
    }
    FILE *in = fopen(argv[1], "r"); 
    if(!in){
        printf("FILE NOT EXIST!\n");
        return 0;
    }
    fscanf(in, "%d%d", &n, &m); // n: |V|, m: |E|
    // generate adjacency list
    printf("%d %d\n", n, m);
    for(int i=0;i<m;i++){
        int from, to; double prob;
        fscanf(in, "%d%d%lf", &from, &to, &prob);
        for(int j=0;j<BATCH_SIZE;j++){
            E[j][from].push_back({to, prob});
        }
    }
    fclose(in);
    // input seedset (size should be 100)
    FILE *ins = fopen(argv[2], "r");
    if(!ins){
        printf("FILE NOT EXIST!\n");
        return 0;
    }
    seed.resize(100);
    for(int i=0;i<100;i++) fscanf(ins, "%d", &seed[i]);
    fclose(ins);
    for(int len=10;len<=100;len+=10){
        for(int i=0;i<BATCH_SIZE;i++){
            mc_result[i] = 0;
        }
        // run MC simulations
        for(int tc=0;tc<TEST_COUNT;tc+=BATCH_SIZE){
            int bsize = std::min(BATCH_SIZE, (TEST_COUNT-tc));
            #pragma omp parallel for
            for(int i=0;i<bsize;i++){
                mc_result[i] += do_task(i, len);
            }
        }
        // output simluation results
        long long sum = 0;
        for(int i=0;i<BATCH_SIZE;i++) sum += mc_result[i];
        printf("k=%d: average influence is %.6f\n", len, (double)sum / (double)(TEST_COUNT));
    }
    printf("\n");
    return 0;
}
