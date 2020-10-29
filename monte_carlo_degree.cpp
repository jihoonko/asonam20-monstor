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

#define BATCH_SIZE 20
#define MAX_N 20000
#define TEST_COUNT 10000
#define MIN_SEED_SIZE 1
#define MAX_ITER 100

int n, m;
std::vector<int> V[BATCH_SIZE];
std::vector< std::pair<int, double> > E[BATCH_SIZE][MAX_N]; // adjacency list
std::bitset<MAX_N> pi[BATCH_SIZE][MAX_ITER+1];
int tot[MAX_ITER+1][MAX_N];

std::queue<int> que[BATCH_SIZE];

std::random_device rd[BATCH_SIZE];
std::mt19937 gen[BATCH_SIZE];
std::uniform_real_distribution<double> unif[BATCH_SIZE];

int limit_iter;

std::set<int> pset;
long long mc_result[BATCH_SIZE];

std::vector<int> EV;
int upd[MAX_N];

// MC simulation function
int do_task(int i, int real_seed_size){
	// set initial seeds
	pi[i][0].reset();
	for(int j=0;j<real_seed_size;j++){
		pi[i][0].flip(V[i][j]);
		que[i].push(V[i][j]);
	}
	int ret = 0;
	int iter = 1;
	while(!que[i].empty() && iter <= limit_iter){
		pi[i][iter] = pi[i][iter-1];
		int lft = (int)que[i].size();
		ret += lft;
		while(lft--){
			int now = que[i].front(); que[i].pop();
			for(auto &nxt: E[i][now]){
				if(pi[i][iter].test(nxt.first)) continue;
                // cascading behavior
				if((unif[i])(gen[i]) < nxt.second){
					pi[i][iter].flip(nxt.first);
					que[i].push(nxt.first);
				}
			}
		}
		iter++;
	}
    // When MC simulation is not finished...
	if(!que[i].empty()){
		while(!que[i].empty()) que[i].pop();
		return -10000000;
	}
	for(int j=iter;j<=limit_iter;j++){
		pi[i][j] = pi[i][iter-1];
	}
	return ret;
}


int main(int argc, char **argv){
	omp_set_num_threads(4);
    // initialization for generating random numbers
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
	
	printf("%d %d\n", n, m);
	V[0].resize(n);
	for(int i=0;i<m;i++){
		int from, to; double prob;
		fscanf(in, "%d%d%lf", &from, &to, &prob);
		for(int j=0;j<BATCH_SIZE;j++){
			E[j][from].push_back({to, prob});
		}
		EV.push_back(from);
	}
	fclose(in);

	int len_path = strlen(argv[1]);
	std::string input_path(argv[1]);
	char output_path[100] = "raw_data/";
	input_path.copy(output_path+9, len_path-11, 7);
	printf("output_path: %s\n", output_path);

	for(int i=0;i<n;i++) upd[i] = -1;

	int turn = 2000;
	while(turn < 4000){
		++turn;
		// uniformly random
		/*
		{
			auto it = V[0].begin();
			std::shuffle(it, V[0].end(), gen[0]);
		}
		*/
		// out-degree propotional sampling
		{
			auto it = V[0].begin();
			int idx = 0;
			std::shuffle(EV.begin(), EV.end(), gen[1]);
			for(auto &v: EV){
				if(upd[v] < turn){
					upd[v] = turn;
					V[0][idx++] = v;
					it++;
				}
			}
			for(int v=0;v<n;v++){
				if(upd[v] < turn){
					upd[v] = turn;
					V[0][idx++] = v;
				}
			}
			std::shuffle(it, V[0].end(), gen[1]);
		}
		for(int i=1;i<BATCH_SIZE;i++){
			V[i] = V[0];
		}
		limit_iter = std::min(n, MAX_ITER);

		for(int i=0;i<=limit_iter;i++){
			for(int j=0;j<n;j++){
				tot[i][j] = 0;
			}
		}
		for(int i=0;i<BATCH_SIZE;i++){
			mc_result[i] = 0;
		}
        // determine size of seedset (<= 2% of total nodes)
		int seed_size = (((unif[0])(gen[0])) * ((0.02 * n) - MIN_SEED_SIZE)) + MIN_SEED_SIZE;
		
        for(int tc=0;tc<TEST_COUNT;tc+=BATCH_SIZE){
			int bsize = std::min(BATCH_SIZE, (TEST_COUNT-tc));
			#pragma omp parallel for
			for(int i=0;i<bsize;i++){
                // run MC simulations
				int ret = do_task(i, seed_size);
				if(ret < 0 || mc_result[i] < 0) mc_result[i] = -10000000;
				else mc_result[i] += ret;
			}
			
			for(int i=0;i<bsize;i++){
				if(mc_result[i] < 0){
					printf("ERROR!\n");
					return 0;
				}
				for(int ii=0;ii<=limit_iter;ii++){
					#pragma omp parallel for
					for(int jj=0;jj<n;jj++){
						tot[ii][jj] += pi[i][ii].test(jj);
					}
				}
			}
		}
        // output simluation results
		long long sum = 0;
		for(int i=0;i<BATCH_SIZE;i++) sum += mc_result[i];
		sprintf(output_path + (len_path - 2), "/%d.txt", turn);
		printf("Generating %s... (average influence: %f)\n", output_path, sum / (double)TEST_COUNT);
		FILE *out = fopen(output_path, "w");
		for(int i=0;i<=limit_iter;i++){
			for(int j=0;j<n;j++){
				fprintf(out, "%.4f ", tot[i][j] / (double)TEST_COUNT);
			}
			fprintf(out, "\n");
			fflush(out);
		}
		fclose(out);
	}
	return 0;
}
